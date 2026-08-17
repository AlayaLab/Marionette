#!/usr/bin/env python3
"""Lazy-loading dataset for the per-segment dir produced by build_dataset_lazy.py.

Window index is built from per-seg lengths in metadata.npz (no files opened up front);
__getitem__ loads only the one seg_*.npz it needs, normalizes raw-780 with metadata
mean/std, slices to 276D. Drop-in replacement for train_v16.V16Dataset output.
"""
import os
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from src.train_gpt_continuous import slice_to_pos276


class LazyV16Dataset(Dataset):
    def __init__(self, out_dir, seg_idx, seg_files, seg_len, mean, std,
                 m_ids, n_ids, window_size=512, stride=256,
                 use_terrain=False, terr_patch_dim=0, terr_n_key=0,
                 use_hp=False):
        self.dir = out_dir
        self.window_size = window_size
        # HP conditioning: per-frame (T,3) = [monster_hp%, npc_hp%, valid], appended after terrain.
        self.use_hp = use_hp
        self._hp = {}
        # Terrain conditioning (B): load per-seg .terr.npz aligned to each window and return a
        # single bundled tensor terr = concat[patch(NN), clear(K), H_local(K), R_row1(3),
        # contact(K), valid(1)] so the batch stays a flat tuple of tensors.
        self.use_terrain = use_terrain
        self.NN = terr_patch_dim
        self.K = terr_n_key
        self.terr_dim = (terr_patch_dim + 3 * terr_n_key + 3 + 1) if use_terrain else 0
        self._terr = {}
        self.files = seg_files
        self.mean = mean.astype(np.float32)
        self.std = std.astype(np.float32)
        # raw game id -> contiguous vocab slot (slot k+1 = ids[k]); 0 = padding/invalid.
        # LUT (vectorized) instead of per-element dict lookups.
        def build_lut(ids):
            mx = int(max(ids)) if len(ids) else 0
            lut = np.zeros(mx + 1, np.int64)
            for i, v in enumerate(ids):
                lut[int(v)] = i + 1
            return lut
        self.m_lut = build_lut(m_ids); self.n_lut = build_lut(n_ids)
        # per-seg caches (filled lazily, per DataLoader worker) — avoids reopening
        # .npy/.aux.npz every window (networked FS file-open was ~169 ms/item).
        self._feat = {}; self._aux = {}
        self.samples = []  # (seg_i, start, length)
        for i in seg_idx:
            L = int(seg_len[i])
            if L <= window_size:
                self.samples.append((i, 0, L))
            else:
                for s in range(0, L - window_size + 1, stride):
                    self.samples.append((i, s, window_size))
                if (L - window_size) % stride != 0:
                    self.samples.append((i, L - window_size, window_size))

    def __len__(self):
        return len(self.samples)

    def _feat_mm(self, seg_i, base):
        m = self._feat.get(seg_i)
        if m is None:
            m = np.load(os.path.join(self.dir, base + ".npy"), mmap_mode="r")
            self._feat[seg_i] = m
        return m

    def _aux_arr(self, seg_i, base):
        a = self._aux.get(seg_i)
        if a is None:
            z = np.load(os.path.join(self.dir, base + ".aux.npz"))
            a = (np.asarray(z["action_m"]), np.asarray(z["action_n"]), np.asarray(z["progress"], np.float32))
            self._aux[seg_i] = a
        return a

    def _hp_arr(self, seg_i, base):
        """Cached per-seg HP bundle (T, 3): [monster_hp%, npc_hp%, valid]."""
        h = self._hp.get(seg_i)
        if h is None:
            z = np.load(os.path.join(self.dir, base + ".aux.npz"))
            hp = np.asarray(z["hp"], np.float32)              # (T, 2)
            valid = np.asarray(z["hp_valid"], np.float32)[:, None]  # (T, 1)
            h = np.concatenate([hp, valid], axis=1)           # (T, 3)
            self._hp[seg_i] = h
        return h

    def _terr_arr(self, seg_i, base):
        """Cached per-seg terrain bundle (T, terr_dim): [patch, clear, H_local, R_row1, contact, valid]."""
        t = self._terr.get(seg_i)
        if t is None:
            z = np.load(os.path.join(self.dir, base + ".terr.npz"))
            parts = [np.asarray(z["patch"], np.float32),
                     np.asarray(z["clear"], np.float32),
                     np.asarray(z["H_local"], np.float32),
                     np.asarray(z["R_row1"], np.float32),
                     np.asarray(z["contact"], np.float32),
                     np.asarray(z["valid"], np.float32)[:, None]]
            t = np.concatenate(parts, axis=1)
            self._terr[seg_i] = t
        return t

    def __getitem__(self, idx):
        seg_i, start, length = self.samples[idx]
        base = self.files[seg_i][:-4]  # strip .npz
        feat_mm = self._feat_mm(seg_i, base)                            # cached mmap handle
        feat = np.asarray(feat_mm[start:start + length], np.float32)     # window only
        feat = (feat - self.mean) / (self.std + 1e-8)                    # normalize
        frames = slice_to_pos276(feat).astype(np.float32)               # -> 276
        am_all, an_all, pg_all = self._aux_arr(seg_i, base)             # cached in RAM
        am_raw = np.clip(am_all[start:start + length], 0, len(self.m_lut) - 1)
        an_raw = np.clip(an_all[start:start + length], 0, len(self.n_lut) - 1)
        am = self.m_lut[am_raw]                                          # vectorized remap
        an = self.n_lut[an_raw]
        pg = pg_all[start:start + length]
        terr = self._terr_arr(seg_i, base)[start:start + length] if self.use_terrain else None
        hp = self._hp_arr(seg_i, base)[start:start + length] if self.use_hp else None

        if length < self.window_size:
            pad = self.window_size - length
            frames = np.concatenate([frames, np.zeros((pad, frames.shape[-1]), np.float32)], 0)
            am = np.concatenate([am, np.zeros(pad, np.int64)])
            an = np.concatenate([an, np.zeros(pad, np.int64)])
            pg = np.concatenate([pg, np.zeros((pad, 2), np.float32)], 0)
            mask = np.zeros(self.window_size, np.float32); mask[:length] = 1.0
            if terr is not None:
                terr = np.concatenate([terr, np.zeros((pad, self.terr_dim), np.float32)], 0)
            if hp is not None:
                hp = np.concatenate([hp, np.zeros((pad, 3), np.float32)], 0)  # pad valid=0
        else:
            mask = np.ones(self.window_size, np.float32)

        out = [torch.from_numpy(frames), torch.from_numpy(am).long(),
               torch.from_numpy(an).long(), torch.from_numpy(pg), torch.from_numpy(mask)]
        if self.use_terrain:
            out.append(torch.from_numpy(np.ascontiguousarray(terr)))
        if self.use_hp:
            out.append(torch.from_numpy(np.ascontiguousarray(hp)))
        return tuple(out)


def load_data_lazy(config):
    """Mirror of train_v16.load_data, but for a lazy per-segment dir (metadata.npz)."""
    out_dir = config["data"]["motion_dir"]
    meta = np.load(os.path.join(out_dir, "metadata.npz"), allow_pickle=True)
    seg_files = [str(x) for x in meta["seg_files"]]
    seg_len = meta["seg_len"]
    mean, std = meta["mean"], meta["std"]
    if bool(config["data"].get("zero_center_root", False)):
        # Root-delta channels are normalized WITHOUT mean-centering. The dataset mean of the
        # local-frame root delta is dominated by forward locomotion (npc z: +0.105 m/f), so a
        # regression that shrinks toward normalized-zero decodes to a constant forward velocity
        # (systematic root drift in idle rollouts). With mean=0 here, normalized-zero = true rest.
        mean = np.array(mean, dtype=np.float64).copy()
        mean[0:3] = 0.0        # monster root_delta
        mean[486:489] = 0.0    # npc root_delta
        print("[lazy] zero_center_root ON: root-delta channels use mean=0 normalization")
    m_ids, n_ids = meta["m_ids"], meta["n_ids"]
    m_vocab = int(meta["m_action_vocab_size"]); n_vocab = int(meta["n_action_vocab_size"])
    num_seg = int(meta["num_segments"])

    idx = list(range(num_seg))
    rng = random.Random(config["train"].get("seed", 42)); rng.shuffle(idx)
    n_val = max(1, int(num_seg * config["data"]["val_ratio"]))
    val_idx, train_idx = idx[:n_val], idx[n_val:]
    ws = config["data"]["window_size"]; st = config["data"]["window_stride"]

    # Terrain conditioning (B): read shared params from metadata.terr.npz if data.terrain enabled.
    use_terrain = bool(config["data"].get("terrain", False))
    NN = K = 0
    if use_terrain:
        tm = np.load(os.path.join(out_dir, "metadata.terr.npz"), allow_pickle=True)
        NN = int(tm["patch_dim"]); K = int(tm["n_key"])
        print(f"[lazy] terrain ON: patch_dim={NN} n_key={K} (terr_dim={NN+3*K+4})")
    use_hp = bool(config["data"].get("hp", False))
    if use_hp:
        print("[lazy] HP ON: per-frame [monster_hp%, npc_hp%, valid] from aux.npz")
    tk = dict(use_terrain=use_terrain, terr_patch_dim=NN, terr_n_key=K, use_hp=use_hp)

    train_ds = LazyV16Dataset(out_dir, train_idx, seg_files, seg_len, mean, std, m_ids, n_ids, ws, st, **tk)
    val_ds = LazyV16Dataset(out_dir, val_idx, seg_files, seg_len, mean, std, m_ids, n_ids, ws, st, **tk)
    print(f"[lazy] {num_seg} segs ({len(train_idx)} train / {len(val_idx)} val), "
          f"{len(train_ds)} train windows / {len(val_ds)} val windows, vocab m={m_vocab} n={n_vocab}")
    return train_ds, val_ds, m_vocab, n_vocab
