#!/usr/bin/env python3
"""
Pose + terrain renderer using moderngl (OpenGL via Xvfb + Mesa llvmpipe).

输出 multi-channel encoded "pose video" (Plan D 风格) 给 v2v 模型用:
  - R channel: terrain/joint height (世界 Y 坐标，归一化到相机 Y±N 区间)
  - G channel: entity ID (0=terrain, 85=npc-body, 170=monster, 255=weapon)
  - B channel: inverse depth (近=亮, 远=暗)
  - depth buffer 做 occlusion

直接 pipe 渲染帧到 ffmpeg（不写 PNG）。
"""

import argparse
import json
import math
import os
import struct
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from utils.rec_bin_utils import RecBinReader, read_bin_header

FFMPEG = '/usr/local/lib/python3.12/dist-packages/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2'
# NVENC-capable ffmpeg (n7.1.1, NVENC SDK 12.2 — matches driver 550.x; SDK 13
# from BtbN nightlies needs driver 570+). When env POSE_ENCODER=nvenc is set,
# start_ffmpeg_encoder uses this binary + hevc_nvenc. Install via setup/install_nvenc_ffmpeg.sh.
import os as _os
NVENC_FFMPEG = _os.environ.get(
    'POSE_NVENC_FFMPEG', '/root/nvenc-ffmpeg/bin/ffmpeg')
POSE_ENCODER = _os.environ.get('POSE_ENCODER', 'nvenc')  # 'libx264' | 'libx265' | 'nvenc'
# v7.6 near-plane terrain cull: terrain fragments closer to the camera than this
# many world units are dropped per-frame, so near terrain (a wall/cliff right at
# the lens) can't occlude the character. Skeleton bones are NEVER affected. The
# character sits ~5-6 units from the third-person camera, so 3.0 removes only the
# in-your-face terrain. 0 disables the cull. Override via env POSE_TERRAIN_NEAR_CULL
# or the --terrain-near-cull CLI flag. Applies to new renders only (the pipeline
# skips already-produced segments).
TERRAIN_NEAR_CULL = float(_os.environ.get('POSE_TERRAIN_NEAR_CULL', '3.0'))
# Data roots, defaulting to the copies shipped in this project so the renderer runs with no
# environment set. The upstream default pointed at a cluster path that no longer exists, which
# fails with a confusing "no terrain" rather than a missing-path error.
#   POSE_DATA_DIR    — base dir (holds filter/ and terrain/)
#   POSE_FILTER_DIR  — skeleton json dir             (default $POSE_DATA_DIR/filter)
#   POSE_TERRAIN_DIR — terrain stage_<N>/c_*.bin dir (default $POSE_DATA_DIR/terrain)
_MROOT = _os.environ.get('MARIONETTE_ROOT') or _os.path.dirname(
    _os.path.dirname(_os.path.abspath(__file__)))
DATA_DIR = _os.environ.get('POSE_DATA_DIR', _os.path.join(_MROOT, 'data'))
FILTER_DIR = _os.environ.get('POSE_FILTER_DIR', f'{DATA_DIR}/filter')
TERRAIN_DIR = _os.environ.get('POSE_TERRAIN_DIR', f'{DATA_DIR}/terrain')

W = 1280
H = 704

# Entity IDs (written to G channel)
ENT_TERRAIN = 0
ENT_PLATFORM = 32   # synthetic "platform disc" under NPC's feet (for unmapped structures)
ENT_NPC = 85
ENT_MONSTER = 170
ENT_WEAPON = 255

# v7.3 G-channel "color id" allocation (per-joint coloring for v2v training).
# Stored as G = id / 255.0.
#
# Visual contrast goal: terrain pixels have G≈0 (RGB is dominated by R=height
# + B=depth → purple/magenta family). Character/monster should sit at the
# opposite end of the G axis so the bones don't drown in the background:
#
#   id  range   category               visual effect
#     0          terrain                bg purple/red
#     5, 6       platform discs         near-bg (minor marker)
#    60..99    weapon       (40 ids)    blue-green
#   105..194   monster      (90 ids)    green
#   195..254   npc          (60 ids)    yellow-green / high contrast vs bg
#
# Bones get one unique id per child joint within their entity (stable across
# frames). The 4–6 unit gap between category ranges absorbs the chroma-
# interpolation noise from yuv444p10le compression.
COLOR_TERRAIN = 0
COLOR_PLATFORM_NPC = 5
COLOR_PLATFORM_MONSTER = 6
COLOR_WEAPON_BASE = 60       # 60..99
COLOR_WEAPON_MAX = 40
COLOR_MONSTER_BASE = 105     # 105..194
COLOR_MONSTER_MAX = 90
COLOR_NPC_BASE = 195         # 195..254
COLOR_NPC_MAX = 60


# ============================================================
# Camera reconstruction
# ============================================================
def compute_view_proj(row, near=0.1, far=10000.0):
    eye = np.array([row['camera.ours.eye.x'],
                    row['camera.ours.eye.y'],
                    row['camera.ours.eye.z']], dtype=np.float32)
    fwd = np.array([row['camera.ours.fwd.x'],
                    row['camera.ours.fwd.y'],
                    row['camera.ours.fwd.z']], dtype=np.float32)
    up = np.array([row['camera.ours.up.x'],
                   row['camera.ours.up.y'],
                   row['camera.ours.up.z']], dtype=np.float32)
    fov_deg = float(row['camera.ours.fov_deg'])
    aspect = float(row['camera.ours.aspect'])

    fn = np.linalg.norm(fwd)
    if fn < 1e-8:
        return None, eye, fov_deg, aspect
    look = -fwd / fn
    right = np.cross(look, up)
    rn = np.linalg.norm(right)
    if rn < 1e-8:
        return None, eye, fov_deg, aspect
    right = right / rn
    true_up = np.cross(right, look)

    view = np.eye(4, dtype=np.float32)
    view[0, :3] = right
    view[1, :3] = true_up
    view[2, :3] = -look
    view[0, 3] = -np.dot(right, eye)
    view[1, 3] = -np.dot(true_up, eye)
    view[2, 3] = np.dot(look, eye)

    f = 1.0 / math.tan(math.radians(fov_deg) * 0.5)
    proj = np.zeros((4, 4), dtype=np.float32)
    proj[0, 0] = f / aspect
    proj[1, 1] = f
    proj[2, 2] = (far + near) / (near - far)
    proj[2, 3] = (2 * far * near) / (near - far)
    proj[3, 2] = -1
    return (proj @ view).astype(np.float32), eye, fov_deg, aspect


# ============================================================
# Schema detection
# ============================================================
def detect_schema(row):
    """Return (schema, em_id, weapon_type, monster_prefix, npc_prefix, weapon_prefix).

    Raises a descriptive RuntimeError on an unrecognized schema OR when a required
    field is missing for the detected schema — never fail silently, so a bad/new
    rec layout surfaces loudly and can be fixed case-by-case (do NOT add a silent
    fallback here).
    """
    def req(key, schema):
        # Loud read: missing-when-expected is a data/schema bug, not a default-to-0.
        if key not in row:
            raise RuntimeError(
                f"{schema} rec missing required field {key!r}. "
                f"top-level prefixes present: "
                f"{sorted({str(k).split('.')[0] for k in row})}")
        return int(row[key])

    if 'quest_meta.em_id_int' in row:
        return ('NEW',
                req('quest_meta.em_id_int', 'NEW'),
                req('npc.list.1.weapon_type_int', 'NEW'),
                'monster.list.1.joints',
                'npc_joints.body.1',
                'npc_joints.weapon.1')
    if 'em.em.1.id_int' in row:
        return ('HYBRID',
                req('em.em.1.id_int', 'HYBRID'),
                req('NPC Data.npc.weaponType_int', 'HYBRID'),
                'em.em.1.joints',
                'NPC Joints.npc',
                'NPC Joints.weapon')
    raise RuntimeError(
        "unknown rec schema: neither 'quest_meta.em_id_int' (NEW) nor "
        "'em.em.1.id_int' (HYBRID) present. top-level prefixes: "
        f"{sorted({str(k).split('.')[0] for k in row})}")


def gather_joints(row, prefix):
    plen = len(prefix) + 1
    out: Dict[str, Dict[str, float]] = {}
    for k, v in row.items():
        if not k.startswith(prefix + '.'):
            continue
        rest = k[plen:]
        for axis in ('x', 'y', 'z'):
            sfx = f'.pos.{axis}'
            if rest.endswith(sfx):
                jname = rest[:-len(sfx)]
                out.setdefault(jname, {})[axis] = float(v)
                break
    return {j: (d['x'], d['y'], d['z'])
            for j, d in out.items() if {'x', 'y', 'z'} <= d.keys()}


def load_edges(json_path: Path):
    if not json_path.exists():
        return []
    with open(json_path) as f:
        data = json.load(f)
    edges = []
    def visit(n, p):
        nm = n.get('name')
        if p and nm:
            edges.append((p, nm))
        for c in n.get('childs') or []:
            visit(c, nm)
    visit(data, None)
    return [(p, c) for p, c in edges if p.lower() != 'root' and c.lower() != 'root']


# ============================================================
# Terrain loading (TRN2 multi-layer support; falls back to TRN1)
# ============================================================
TRN2_LAYER_DTYPE = np.dtype([
    ('h', 'f4'),
    ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
    ('hit_count', 'u2'),
    ('flags', 'u1'),
    ('pad', 'u1'),
    ('last_ts', 'u4'),
])  # 24 bytes
TRN1_CELL_DTYPE = np.dtype([
    ('h', 'f4'), ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
    ('c4', 'i4'), ('c5', 'u4'),
])  # 24 bytes


def _fill_holes_cross_chunk(chunks_dict, fill_tol: float = 2.0,
                             min_neighbors: int = 3, n_iter: int = 5) -> int:
    """Stitch all chunks of a stage into one global floor-h map; dilate floor
    into empty cells whose 8-neighbors agree on a height (spread ≤ fill_tol)
    for up to ``n_iter`` rounds; scatter synthetic floor layers back to per-chunk
    arrays as a single layer at slot 0 with hit_count=2 (so survives min_hits=1).

    Only fills cells inside a loaded chunk's grid. Cells in non-loaded chunks
    stay empty. Synthetic cells keep ny=1.0 (flat up normal).
    """
    keys = list(chunks_dict.keys())
    if not keys:
        return 0
    cx0 = min(k[0] for k in keys); cx1 = max(k[0] for k in keys)
    cy0 = min(k[1] for k in keys); cy1 = max(k[1] for k in keys)
    nC = cx1 - cx0 + 1  # x direction (cols)
    nR = cy1 - cy0 + 1  # z direction (rows)
    H = 100 * nR
    W = 100 * nC
    floor_h = np.full((H, W), np.nan, dtype=np.float64)
    chunk_id = -np.ones((H, W), dtype=np.int32)
    key_list = []
    for idx, k in enumerate(keys):
        cx, cy = k
        key_list.append(k)
        layers = chunks_dict[k]['layers']
        v = (layers['flags'].reshape(100, 100, -1) & 0x01).astype(bool)
        h = layers['h'].reshape(100, 100, -1)
        h_key = np.where(v, h, np.inf)
        floor_idx = np.argmin(h_key, axis=2)
        cell_floor = np.take_along_axis(h, floor_idx[..., None], axis=2)[..., 0]
        cell_valid = v.any(axis=2)
        cell_floor = np.where(cell_valid, cell_floor, np.nan)
        r0 = (cy - cy0) * 100
        c0 = (cx - cx0) * 100
        floor_h[r0:r0+100, c0:c0+100] = cell_floor
        chunk_id[r0:r0+100, c0:c0+100] = idx

    total = 0
    for _ in range(n_iter):
        valid = np.isfinite(floor_h)
        padded = np.pad(floor_h, 1, constant_values=np.nan)
        nbrs = np.stack([
            padded[0:H,   0:W  ], padded[0:H,   1:W+1], padded[0:H,   2:W+2],
            padded[1:H+1, 0:W  ],                       padded[1:H+1, 2:W+2],
            padded[2:H+2, 0:W  ], padded[2:H+2, 1:W+1], padded[2:H+2, 2:W+2],
        ], axis=-1)
        n_valid_n = np.isfinite(nbrs).sum(axis=-1)
        with np.errstate(all='ignore'):
            nmax = np.nanmax(nbrs, axis=-1)
            nmin = np.nanmin(nbrs, axis=-1)
            nmean = np.nanmean(nbrs, axis=-1)
        spread = nmax - nmin
        empty = ~valid & (chunk_id >= 0)
        good = empty & (n_valid_n >= min_neighbors) & (spread <= fill_tol)
        if not good.any():
            break
        floor_h = np.where(good, nmean, floor_h)
        ii, jj = np.where(good)
        total += len(ii)
        cids = chunk_id[ii, jj]
        new_h = floor_h[ii, jj].astype(np.float32)
        for cid in np.unique(cids):
            m = cids == cid
            cx, cy = key_list[cid]
            ri = ii[m] - (cy - cy0) * 100
            ci = jj[m] - (cx - cx0) * 100
            cell_idx = ri * 100 + ci
            layers = chunks_dict[(cx, cy)]['layers']
            layers['h'][cell_idx, 0] = new_h[m]
            layers['hit_count'][cell_idx, 0] = 2
            layers['ny'][cell_idx, 0] = 1.0
            layers['flags'][cell_idx, 0] = layers['flags'][cell_idx, 0] | 0x01
    return total


def _load_chunk(path: Path):
    """Return (cx, cy, layers_array, max_layers) where layers_array shape
    (cell_count, max_layers) structured (TRN2 dtype). TRN1 files are
    upgraded inline (max_layers=1, single layer per cell, valid mask from
    normal magnitude).
    """
    raw = path.read_bytes()
    magic = raw[:4]
    if magic == b'TRN2':
        version, cx, cz, chunk_m, cell_m_b, cell_count, max_layers = struct.unpack(
            '<iiiifii', raw[4:32])
        # cell_m is f32 — but we already consumed 4 bytes. Re-parse:
        # offset 4: i32 version
        # offset 8: i32 cx
        # offset 12: i32 cz
        # offset 16: i32 chunk_m
        # offset 20: f32 cell_m
        # offset 24: i32 cell_count
        # offset 28: i32 max_layers
        # `<iiiifii` reads i,i,i,i,f,i,i = 4+4+4+4+4+4+4 = 28 bytes -- right.
        if version != 2 or max_layers <= 0:
            raise ValueError(f'TRN2 bad header: ver={version}, max_layers={max_layers}')
        layers = np.frombuffer(raw[32:], dtype=TRN2_LAYER_DTYPE).reshape(
            cell_count, max_layers).copy()  # copy → writable (for hole fill)
        return cx, cz, layers, max_layers
    elif magic == b'TRN1':
        version, cx, cz, n_side, scale, count, _ = struct.unpack(
            '<iiiifii', raw[4:32])
        if version != 1:
            raise ValueError(f'TRN1 bad version: {version}')
        cells = np.frombuffer(raw[32:], dtype=TRN1_CELL_DTYPE).reshape(count)
        # Synthesize a TRN2 array with one layer per cell. Valid = normal mag > 0.5
        layers = np.zeros((count, 1), dtype=TRN2_LAYER_DTYPE)
        layers['h'][:, 0] = cells['h']
        layers['nx'][:, 0] = cells['nx']
        layers['ny'][:, 0] = cells['ny']
        layers['nz'][:, 0] = cells['nz']
        nmag = np.sqrt(cells['nx'] ** 2 + cells['ny'] ** 2 + cells['nz'] ** 2)
        layers['flags'][:, 0] = (nmag > 0.5).astype(np.uint8)  # valid bit
        return cx, cz, layers, 1
    else:
        raise ValueError(f'Unknown terrain magic in {path}: {magic!r}')


class TerrainStage:
    """Multi-layer terrain (TRN2). Stores chunks; mesh is built per-frame
    by selecting the right layer per cell based on a query (e.g., NPC.y)."""

    def __init__(self, stage_id: int, stride: int = 2):
        self.stage_id = stage_id
        self.stride = stride
        self.stage_dir = Path(f'{TERRAIN_DIR}/stage_{stage_id}')
        # chunks[(cx, cy)] = {'layers': (cell_count, max_layers) TRN2_LAYER_DTYPE,
        #                     'max_layers': int}
        self.chunks: Dict[Tuple[int, int], Dict] = {}
        all_min = all_max = None

        chunk_size = 100
        for path in sorted(self.stage_dir.glob('c_*.bin')):
            try:
                cx, cy, layers, max_layers = _load_chunk(path)
            except Exception as e:
                print(f'  skip {path.name}: {e}')
                continue
            self.chunks[(cx, cy)] = {
                'layers': layers,       # (10000, max_layers)
                'max_layers': max_layers,
            }
            # Track global height range over valid layers
            valid = (layers['flags'] & 0x01) == 1
            vh = layers['h'][valid]
            if len(vh) > 0:
                ymin, ymax = float(vh.min()), float(vh.max())
                all_min = ymin if all_min is None else min(all_min, ymin)
                all_max = ymax if all_max is None else max(all_max, ymax)

        # Cross-chunk hole fill: dilate floor into empty cells whose
        # 8-neighbors agree on a height (≤ fill_tol). Synthetic cells get
        # hit_count=2 (survives min_hits=1). Updates global h range.
        _fill_holes_cross_chunk(self.chunks, fill_tol=2.0,
                                min_neighbors=3, n_iter=5)
        for info in self.chunks.values():
            layers = info['layers']
            v = (layers['flags'] & 0x01) == 1
            vh = layers['h'][v]
            if len(vh) > 0:
                ymin, ymax = float(vh.min()), float(vh.max())
                all_min = ymin if all_min is None else min(all_min, ymin)
                all_max = ymax if all_max is None else max(all_max, ymax)

        self.height_min = all_min if all_min is not None else 0.0
        self.height_max = all_max if all_max is not None else 0.0

    @staticmethod
    def _dedupe_layers(heights: np.ndarray, hits: np.ndarray, valid: np.ndarray,
                       dedupe_tol: float = 0.5):
        """Within each cell, merge layers whose h's are within ``dedupe_tol``
        meters. Inputs are (N_cells, K). Returns (heights_out, hits_out,
        valid_out) same shape. Merged layer: weighted-avg h, summed hits.
        Trailing slots are zero-padded and marked invalid.
        """
        n_cells, K = heights.shape
        # Sort each cell by h ascending; invalid h sent to +inf so they trail.
        h_key = np.where(valid, heights, np.inf)
        order = np.argsort(h_key, axis=1)
        h_s = np.take_along_axis(heights, order, axis=1)
        v_s = np.take_along_axis(valid, order, axis=1)
        c_s = np.take_along_axis(hits, order, axis=1)

        # Group-id per slot: increments when |h - prev_h| > dedupe_tol OR slot invalid.
        # Compute consecutive diffs.
        diffs = np.diff(h_s, axis=1)                            # (n_cells, K-1)
        # A "break" between slot i and slot i+1 if either slot is invalid OR diff > tol
        breaks = (diffs > dedupe_tol) | (~v_s[:, :-1]) | (~v_s[:, 1:])
        gid = np.zeros((n_cells, K), dtype=np.int32)
        gid[:, 1:] = np.cumsum(breaks.astype(np.int32), axis=1)

        # Vectorized groupby: weighted sum per (cell, gid).
        # Mask out invalid entries.
        w = c_s.astype(np.int64) * v_s.astype(np.int64)         # weight = hits if valid else 0
        wh = w * h_s.astype(np.float64)
        # max group id per cell ≤ K-1
        # Build dense buckets shape (n_cells, K).
        sum_w = np.zeros((n_cells, K), dtype=np.int64)
        sum_wh = np.zeros((n_cells, K), dtype=np.float64)
        np.add.at(sum_w,  (np.arange(n_cells)[:, None], gid), w)
        np.add.at(sum_wh, (np.arange(n_cells)[:, None], gid), wh)
        valid_g = sum_w > 0
        h_out = np.zeros_like(sum_wh, dtype=np.float32)
        np.divide(sum_wh, sum_w, out=h_out.astype(np.float64), where=valid_g)
        # numpy out= must match dtype; redo manually
        h_out = np.where(valid_g, sum_wh / np.maximum(sum_w, 1), 0.0).astype(np.float32)
        hits_out = sum_w.astype(np.uint16)
        return h_out, hits_out, valid_g

    def build_all_layers_mesh(self,
                              min_hits: int = 1,
                              connect_tol: float = 2.0,
                              dedupe_tol: float = 0.5) -> np.ndarray:
        """Build a static triangle vertex array (N_verts, 3) by greedy
        gradient-based neighbor matching.

        For each cell `s = (r, c)` and each valid layer `h_self` in s, look at
        3 forward neighbors E=(r, c+1), S=(r+1, c), SE=(r+1, c+1). For each
        neighbor we pick its valid layer with the smallest `|h_neigh - h_self|`
        — if that minimum is ≤ `connect_tol`, the neighbor matches; else fail.
        Only if all 3 neighbors match do we emit a 4-corner quad with the
        matched heights at each corner. Single-cell layers that never connect
        get a flat 1×1m plate.

        Effect:
          * gentle gradients (slopes) connect into continuous meshes
          * cliffs / overhangs (gradient > tol) naturally break into separate
            floating plates without producing 20m vertical bridges
          * cells with multiple layers can have different layers participate
            in different quads (floor connects to ground, roof to roof) since
            matching is per-layer not per-rank
          * `min_hits` filters probe-noise layers (default ≥2 hits)
        """
        chunk_size = 100
        all_verts: List[np.ndarray] = []

        # Pre-pass: dedupe + min_hits filter each chunk independently. The
        # deduped (heights, valid) arrays are then used both by the chunk
        # itself AND by its NW neighbor as right/bottom padding (so the
        # gradient-match can span chunk boundaries — otherwise every chunk
        # boundary leaks a 1-cell-wide seam in the mesh).
        deduped: Dict[Tuple[int, int], Tuple[np.ndarray, np.ndarray]] = {}
        for (cx, cy), info in self.chunks.items():
            layers = info['layers']
            heights_flat = layers['h']                                   # (10000, K)
            hits_flat = layers['hit_count']
            valid_flat = (layers['flags'] & 0x01) == 1
            heights_flat, hits_flat, valid_flat = self._dedupe_layers(
                heights_flat, hits_flat, valid_flat, dedupe_tol=dedupe_tol)
            if min_hits > 0:
                valid_flat &= hits_flat >= min_hits
            K = heights_flat.shape[1]
            deduped[(cx, cy)] = (
                heights_flat.reshape(chunk_size, chunk_size, K),
                valid_flat.reshape(chunk_size, chunk_size, K),
            )

        for (cx, cy), (heights2d, valid2d) in deduped.items():
            K = heights2d.shape[2]
            # Pad to (101, 101, K) with neighbor data on the south (row 100),
            # east (col 100), and SE-corner (cell 100, 100).
            heights = np.zeros((chunk_size + 1, chunk_size + 1, K), dtype=heights2d.dtype)
            valid   = np.zeros((chunk_size + 1, chunk_size + 1, K), dtype=bool)
            heights[:chunk_size, :chunk_size, :] = heights2d
            valid  [:chunk_size, :chunk_size, :] = valid2d
            for nb_off, dst in [((1, 0), 'E'), ((0, 1), 'S'), ((1, 1), 'SE')]:
                nb = deduped.get((cx + nb_off[0], cy + nb_off[1]))
                if nb is None: continue
                nb_h, nb_v = nb
                cK = min(K, nb_h.shape[2])
                if dst == 'E':
                    heights[:chunk_size, chunk_size, :cK] = nb_h[:, 0, :cK]
                    valid  [:chunk_size, chunk_size, :cK] = nb_v[:, 0, :cK]
                elif dst == 'S':
                    heights[chunk_size, :chunk_size, :cK] = nb_h[0, :, :cK]
                    valid  [chunk_size, :chunk_size, :cK] = nb_v[0, :, :cK]
                else:  # SE
                    heights[chunk_size, chunk_size, :cK] = nb_h[0, 0, :cK]
                    valid  [chunk_size, chunk_size, :cK] = nb_v[0, 0, :cK]

            # Self / neighbors aligned to a 100×100 window so all 3 neighbors
            # exist (using the padded row/col for cells at chunk edge).
            h_self = heights[:-1, :-1, :]              # (100, 100, K)
            v_self = valid[:-1, :-1, :]
            h_E    = heights[:-1, 1:,  :]
            v_E    = valid[:-1, 1:,  :]
            h_S    = heights[1:,  :-1, :]
            v_S    = valid[1:,  :-1, :]
            h_SE   = heights[1:,  1:,  :]
            v_SE   = valid[1:,  1:,  :]

            def best_match(h_n, v_n):
                # h_n, v_n: (99, 99, K).  Returns (best_h, best_idx, matched)
                # all of shape (99, 99, K_self).
                diff = np.abs(h_self[..., :, None] - h_n[..., None, :])
                invalid = (~v_n[..., None, :]) | (~v_self[..., :, None])
                diff = np.where(invalid, np.inf, diff)
                best_idx = np.argmin(diff, axis=-1)
                best_diff = np.take_along_axis(
                    diff, best_idx[..., None], axis=-1)[..., 0]
                matched = best_diff <= connect_tol
                # best_h[r,c,L_self] = h_n[r,c, best_idx[r,c,L_self]]
                h_n_b = np.broadcast_to(h_n[..., None, :], diff.shape)
                best_h = np.take_along_axis(
                    h_n_b, best_idx[..., None], axis=-1)[..., 0]
                return best_h, best_idx.astype(np.int8), matched

            h_E_b,  iE,  m_E  = best_match(h_E,  v_E)
            h_S_b,  iS,  m_S  = best_match(h_S,  v_S)
            h_SE_b, iSE, m_SE = best_match(h_SE, v_SE)

            quad_ok = v_self & m_E & m_S & m_SE       # (100, 100, K)

            # Track which (cell, layer) participated in any quad. Pad-sized
            # so writes at R+1=100 / C+1=100 (neighbor edge) don't IndexError;
            # only the [:100, :100, :] slice is consulted for plate fallback.
            covered = np.zeros((chunk_size + 1, chunk_size + 1, K), dtype=bool)

            if quad_ok.any():
                R, C, L = np.where(quad_ok)
                n = R.size
                wx_l = (cx * chunk_size + C).astype(np.float32)
                wx_r = wx_l + 1.0
                wz_t = (cy * chunk_size + R).astype(np.float32)
                wz_b = wz_t + 1.0
                h_tl = h_self[R, C, L]
                h_tr = h_E_b[R, C, L]
                h_bl = h_S_b[R, C, L]
                h_br = h_SE_b[R, C, L]
                tris = np.empty((n, 6, 3), dtype=np.float32)
                tris[:, 0, 0] = wx_l; tris[:, 0, 1] = h_tl; tris[:, 0, 2] = wz_t
                tris[:, 1, 0] = wx_r; tris[:, 1, 1] = h_tr; tris[:, 1, 2] = wz_t
                tris[:, 2, 0] = wx_l; tris[:, 2, 1] = h_bl; tris[:, 2, 2] = wz_b
                tris[:, 3, 0] = wx_r; tris[:, 3, 1] = h_tr; tris[:, 3, 2] = wz_t
                tris[:, 4, 0] = wx_r; tris[:, 4, 1] = h_br; tris[:, 4, 2] = wz_b
                tris[:, 5, 0] = wx_l; tris[:, 5, 1] = h_bl; tris[:, 5, 2] = wz_b
                all_verts.append(tris.reshape(-1, 3))

                # Scatter coverage to all 4 cells / layers participating
                covered[R, C, L]                = True             # TL = self
                covered[R, C + 1, iE[R, C, L]]  = True             # TR = E
                covered[R + 1, C, iS[R, C, L]]  = True             # BL = S
                covered[R + 1, C + 1, iSE[R, C, L]] = True         # BR = SE

            # Flat plates for valid (cell, layer) that never connected.
            # Only emit plates for cells this chunk OWNS ([:100, :100]); cells
            # in the padded edge belong to neighbor chunks and they will emit
            # their own plates.
            plate_mask_3d = valid[:chunk_size, :chunk_size, :] & \
                            ~covered[:chunk_size, :chunk_size, :]
            if plate_mask_3d.any():
                R, C, L = np.where(plate_mask_3d)
                n = R.size
                wx_l = (cx * chunk_size + C).astype(np.float32)
                wx_r = wx_l + 1.0
                wz_t = (cy * chunk_size + R).astype(np.float32)
                wz_b = wz_t + 1.0
                h_q = heights[R, C, L]
                tris = np.empty((n, 6, 3), dtype=np.float32)
                tris[:, 0, 0] = wx_l; tris[:, 0, 1] = h_q; tris[:, 0, 2] = wz_t
                tris[:, 1, 0] = wx_r; tris[:, 1, 1] = h_q; tris[:, 1, 2] = wz_t
                tris[:, 2, 0] = wx_l; tris[:, 2, 1] = h_q; tris[:, 2, 2] = wz_b
                tris[:, 3, 0] = wx_r; tris[:, 3, 1] = h_q; tris[:, 3, 2] = wz_t
                tris[:, 4, 0] = wx_r; tris[:, 4, 1] = h_q; tris[:, 4, 2] = wz_b
                tris[:, 5, 0] = wx_l; tris[:, 5, 1] = h_q; tris[:, 5, 2] = wz_b
                all_verts.append(tris.reshape(-1, 3))

        if not all_verts:
            return np.zeros((0, 3), dtype=np.float32)
        return np.concatenate(all_verts, axis=0)


# ============================================================
# GL renderer
# ============================================================
TERRAIN_VS = """
#version 330
uniform mat4 u_vp;
in vec3 in_pos;
in float in_color_id;   // [0..1] = id / 255.0
out float v_height;
out float v_depth;
out float v_color_id;
void main() {
    vec4 clip = u_vp * vec4(in_pos, 1.0);
    gl_Position = clip;
    v_height = in_pos.y;
    v_depth = clip.w;
    v_color_id = in_color_id;
}
"""

TERRAIN_FS = """
#version 330
in float v_height;
in float v_depth;
in float v_color_id;
out vec4 fragColor;
uniform float u_height_lo;
uniform float u_height_hi;
uniform float u_depth_near;
uniform float u_depth_far;
uniform float u_terrain_near_cull;  // >0: discard terrain (non-bone) frags with v_depth < this
// Encoding v7.4: terrain and bone live in DISJOINT R ranges so the
// pose.mp4 visually separates them (otherwise both end up in the same
// magenta hue when R encodes the same height for both).
//
//   v_color_id  ≤ 0.15  → terrain branch:   R = 0.5 + 0.5 * h_norm   ∈ [0.5, 1.0]
//   v_color_id  >  0.15 → bone    branch:   R = 0.0 + 0.2 * h_norm   ∈ [0.0, 0.2]
//   G                      = v_color_id                       (categorical)
//   B                      = inv_depth                        (always)
//
// Δhue between terrain (magenta-dominant) and bone (cyan/green-dominant)
// is now ≥ 0.3 in R and ≥ 0.235 in G across the whole image.
//
// Model recovery for height:
//   if   G > 0.15 → h_norm = R / 0.2
//   else          → h_norm = (R - 0.5) / 0.5
void main() {
    float is_bone = step(0.15, v_color_id);
    // v7.6 near-plane terrain cull: terrain fragments (is_bone==0) closer to the
    // camera than u_terrain_near_cull are dropped this frame so near terrain can't
    // occlude the character. v_depth = clip.w = metric distance in front of the
    // camera. Bones (is_bone==1) are never culled. <=0 disables.
    if (is_bone < 0.5 && u_terrain_near_cull > 0.0 && v_depth < u_terrain_near_cull) {
        discard;
    }
    float h_norm = clamp((v_height - u_height_lo) /
                         max(u_height_hi - u_height_lo, 1e-6), 0.0, 1.0);
    float d_norm = 1.0 - clamp((v_depth - u_depth_near) /
                               max(u_depth_far - u_depth_near, 1e-6), 0.0, 1.0);
    float r_out = mix(0.5 + 0.5 * h_norm,    // terrain
                      0.0 + 0.2 * h_norm,    // bone (capsule / disc)
                      is_bone);
    fragColor = vec4(r_out, v_color_id, d_norm, 1.0);
}
"""


class GLRenderer:
    def __init__(self, width=W, height=H, backend: Optional[str] = None):
        """backend: 'egl' (NVIDIA hardware), 'x11' (Xvfb+Mesa llvmpipe), or None to auto-pick."""
        import moderngl
        self.width = width
        self.height = height
        if backend is None:
            backend = os.environ.get('MGL_BACKEND', 'egl')
        last_err = None
        self.ctx = None
        tried = []
        for cand in [backend, 'egl', 'x11']:
            if cand is None or cand in tried:
                continue
            tried.append(cand)
            try:
                self.ctx = moderngl.create_standalone_context(
                    require=330, backend=cand)
                self.backend = cand
                break
            except Exception as e:
                last_err = e
        if self.ctx is None:
            raise RuntimeError(f'failed to create GL context (tried {tried}): {last_err}')
        self.terrain_prog = self.ctx.program(
            vertex_shader=TERRAIN_VS, fragment_shader=TERRAIN_FS)
        self.skel_prog = self.terrain_prog  # same shader
        # 16-bit float per channel — gives the headroom for 10-bit mp4 encoding
        # without quantization loss in the fbo. Read back as float16 → uint16.
        self.color_tex = self.ctx.texture((width, height), 4, dtype='f2')
        self.depth_tex = self.ctx.depth_texture((width, height))
        self.fbo = self.ctx.framebuffer(color_attachments=[self.color_tex],
                                        depth_attachment=self.depth_tex)
        self.ctx.enable(moderngl.DEPTH_TEST)
        self.ctx.depth_func = '<'
        self._terrain_buf = None
        self._terrain_vao = None

    def bind_terrain(self, terrain: TerrainStage):
        """Set the current terrain stage and immediately build + upload
        the full multi-layer static mesh. The mesh does not change across
        frames; depth buffer handles multi-layer occlusion."""
        self._terrain = terrain
        self._height_lo = terrain.height_min
        self._height_hi = terrain.height_max
        if self._terrain_vao is not None:
            self._terrain_vao.release()
            self._terrain_vao = None
        if self._terrain_buf is not None:
            self._terrain_buf.release()
            self._terrain_buf = None
        verts = terrain.build_all_layers_mesh()
        self._terrain_n_verts = int(verts.shape[0])
        if verts.shape[0] == 0:
            return
        self._terrain_buf = self.ctx.buffer(verts.tobytes())
        # Terrain pixels colored as COLOR_TERRAIN (=0). One float per vertex.
        terrain_color = np.full(self._terrain_n_verts,
                                COLOR_TERRAIN / 255.0, dtype=np.float32)
        self._terrain_color_buf = self.ctx.buffer(terrain_color.tobytes())
        self._terrain_vao = self.ctx.vertex_array(
            self.terrain_prog,
            [(self._terrain_buf, '3f', 'in_pos'),
             (self._terrain_color_buf, '1f', 'in_color_id')])

    def render_bone_capsules(self, lines: np.ndarray, color_ids: np.ndarray,
                             radius: float = 0.025, n_seg: int = 8) -> None:
        """Render bones as 3D triangle-cylinder capsules (side wall only) for
        proper per-pixel z-buffer occlusion. ``lines`` shape (N_bones, 2, 3)
        gives endpoint xyz per bone. ``color_ids`` shape (N_bones,) is the
        per-bone G-channel value already normalized to [0, 1] (the caller
        should pass e.g. (base_id + child_idx) / 255.0). All 6 * n_seg
        vertices of a bone share that bone's color id.
        """
        import moderngl
        if lines is None or len(lines) == 0:
            return
        lines = np.asarray(lines, dtype=np.float32)
        color_ids = np.asarray(color_ids, dtype=np.float32)
        if lines.ndim != 3 or lines.shape[1] != 2 or len(color_ids) != len(lines):
            return
        p1 = lines[:, 0, :]                                 # (B, 3)
        p2 = lines[:, 1, :]
        axis = p2 - p1
        L = np.linalg.norm(axis, axis=1, keepdims=True)
        ok = (L[:, 0] > 1e-6)
        if not ok.any():
            return
        p1 = p1[ok]; p2 = p2[ok]; axis = axis[ok]; L = L[ok]
        color_ids = color_ids[ok]
        axis_n = axis / L                                   # (B, 3)
        # Pick a perpendicular direction stably (avoid axis‖up degeneracy)
        up_world = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        is_vert = np.abs(axis_n[:, 1]) > 0.99
        alt = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        perp_a = np.where(is_vert[:, None],
                          np.cross(axis_n, alt),
                          np.cross(axis_n, up_world))
        perp_a /= np.linalg.norm(perp_a, axis=1, keepdims=True).clip(1e-6)
        perp_b = np.cross(axis_n, perp_a)                   # already unit
        # ring of n_seg offsets around the axis
        angles = np.linspace(0, 2 * math.pi, n_seg, endpoint=False)
        cos_a = np.cos(angles).astype(np.float32).reshape(1, n_seg, 1)
        sin_a = np.sin(angles).astype(np.float32).reshape(1, n_seg, 1)
        offsets = radius * (cos_a * perp_a[:, None, :]
                            + sin_a * perp_b[:, None, :])    # (B, n_seg, 3)
        ring1 = p1[:, None, :] + offsets
        ring2 = p2[:, None, :] + offsets
        next_k = (np.arange(n_seg) + 1) % n_seg
        nB = ring1.shape[0]
        tris = np.empty((nB, n_seg, 6, 3), dtype=np.float32)
        tris[:, :, 0, :] = ring1
        tris[:, :, 1, :] = ring1[:, next_k, :]
        tris[:, :, 2, :] = ring2[:, next_k, :]
        tris[:, :, 3, :] = ring1
        tris[:, :, 4, :] = ring2[:, next_k, :]
        tris[:, :, 5, :] = ring2
        verts = tris.reshape(-1, 3)
        # Parallel color buffer: same bone color replicated for all 6*n_seg verts.
        col = np.broadcast_to(color_ids[:, None, None],
                              (nB, n_seg, 6)).reshape(-1).astype(np.float32)
        buf = self.ctx.buffer(verts.tobytes())
        cbuf = self.ctx.buffer(col.tobytes())
        vao = self.ctx.vertex_array(
            self.terrain_prog,
            [(buf, '3f', 'in_pos'), (cbuf, '1f', 'in_color_id')])
        vao.render(mode=moderngl.TRIANGLES)
        buf.release(); cbuf.release(); vao.release()

    def render_disc(self, center_world: Tuple[float, float, float],
                    radius: float = 1.0, n_segments: int = 16,
                    color_id: int = COLOR_PLATFORM_NPC):
        """Draw a small horizontal disc (triangle fan) at the given world point.
        Useful for showing 'unmapped platform' under floating NPCs.
        ``color_id`` in [0, 255] integer; stored as G channel value / 255.0."""
        import moderngl
        cx, cy, cz = center_world
        verts = [(cx, cy, cz)]
        for i in range(n_segments + 1):
            a = 2 * math.pi * i / n_segments
            verts.append((cx + radius * math.cos(a), cy, cz + radius * math.sin(a)))
        arr = np.array(verts, dtype=np.float32)
        col = np.full(len(arr), color_id / 255.0, dtype=np.float32)
        buf = self.ctx.buffer(arr.tobytes())
        cbuf = self.ctx.buffer(col.tobytes())
        vao = self.ctx.vertex_array(
            self.terrain_prog,
            [(buf, '3f', 'in_pos'), (cbuf, '1f', 'in_color_id')])
        vao.render(mode=moderngl.TRIANGLE_FAN)
        buf.release(); cbuf.release(); vao.release()

    def render_frame(self, vp: np.ndarray,
                     skel_lines: Optional[List[Tuple[str, np.ndarray]]] = None,
                     depth_near: float = 5.0, depth_far: float = 80.0,
                     height_band=None,
                     npc_foot: Optional[Tuple[float, float, float]] = None,
                     monster_foot: Optional[Tuple[float, float, float]] = None,
                     terrain_near_cull: float = 0.0) -> np.ndarray:
        import moderngl
        self.fbo.use()
        self.fbo.clear(0.0, 0.0, 0.0, 1.0, depth=1.0)
        if height_band is None:
            hl, hh = self._height_lo, self._height_hi
        else:
            hl, hh = height_band

        prog = self.terrain_prog
        vp_t = vp.T.astype('f4').tobytes()
        prog['u_vp'].write(vp_t)
        prog['u_height_lo'].value = hl
        prog['u_height_hi'].value = hh
        prog['u_depth_near'].value = depth_near
        prog['u_depth_far'].value = depth_far
        prog['u_terrain_near_cull'].value = terrain_near_cull

        # Terrain (color id is baked into the VAO's color buffer = 0)
        if self._terrain_vao is not None:
            self._terrain_vao.render(mode=moderngl.TRIANGLES)

        # Synthetic platform discs (visualize NPC/monster contact when they
        # appear to float above the recorded terrain — e.g., on buildings/cliffs
        # whose floor isn't captured in the heightmap).
        if npc_foot is not None:
            self.render_disc(npc_foot, radius=1.2,
                             color_id=COLOR_PLATFORM_NPC)
        if monster_foot is not None:
            self.render_disc(monster_foot, radius=2.0,
                             color_id=COLOR_PLATFORM_MONSTER)

        # Skeleton bones as 3D triangle capsules (per-pixel z, robust occlusion).
        # skel_lines: list of (ent_name, lines_or_pair). Callers may supply
        # either bare ``lines (N, 2, 3)`` (legacy) or the v7.2 tuple
        # ``(lines, color_ids)``. Bare-lines fallback colors all bones with
        # the entity's base id.
        if skel_lines:
            radius_by_ent = {'monster': 0.05, 'npc': 0.025, 'weapon': 0.015}
            base_by_ent = {'npc': COLOR_NPC_BASE,
                           'monster': COLOR_MONSTER_BASE,
                           'weapon': COLOR_WEAPON_BASE}
            for ent_name, payload in skel_lines:
                if payload is None:
                    continue
                if isinstance(payload, tuple):
                    lines, color_ids = payload
                else:
                    lines = payload
                    nb = len(lines) if lines is not None else 0
                    if nb == 0:
                        continue
                    base = base_by_ent.get(ent_name, 0)
                    color_ids = np.full(nb, base / 255.0, dtype=np.float32)
                if lines is None or len(lines) == 0:
                    continue
                self.render_bone_capsules(
                    lines, color_ids=color_ids,
                    radius=radius_by_ent.get(ent_name, 0.025),
                    n_seg=8)

        # Read 16-bit float framebuffer → uint16 (suitable for 10/12-bit yuv
        # pipe into ffmpeg via rgb48le).
        # Fragment shader guarantees output ∈ [0, 1] (clamps on every
        # channel) so we can skip np.clip — which was a 43 ms/frame
        # hot spot on float16 inputs (numpy falls back to a slow path).
        # Fused float32 promotion + scale + uint16 cast = ~15 ms/frame.
        data = self.fbo.read(components=3, dtype='f2')
        img = np.frombuffer(data, dtype=np.float16).reshape(
            self.height, self.width, 3)
        img16 = (img.astype(np.float32) * 65535.0).astype(np.uint16)
        # OpenGL Y-flip
        return np.ascontiguousarray(img16[::-1])

    def release(self):
        try:
            if self._terrain_vao is not None:
                self._terrain_vao.release()
                self._terrain_buf.release()
            self.fbo.release()
            self.color_tex.release()
            self.depth_tex.release()
            self.ctx.release()
        except Exception:
            pass


# ============================================================
# Skeleton lines builder
# ============================================================
def build_skel_lines(row, prefix, edges,
                     color_base: int = 0, color_max: int = 256
                     ) -> Tuple[np.ndarray, np.ndarray]:
    """Return (lines, color_ids).
      lines     (N, 2, 3) bone endpoints in world coords
      color_ids (N,)     float, = (color_base + child_joint_idx) / 255.0

    child_joint_idx is enumerated in edge-traversal order so the same joint
    name maps to the same id frame-after-frame. Capped at color_max to keep
    different entity ranges from colliding.
    """
    joints = gather_joints(row, prefix)
    # Stable joint index map: visit edges in fixed order, assign first-seen idx.
    joint_to_idx: Dict[str, int] = {}
    for p, c in edges:
        for n in (p, c):
            if n not in joint_to_idx:
                joint_to_idx[n] = len(joint_to_idx)
    out_pts: List[List[Tuple[float, float, float]]] = []
    out_ids: List[float] = []
    for p, c in edges:
        if p not in joints or c not in joints:
            continue
        out_pts.append([joints[p], joints[c]])
        idx = min(joint_to_idx[c], color_max - 1)
        out_ids.append((color_base + idx) / 255.0)
    if not out_pts:
        return (np.zeros((0, 2, 3), dtype=np.float32),
                np.zeros((0,), dtype=np.float32))
    return (np.array(out_pts, dtype=np.float32),
            np.array(out_ids, dtype=np.float32))


# ============================================================
# ffmpeg pipe encoder
# ============================================================
def start_ffmpeg_encoder(output_path: str, fps: int = 30,
                         width: int = W, height: int = H) -> subprocess.Popen:
    # Input is 16-bit RGB per channel (rgb48le). Three backends:
    #  - libx264 (default): h264 high444 10-bit yuv444p10le (CPU-bound)
    #  - libx265: hevc Main 4:4:4 10 yuv444p10le (CPU-bound). Decodes bit-for-bit
    #    to the SAME codec+pix_fmt as the nvenc output, so processed_v2 stays
    #    codec-uniform. Use this on the 4090 cluster: hevc_nvenc dies on spawn
    #    (BrokenPipe) once the 4090's NVENC session pool is exhausted under
    #    --workers>1, leaving 0-byte pose.mp4 while the job still exits 0.
    #  - nvenc: hevc_nvenc 10-bit yuv444p16le on GPU (CPU-free). Note the codec
    #    change h264→hevc — h264_nvenc on Ada has no 4:4:4 10-bit path.
    if POSE_ENCODER == 'nvenc':
        cmd = [NVENC_FFMPEG, '-hide_banner', '-loglevel', 'error', '-y',
               '-f', 'rawvideo', '-pix_fmt', 'rgb48le',
               '-s', f'{width}x{height}', '-r', str(fps),
               '-i', '-',
               '-c:v', 'hevc_nvenc', '-preset', 'p4', '-cq', '20',
               '-pix_fmt', 'yuv444p16le',
               output_path]
    elif POSE_ENCODER in ('libx265', 'x265', 'software-hevc'):
        cmd = [FFMPEG, '-hide_banner', '-loglevel', 'error', '-y',
               '-f', 'rawvideo', '-pix_fmt', 'rgb48le',
               '-s', f'{width}x{height}', '-r', str(fps),
               '-i', '-',
               '-c:v', 'libx265', '-preset', 'ultrafast',
               '-x265-params', 'log-level=error',
               '-crf', '20',
               '-pix_fmt', 'yuv444p10le',
               output_path]
    else:
        cmd = [FFMPEG, '-hide_banner', '-loglevel', 'error', '-y',
               '-f', 'rawvideo', '-pix_fmt', 'rgb48le',
               '-s', f'{width}x{height}', '-r', str(fps),
               '-i', '-',
               '-c:v', 'libx264', '-profile:v', 'high444',
               '-preset', 'ultrafast', '-crf', '20',
               '-pix_fmt', 'yuv444p10le',
               '-threads', '1',
               output_path]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE,
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.PIPE)


# ============================================================
# Per-task render
# ============================================================
def render_task(uuid_csv: Path, rec_path: Path,
                output_mp4: Path,
                renderer: GLRenderer,
                terrain: TerrainStage,
                depth_near: float = 5.0,
                depth_far: float = 80.0,
                fps: int = 30,
                terrain_near_cull: float = TERRAIN_NEAR_CULL,
                ) -> Tuple[int, int]:
    """Render one task's pose_terrain.mp4 (one frame per UUID CSV entry)."""
    import csv as csvlib
    indices = []
    with open(uuid_csv, 'r') as f:
        reader = csvlib.DictReader(f)
        for row in reader:
            indices.append(int(row['rec_row_idx']))
    if not indices:
        return 0, 1

    output_mp4.parent.mkdir(parents=True, exist_ok=True)
    enc = start_ffmpeg_encoder(str(output_mp4), fps=fps)

    schema_info = None
    em_edges = npc_edges = wp_edges = None
    n_rendered = 0
    with RecBinReader(rec_path) as reader:
        for ri in indices:
            reader.seek_to_row(ri)
            row = reader.read_row()
            if schema_info is None:
                schema_info = detect_schema(row)
                em_edges = load_edges(Path(FILTER_DIR) / f'em_{schema_info[1]}.json')
                npc_edges = load_edges(Path(FILTER_DIR) / 'npc.json')
                wp_edges = load_edges(Path(FILTER_DIR) / f'wp_{schema_info[2]}.json')

            vp, eye, _, _ = compute_view_proj(row)
            if vp is None:
                img = np.zeros((H, W, 3), dtype=np.uint8)
            else:
                cam_y = float(eye[1])
                hl = cam_y - 60.0
                hh = cam_y + 30.0
                _, _, _, monster_pref, npc_pref, weapon_pref = schema_info
                m_lines = build_skel_lines(row, monster_pref, em_edges,
                    color_base=COLOR_MONSTER_BASE, color_max=COLOR_MONSTER_MAX)
                n_lines = build_skel_lines(row, npc_pref, npc_edges,
                    color_base=COLOR_NPC_BASE, color_max=COLOR_NPC_MAX)
                w_lines = build_skel_lines(row, weapon_pref, wp_edges,
                    color_base=COLOR_WEAPON_BASE, color_max=COLOR_WEAPON_MAX)
                img = renderer.render_frame(
                    vp,
                    skel_lines=[('monster', m_lines),
                                ('npc', n_lines),
                                ('weapon', w_lines)],
                    depth_near=depth_near, depth_far=depth_far,
                    height_band=(hl, hh),
                    terrain_near_cull=terrain_near_cull,
                )
            try:
                enc.stdin.write(img.tobytes())
                n_rendered += 1
            except BrokenPipeError:
                break

    try:
        enc.stdin.close()
    except Exception:
        pass
    rc = enc.wait()
    return n_rendered, (1 if rc != 0 else 0)


# ============================================================
# Main CLI (single-task demo)
# ============================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--rec', required=True)
    ap.add_argument('--start-row', type=int, required=True)
    ap.add_argument('--n-frames', type=int, default=150)
    ap.add_argument('--stage', type=int, required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--depth-near', type=float, default=5.0)
    ap.add_argument('--depth-far', type=float, default=80.0)
    ap.add_argument('--terrain-near-cull', type=float, default=TERRAIN_NEAR_CULL,
                    help='drop terrain closer than this many units to the camera '
                         '(0 disables; skeletons never culled). Default from '
                         'POSE_TERRAIN_NEAR_CULL env.')
    ap.add_argument('--fps', type=int, default=30)
    args = ap.parse_args()

    terrain = TerrainStage(args.stage)
    print(f'[stage {args.stage}] {len(terrain.chunks)} chunks, '
          f'h=[{terrain.height_min:.1f},{terrain.height_max:.1f}]')
    print(f'terrain_near_cull = {args.terrain_near_cull} '
          f'({"disabled" if args.terrain_near_cull <= 0 else "units from camera"})')
    renderer = GLRenderer()
    print(f'GL: {renderer.ctx.info.get("GL_RENDERER")}')
    renderer.bind_terrain(terrain)

    enc = start_ffmpeg_encoder(args.out, fps=args.fps)
    t0 = time.time()
    schema_info = None
    em_edges = npc_edges = wp_edges = None
    with RecBinReader(args.rec) as reader:
        for i in range(args.n_frames):
            reader.seek_to_row(args.start_row + i)
            row = reader.read_row()
            if schema_info is None:
                schema_info = detect_schema(row)
                em_edges = load_edges(Path(FILTER_DIR) / f'em_{schema_info[1]}.json')
                npc_edges = load_edges(Path(FILTER_DIR) / 'npc.json')
                wp_edges = load_edges(Path(FILTER_DIR) / f'wp_{schema_info[2]}.json')
                print(f'schema={schema_info[0]} em={schema_info[1]} wp={schema_info[2]}')

            vp, eye, _, _ = compute_view_proj(row)
            if vp is None:
                img = np.zeros((H, W, 3), dtype=np.uint8)
            else:
                cam_y = float(eye[1])
                hl, hh = cam_y - 60.0, cam_y + 30.0
                _, _, _, monster_pref, npc_pref, weapon_pref = schema_info
                m_lines = build_skel_lines(row, monster_pref, em_edges,
                    color_base=COLOR_MONSTER_BASE, color_max=COLOR_MONSTER_MAX)
                n_lines = build_skel_lines(row, npc_pref, npc_edges,
                    color_base=COLOR_NPC_BASE, color_max=COLOR_NPC_MAX)
                w_lines = build_skel_lines(row, weapon_pref, wp_edges,
                    color_base=COLOR_WEAPON_BASE, color_max=COLOR_WEAPON_MAX)
                # Compute platform foot positions (NPC / monster origin pos)
                npx = row.get('npc.list.1.pos.x')
                npy = row.get('npc.list.1.pos.y')
                npz = row.get('npc.list.1.pos.z')
                npc_foot = None
                monster_foot = None
                if npx is not None and npy is not None and npz is not None:
                    npc_foot = (float(npx), float(npy), float(npz))
                mpx = row.get('monster.list.1.pos.x')
                mpy = row.get('monster.list.1.pos.y')
                mpz = row.get('monster.list.1.pos.z')
                if mpx is not None and mpy is not None and mpz is not None:
                    monster_foot = (float(mpx), float(mpy), float(mpz))
                img = renderer.render_frame(
                    vp,
                    skel_lines=[('monster', m_lines),
                                ('npc', n_lines),
                                ('weapon', w_lines)],
                    depth_near=args.depth_near, depth_far=args.depth_far,
                    height_band=(hl, hh),
                    npc_foot=npc_foot, monster_foot=monster_foot,
                    terrain_near_cull=args.terrain_near_cull,
                )
            enc.stdin.write(img.tobytes())
            if i % 30 == 0:
                print(f'  frame {i}/{args.n_frames}  '
                      f'elapsed {time.time() - t0:.1f}s')
    try:
        enc.stdin.close()
    except Exception:
        pass
    rc = enc.wait()
    elapsed = time.time() - t0
    print(f'done -> {args.out} ({elapsed:.1f}s, '
          f'{args.n_frames / max(elapsed, 1e-3):.1f} fps, rc={rc})')
    renderer.release()


if __name__ == '__main__':
    main()
