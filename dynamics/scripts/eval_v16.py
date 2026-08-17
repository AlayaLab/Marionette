#!/usr/bin/env python3
"""V16 evaluation: combined ActionGPT + PoseGPT autoregressive rollout.

Primary metric: Action Compliance Rate (ACR) at horizons 50/100/200/300.
Secondary: drift, RA-MPJPE.

Usage:
  python scripts/eval_v16.py \
    --action_ckpt output/dynamics/action_gpt.pt \
    --pose_ckpt output/dynamics/pose_gpt.pt \
    --segments 85 572 1202 \
    --seed_frames 64 \
    --horizon 300 \
    --output_dir output/dynamics/
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from models.action_gpt import ActionGPT
from models.pose_gpt import PoseGPT
from src.train_v16 import slice_root_18d, slice_body_258d, slice_weapon_6d, slice_root_delta_6d
from src.train_gpt_continuous import slice_to_pos276
from scripts.eval_continuous import (
    features276_to_positions,
    compute_all_metrics,
    denormalize_276,
)


# ============================================================
# Model loading
# ============================================================

def load_action_gpt(ckpt_path, m_vocab, n_vocab, device, combined_dir=None):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt['config']['model']
    # Planner terrain (B v2): terrain dims from metadata.terr.npz if this ckpt was trained with it
    a_tp = a_tc = 0
    if cfg.get('terrain_planner', False) and combined_dir:
        import numpy as _np
        tm = _np.load(f"{combined_dir}/metadata.terr.npz", allow_pickle=True)
        a_tp = int(tm['patch_dim']); a_tc = int(tm['n_key'])
    model = ActionGPT(
        m_vocab_size=m_vocab,
        n_vocab_size=n_vocab,
        action_emb_dim=cfg.get('action_emb_dim', 64),
        progress_dim=2,
        weapon_dim=6,
        distance_dim=cfg.get('distance_dim', 0),
        root_out_dim=cfg.get('root_out_dim', 18),
        embed_dim=cfg['embed_dim'],
        block_size=cfg['block_size'],
        num_layers=cfg['num_layers'],
        n_head=cfg['n_head'],
        drop_out_rate=cfg.get('drop_out_rate', 0.0),
        fc_rate=cfg.get('fc_rate', 4),
        use_goal=cfg.get('use_goal', False),
        terrain_patch_dim=a_tp, terrain_clear_dim=a_tc,
        terrain_emb_dim=cfg.get('terrain_emb_dim', 32),
        hp_dim=cfg.get('hp_dim', 0), hp_out_dim=cfg.get('hp_out_dim', 0),
    ).to(device)
    if a_tp > 0:
        model.terrain_patch_dim_meta = a_tp; model.terrain_n_key = a_tc
    # Prefer EMA weights if available
    if 'ema' in ckpt and ckpt['ema'] is not None:
        ema_shadow = ckpt['ema']['shadow']
        sd = model.state_dict()
        for k in sd:
            if k in ema_shadow:
                sd[k] = ema_shadow[k].to(device)
        model.load_state_dict(sd)
        print(f"[load_action_gpt] loaded EMA weights from {ckpt_path}")
    else:
        model.load_state_dict(ckpt['model'])
        print(f"[load_action_gpt] loaded raw weights from {ckpt_path}")
    model.eval()
    return model


def load_pose_gpt(ckpt_path, m_vocab, n_vocab, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt['config']['model']
    model = PoseGPT(
        m_vocab_size=m_vocab,
        n_vocab_size=n_vocab,
        action_emb_dim=cfg.get('action_emb_dim', 64),
        embed_dim=cfg['embed_dim'],
        block_size=cfg['block_size'],
        num_layers=cfg['num_layers'],
        n_head=cfg['n_head'],
        drop_out_rate=cfg.get('drop_out_rate', 0.0),
        fc_rate=cfg.get('fc_rate', 4),
        root_input_dim=cfg.get('root_input_dim', 0),
        root_output_dim=cfg.get('root_output_dim', 0),
    ).to(device)
    if 'ema' in ckpt and ckpt['ema'] is not None:
        ema_shadow = ckpt['ema']['shadow']
        sd = model.state_dict()
        for k in sd:
            if k in ema_shadow:
                sd[k] = ema_shadow[k].to(device)
        model.load_state_dict(sd)
        print(f"[load_pose_gpt] loaded EMA weights (root_in={cfg.get('root_input_dim',0)}, root_out={cfg.get('root_output_dim',0)}) from {ckpt_path}")
    else:
        model.load_state_dict(ckpt['model'])
        print(f"[load_pose_gpt] loaded raw weights from {ckpt_path}")
    model.eval()
    return model


# ============================================================
# Combined autoregressive rollout
# ============================================================

@torch.no_grad()
def combined_rollout(action_gpt, pose_gpt, seed_276, seed_action_m, seed_action_n,
                     seed_progress, num_frames, gt_progress=None,
                     progress_mode='gt',
                     temperature=0.0, repeat_penalty=0.0,
                     median_run_m=None, median_run_n=None,
                     goal_m_seq=None, goal_n_seq=None, seed_hp=None,
                     force_am_seq=None, force_an_seq=None):
    """Goal mode (cloze ActionGPT, action_gpt.use_goal): feed sparse future-action
    GOAL embeddings instead of the leaked dense progress. goal_m_seq/goal_n_seq are
    (1, S+num_frames) action ids = the upcoming-transition action per frame (positioned
    anchors, option A). Progress is predicted internally and ignored here."""
    """Run combined V16 inference.

    Args:
        seed_276: (1, S, 276) seed motion features (normalized)
        seed_action_m: (1, S) seed monster action IDs
        seed_action_n: (1, S) seed NPC action IDs
        seed_progress: (1, S, 2) seed progress
        num_frames: int, number of frames to generate
        gt_progress: (1, num_frames, 2) optional GT progress for the future frames
        progress_mode: 'gt' -> use gt_progress for the rollout
                       'zero' -> zero out future progress
        temperature: float, 0=argmax, >0=sample with temperature
        repeat_penalty: float, logit penalty for action repeated beyond median run length
        median_run_m: dict {action_id: median_run_length} for Monster (for repeat penalty)
        median_run_n: dict {action_id: median_run_length} for NPC

    Returns:
        full_seq: (1, S+num_frames, 276)
        action_m_seq: (1, S+num_frames)
        action_n_seq: (1, S+num_frames)
    """
    device = seed_276.device
    block = action_gpt.block_size
    assert pose_gpt.block_size == block, "ActionGPT and PoseGPT block sizes must match"
    use_goal = getattr(action_gpt, 'use_goal', False)
    if use_goal:
        assert goal_m_seq is not None and goal_n_seq is not None, "goal mode needs goal_m_seq/goal_n_seq"

    # Initialize sequences
    full_seq = seed_276.clone()  # (1, S, 276)
    am_seq = seed_action_m.clone()
    an_seq = seed_action_n.clone()
    prog_seq = seed_progress.clone()
    # HP autoregression (if ActionGPT uses HP): seed from GT, then feed predicted HP forward.
    ag_hp = getattr(action_gpt, 'hp_dim', 0) > 0
    hp_seq = ((seed_hp.clone() if seed_hp is not None
               else torch.ones(1, full_seq.shape[1], 2, device=device)) if ag_hp else None)

    # Track consecutive action counts for repeat penalty
    last_am = seed_action_m[0, -1].item()
    last_an = seed_action_n[0, -1].item()
    consec_m = 1
    consec_n = 1

    def _sample_action(logits, temp, penalty, last_act, consec, median_runs):
        """Sample or argmax an action, with optional repeat penalty.
        logits: (1, 1, vocab). Returns: (1, 1) int64."""
        logits = logits.clone()
        if penalty > 0 and median_runs is not None:
            med = median_runs.get(last_act, 40)
            if consec > med:
                logits[0, 0, last_act] -= penalty
        if temp > 0:
            probs = torch.softmax(logits[0, 0] / temp, dim=-1)  # (vocab,)
            idx = torch.multinomial(probs, 1)  # (1,)
            return idx.view(1, 1)
        return logits.argmax(dim=-1)  # (1, 1)

    # Pre-extract per-frame views we need
    for t in range(num_frames):
        # Build context (last `block` frames)
        ctx_276 = full_seq[:, -block:]
        ctx_am = am_seq[:, -block:]
        ctx_an = an_seq[:, -block:]
        ctx_prog = prog_seq[:, -block:]

        ctx_root = slice_root_18d(ctx_276)
        ctx_body = slice_body_258d(ctx_276)
        ctx_weapon = slice_weapon_6d(ctx_276)

        # ----- ActionGPT step: predict next root + next action -----
        ctx_hp = hp_seq[:, -block:] if ag_hp else None
        if use_goal:
            cl = full_seq.shape[1]
            ctx_mgoal = goal_m_seq[:, max(0, cl - block):cl]
            ctx_ngoal = goal_n_seq[:, max(0, cl - block):cl]
            pred_root, m_logits, n_logits, _, hp_pred, _ = action_gpt(
                ctx_root, ctx_am, ctx_an, None, ctx_weapon, m_goal=ctx_mgoal, n_goal=ctx_ngoal, hp=ctx_hp)
        else:
            pred_root, m_logits, n_logits, _, hp_pred, _ = action_gpt(
                ctx_root, ctx_am, ctx_an, ctx_prog, ctx_weapon, hp=ctx_hp)
        next_root = pred_root[:, -1:, :] if pred_root is not None else None
        next_am = _sample_action(m_logits[:, -1:], temperature, repeat_penalty,
                                  last_am, consec_m, median_run_m)
        next_an = _sample_action(n_logits[:, -1:], temperature, repeat_penalty,
                                  last_an, consec_n, median_run_n)

        # ----- CONTROL injection: override sampled action with forced action-id -----
        # (this is the demo's core control mechanism — force NPC/monster action tokens)
        if force_am_seq is not None:
            next_am = force_am_seq[:, t:t+1].to(next_am.device)
        if force_an_seq is not None:
            next_an = force_an_seq[:, t:t+1].to(next_an.device)

        # Update consecutive counters
        am_val = next_am[0, 0].item()
        an_val = next_an[0, 0].item()
        consec_m = consec_m + 1 if am_val == last_am else 1
        consec_n = consec_n + 1 if an_val == last_an else 1
        last_am, last_an = am_val, an_val

        # ----- PoseGPT step: predict next body (+ optional root), conditioned on action -----
        pose_actm_in = torch.cat([ctx_am[:, 1:], next_am], dim=1)
        pose_actn_in = torch.cat([ctx_an[:, 1:], next_an], dim=1)

        # Determine PoseGPT root mode
        p_rid = pose_gpt.root_input_dim
        p_rod = pose_gpt.root_output_dim
        if p_rid == 6:  # V16.2: root_delta input
            pose_root_in = slice_root_delta_6d(ctx_276)
            pose_pred, _ = pose_gpt(ctx_body, pose_actm_in, pose_actn_in, root=pose_root_in)
        elif p_rid == 18:  # V16.3: full root input
            pose_root_in = slice_root_18d(ctx_276)
            pose_pred, _ = pose_gpt(ctx_body, pose_actm_in, pose_actn_in, root=pose_root_in)
        else:  # V16.0/V16.1: no root input
            pose_pred, _ = pose_gpt(ctx_body, pose_actm_in, pose_actn_in)

        next_body = pose_pred[:, -1:, :258]  # body is always first 258D

        # ----- Assemble next 276D frame -----
        next_276 = torch.zeros(1, 1, 276, device=device, dtype=full_seq.dtype)
        next_276[..., 3:162]   = next_body[..., 0:159]   # M_rel_pos
        next_276[..., 165:258] = next_body[..., 159:252] # N_rel_pos
        next_276[..., 258:264] = next_body[..., 252:258] # weapon

        a_rod = action_gpt.root_out_dim
        if p_rod == 18:
            # V16.3: PoseGPT predicts full root
            pose_root_out = pose_pred[:, -1:, 258:]  # (1,1,18)
            next_276[..., 0:3]     = pose_root_out[..., 0:3]
            next_276[..., 162:165] = pose_root_out[..., 3:6]
            next_276[..., 264:270] = pose_root_out[..., 6:12]
            next_276[..., 270:276] = pose_root_out[..., 12:18]
        elif p_rod == 6:
            # V16.2: PoseGPT predicts root_delta, ActionGPT predicts rot6d
            pose_root_out = pose_pred[:, -1:, 258:]  # (1,1,6) = root_delta
            next_276[..., 0:3]     = pose_root_out[..., 0:3]   # M_root_delta
            next_276[..., 162:165] = pose_root_out[..., 3:6]   # N_root_delta
            next_276[..., 264:270] = next_root[..., 0:6]       # M_rot6d from ActionGPT
            next_276[..., 270:276] = next_root[..., 6:12]      # N_rot6d from ActionGPT
        else:
            # V16.0/V16.1: ActionGPT predicts all root
            next_276[..., 0:3]     = next_root[..., 0:3]
            next_276[..., 162:165] = next_root[..., 3:6]
            next_276[..., 264:270] = next_root[..., 6:12]
            next_276[..., 270:276] = next_root[..., 12:18]

        # Append
        full_seq = torch.cat([full_seq, next_276], dim=1)
        am_seq = torch.cat([am_seq, next_am], dim=1)
        an_seq = torch.cat([an_seq, next_an], dim=1)
        if ag_hp:
            next_hp = hp_pred[:, -1:].clamp(0.0, 1.0) if hp_pred is not None else hp_seq[:, -1:]
            hp_seq = torch.cat([hp_seq, next_hp], dim=1)  # autoregress predicted HP

        # Progress: use GT for the rollout to isolate ActionGPT/PoseGPT errors
        if progress_mode == 'gt' and gt_progress is not None and t < gt_progress.shape[1]:
            next_prog = gt_progress[:, t:t+1]
        else:
            next_prog = torch.zeros(1, 1, 2, device=device, dtype=prog_seq.dtype)
        prog_seq = torch.cat([prog_seq, next_prog], dim=1)

    return full_seq, am_seq, an_seq


# ============================================================
# Action Compliance Rate
# ============================================================

def compute_acr(pred_actions, gt_actions, horizons):
    """Compute Action Compliance Rate at multiple horizons.

    Args:
        pred_actions: (N,) predicted action IDs (from rollout, after seed)
        gt_actions: (N,) GT action IDs (after seed)
        horizons: list of int horizons (e.g., [50, 100, 200, 300])
    Returns:
        dict mapping horizon -> ACR
    """
    pred = np.asarray(pred_actions)
    gt = np.asarray(gt_actions)
    N = min(len(pred), len(gt))
    out = {}
    for h in horizons:
        h_eff = min(h, N)
        if h_eff == 0:
            out[h] = float('nan')
        else:
            out[h] = float((pred[:h_eff] == gt[:h_eff]).mean())
    return out


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--action_ckpt', required=True)
    parser.add_argument('--pose_ckpt', required=True)
    parser.add_argument('--motion_data', default='./data/processed/motion_data.npz')
    parser.add_argument('--segments', type=int, nargs='+', default=[85, 572, 1202])
    parser.add_argument('--seed_frames', type=int, default=64)
    parser.add_argument('--horizon', type=int, default=300)
    parser.add_argument('--output_dir', default='./output/dynamics/')
    parser.add_argument('--progress_mode', choices=['gt', 'zero'], default='gt')
    parser.add_argument('--temperature', type=float, default=0.0,
                        help='Action sampling temperature (0=argmax)')
    parser.add_argument('--repeat_penalty', type=float, default=0.0,
                        help='Logit penalty for actions exceeding median run length')
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Loading data...")
    data = np.load(args.motion_data, allow_pickle=True)
    mean_780 = data['mean']; std_780 = data['std']
    mean_276 = slice_to_pos276(mean_780); std_276 = slice_to_pos276(std_780)
    m_vocab = int(data['m_action_vocab_size'])
    n_vocab = int(data['n_action_vocab_size'])

    print(f"Loading models (m_vocab={m_vocab}, n_vocab={n_vocab})...")
    action_gpt = load_action_gpt(args.action_ckpt, m_vocab, n_vocab, device)
    pose_gpt = load_pose_gpt(args.pose_ckpt, m_vocab, n_vocab, device)

    # Pre-compute median run lengths per action (for repeat penalty)
    median_run_m, median_run_n = None, None
    if args.repeat_penalty > 0:
        from collections import defaultdict
        runs_m, runs_n = defaultdict(list), defaultdict(list)
        num_seg = int(data['num_segments'])
        for i in range(min(100, num_seg)):
            for act_arr, runs_dict in [(data[f'action_m_{i}'], runs_m),
                                        (data[f'action_n_{i}'], runs_n)]:
                changes = np.where(act_arr[1:] != act_arr[:-1])[0]
                starts = np.concatenate([[0], changes + 1])
                ends = np.concatenate([changes + 1, [len(act_arr)]])
                for s, e in zip(starts, ends):
                    runs_dict[int(act_arr[s])].append(e - s)
        median_run_m = {k: int(np.median(v)) for k, v in runs_m.items()}
        median_run_n = {k: int(np.median(v)) for k, v in runs_n.items()}
        print(f"Repeat penalty enabled: {len(median_run_m)} M actions, {len(median_run_n)} N actions")

    if args.temperature > 0:
        print(f"Temperature sampling: T={args.temperature}")

    horizons = [50, 100, 200, args.horizon] if args.horizon not in (50, 100, 200) else [50, 100, 200]
    horizons = sorted(set(horizons))
    all_results = {}

    for seg_idx in args.segments:
        print(f"\n=== Segment {seg_idx} ===")
        seg_780 = data[f'segment_{seg_idx}']
        seg_276 = slice_to_pos276(seg_780).astype(np.float32)
        am = data[f'action_m_{seg_idx}'].astype(np.int64)
        an = data[f'action_n_{seg_idx}'].astype(np.int64)
        prog = data[f'action_progress_{seg_idx}'].astype(np.float32)

        L = len(seg_276)
        if L < args.seed_frames + args.horizon:
            print(f"  segment too short (L={L}), skipping")
            continue

        seed_end = args.seed_frames
        rollout_end = seed_end + args.horizon

        seed_276 = torch.from_numpy(seg_276[:seed_end]).unsqueeze(0).to(device)
        seed_am = torch.from_numpy(am[:seed_end]).unsqueeze(0).to(device)
        seed_an = torch.from_numpy(an[:seed_end]).unsqueeze(0).to(device)
        seed_prog = torch.from_numpy(prog[:seed_end]).unsqueeze(0).to(device)
        gt_prog = torch.from_numpy(prog[seed_end:rollout_end]).unsqueeze(0).to(device)

        full_seq, am_seq, an_seq = combined_rollout(
            action_gpt, pose_gpt, seed_276, seed_am, seed_an, seed_prog,
            num_frames=args.horizon, gt_progress=gt_prog,
            progress_mode=args.progress_mode,
            temperature=args.temperature, repeat_penalty=args.repeat_penalty,
            median_run_m=median_run_m, median_run_n=median_run_n,
        )

        # ACR
        pred_am_after = am_seq[0, seed_end:].cpu().numpy()
        pred_an_after = an_seq[0, seed_end:].cpu().numpy()
        gt_am_after = am[seed_end:rollout_end]
        gt_an_after = an[seed_end:rollout_end]

        acr_m = compute_acr(pred_am_after, gt_am_after, horizons)
        acr_n = compute_acr(pred_an_after, gt_an_after, horizons)

        # Pose metrics (denormalize for proper scale)
        pred_276_after = full_seq[0, seed_end:].cpu().numpy()
        pred_denorm = denormalize_276(pred_276_after, mean_276, std_276)
        gt_denorm = denormalize_276(seg_276[seed_end:rollout_end], mean_276, std_276)

        pose_metrics = compute_all_metrics(pred_denorm, gt_denorm, horizons=horizons)

        seg_result = {
            'seed_frames': args.seed_frames,
            'horizon': args.horizon,
            'acr_m': acr_m,
            'acr_n': acr_n,
            'pose_metrics': {k: float(v) if not hasattr(v, '__len__') else v.tolist()
                             for k, v in pose_metrics.items() if 'curve' not in k},
        }
        all_results[seg_idx] = seg_result

        print(f"  ACR Monster: " + ", ".join(f"@{h}f={v:.3f}" for h, v in acr_m.items()))
        print(f"  ACR NPC:     " + ", ".join(f"@{h}f={v:.3f}" for h, v in acr_n.items()))
        print(f"  M drift:     " + ", ".join(f"@{h}f={pose_metrics[f'monster_root_drift@{h}f']:.2f}m" for h in horizons))
        print(f"  N drift:     " + ", ".join(f"@{h}f={pose_metrics[f'npc_root_drift@{h}f']:.2f}m" for h in horizons))
        print(f"  M RA-MPJPE:  " + ", ".join(f"@{h}f={pose_metrics[f'monster_ra_mpjpe@{h}f']:.3f}" for h in horizons))
        print(f"  N RA-MPJPE:  " + ", ".join(f"@{h}f={pose_metrics[f'npc_ra_mpjpe@{h}f']:.3f}" for h in horizons))

        # Save predicted sequence for visualization
        np.savez(out_dir / f'seg{seg_idx}_pred.npz',
                 pred_276=pred_276_after, pred_action_m=pred_am_after, pred_action_n=pred_an_after,
                 gt_action_m=gt_am_after, gt_action_n=gt_an_after)

    # Aggregate
    if all_results:
        print(f"\n=== AGGREGATE ({len(all_results)} segments) ===")
        agg = {'acr_m': {}, 'acr_n': {}, 'pose': {}}
        for h in horizons:
            agg['acr_m'][h] = float(np.mean([r['acr_m'][h] for r in all_results.values()]))
            agg['acr_n'][h] = float(np.mean([r['acr_n'][h] for r in all_results.values()]))
            for key in [f'monster_root_drift@{h}f', f'npc_root_drift@{h}f',
                        f'monster_ra_mpjpe@{h}f', f'npc_ra_mpjpe@{h}f']:
                agg['pose'][key] = float(np.mean([r['pose_metrics'][key] for r in all_results.values()]))
        print(f"  ACR Monster: " + ", ".join(f"@{h}f={v:.3f}" for h, v in agg['acr_m'].items()))
        print(f"  ACR NPC:     " + ", ".join(f"@{h}f={v:.3f}" for h, v in agg['acr_n'].items()))
        all_results['_aggregate'] = agg

    with open(out_dir / 'results.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {out_dir}/results.json")


if __name__ == '__main__':
    main()
