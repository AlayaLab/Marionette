#!/usr/bin/env python3
"""Comprehensive evaluation and visualization for Continuous Frame GPT rollout.

Analyses:
  1. Root-aligned MPJPE (local frame, no world rot needed)
  2. Absolute MPJPE with root accumulation (root drift analysis)
  3. Root trajectory drift over time
  4. Per-horizon error curves
  5. Side-by-side skeleton visualization (pred vs GT)
"""

import json
import os
import subprocess
import sys
import tempfile
from collections import deque
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from models.gpt_continuous import ContinuousMotionGPT
from scripts.vis_skeleton_video import (
    parse_skeleton, render_frame, compute_frame_bounds,
    _swap_yz, _fix_zero_joints, get_bone_color,
)


# ============================================================
# Feature slicing
# ============================================================

def slice_to_pos276(data_780):
    """Slice 780D -> 276D (position + root rot6d)."""
    monster_pos = data_780[..., :162]
    npc_pos = data_780[..., 486:582]
    weapon_pos = data_780[..., 774:780]
    monster_root_rot = data_780[..., 162:168]
    npc_root_rot = data_780[..., 582:588]
    return np.concatenate([monster_pos, npc_pos, weapon_pos,
                           monster_root_rot, npc_root_rot], axis=-1)


def denormalize_276(features, mean_276, std_276):
    return features * std_276 + mean_276


def _rot6d_to_matrix(rot6d):
    """Convert 6D rotation to 3x3 matrix. rot6d: (..., 6)."""
    a1 = rot6d[..., :3]
    a2 = rot6d[..., 3:6]
    b1 = a1 / (np.linalg.norm(a1, axis=-1, keepdims=True) + 1e-8)
    dot = np.sum(b1 * a2, axis=-1, keepdims=True)
    b2 = a2 - dot * b1
    b2 = b2 / (np.linalg.norm(b2, axis=-1, keepdims=True) + 1e-8)
    b3 = np.cross(b1, b2, axis=-1)
    return np.stack([b1, b2, b3], axis=-1)


# ============================================================
# Reconstruct positions from 276D features
# ============================================================

def features276_to_positions(features_276_denorm):
    """Convert denormalized 276D features to world-frame joint positions.

    Uses rot6d to correctly rotate local deltas and rel_pos to world frame.

    Returns:
        monster_pos: (T, 54, 3) world-frame positions
        npc_pos: (T, 32, 3)
        weapon_pos: (T, 3, 3)
        m_root_traj: (T, 3) accumulated root trajectory
        n_root_traj: (T, 3)
    """
    T = features_276_denorm.shape[0]
    f = features_276_denorm

    m_root_delta = f[:, 0:3]
    m_rel_pos = f[:, 3:162].reshape(T, 53, 3)
    n_root_delta = f[:, 162:165]
    n_rel_pos = f[:, 165:258].reshape(T, 31, 3)
    w_rel_pos = f[:, 258:264].reshape(T, 2, 3)
    m_rot6d = f[:, 264:270]
    n_rot6d = f[:, 270:276]

    m_world_rot = _rot6d_to_matrix(m_rot6d)  # (T, 3, 3)
    n_world_rot = _rot6d_to_matrix(n_rot6d)

    # World delta = world_rot @ local_delta
    m_world_delta = np.einsum('tij,tj->ti', m_world_rot, m_root_delta)
    n_world_delta = np.einsum('tij,tj->ti', n_world_rot, n_root_delta)

    m_root_traj = np.cumsum(m_world_delta, axis=0)
    n_root_traj = np.cumsum(n_world_delta, axis=0)

    # World positions: root + rot @ rel_pos_local
    monster_pos = np.zeros((T, 54, 3))
    npc_pos = np.zeros((T, 32, 3))
    weapon_pos = np.zeros((T, 3, 3))

    for t in range(T):
        monster_pos[t, 0] = m_root_traj[t]
        monster_pos[t, 1:] = m_root_traj[t] + np.einsum('ij,kj->ki', m_world_rot[t], m_rel_pos[t])

        npc_pos[t, 0] = n_root_traj[t]
        npc_pos[t, 1:] = n_root_traj[t] + np.einsum('ij,kj->ki', n_world_rot[t], n_rel_pos[t])

        weapon_pos[t, 0] = n_root_traj[t]
        weapon_pos[t, 1:] = n_root_traj[t] + np.einsum('ij,kj->ki', n_world_rot[t], w_rel_pos[t])

    return monster_pos, npc_pos, weapon_pos, m_root_traj, n_root_traj


# ============================================================
# Metrics
# ============================================================

def compute_all_metrics(pred_276_denorm, gt_276_denorm, horizons=None):
    """Compute comprehensive metrics."""
    T = min(pred_276_denorm.shape[0], gt_276_denorm.shape[0])
    pred, gt = pred_276_denorm[:T], gt_276_denorm[:T]

    if horizons is None:
        horizons = [10, 25, 50, 100, T]
    horizons = sorted(set(min(h, T) for h in horizons))

    metrics = {}

    # Root-aligned MPJPE (= local frame rel_pos error, no root accumulation)
    for h in horizons:
        p, g = pred[:h], gt[:h]

        # Monster rel_pos [3:162]
        m_rp_pred = p[:, 3:162].reshape(-1, 53, 3)
        m_rp_gt = g[:, 3:162].reshape(-1, 53, 3)
        m_ra = np.sqrt(np.sum((m_rp_pred - m_rp_gt) ** 2, axis=-1))  # (T, 53)

        # NPC rel_pos [165:258]
        n_rp_pred = p[:, 165:258].reshape(-1, 31, 3)
        n_rp_gt = g[:, 165:258].reshape(-1, 31, 3)
        n_ra = np.sqrt(np.sum((n_rp_pred - n_rp_gt) ** 2, axis=-1))

        metrics[f'monster_ra_mpjpe@{h}f'] = m_ra.mean()
        metrics[f'npc_ra_mpjpe@{h}f'] = n_ra.mean()

        # Root delta error
        m_rd_err = np.sqrt(np.sum((p[:, :3] - g[:, :3]) ** 2, axis=-1)).mean()
        n_rd_err = np.sqrt(np.sum((p[:, 162:165] - g[:, 162:165]) ** 2, axis=-1)).mean()
        metrics[f'monster_root_delta@{h}f'] = m_rd_err
        metrics[f'npc_root_delta@{h}f'] = n_rd_err

    # Absolute MPJPE (with root accumulation, no world rot)
    pred_m, pred_n, pred_w, pred_m_traj, pred_n_traj = features276_to_positions(pred)
    gt_m, gt_n, gt_w, gt_m_traj, gt_n_traj = features276_to_positions(gt)

    for h in horizons:
        m_abs = np.sqrt(np.sum((pred_m[:h] - gt_m[:h]) ** 2, axis=-1)).mean()
        n_abs = np.sqrt(np.sum((pred_n[:h] - gt_n[:h]) ** 2, axis=-1)).mean()
        metrics[f'monster_abs_mpjpe@{h}f'] = m_abs
        metrics[f'npc_abs_mpjpe@{h}f'] = n_abs

    # Root trajectory drift (cumulative error)
    m_root_drift = np.sqrt(np.sum((pred_m_traj - gt_m_traj) ** 2, axis=-1))
    n_root_drift = np.sqrt(np.sum((pred_n_traj - gt_n_traj) ** 2, axis=-1))
    metrics['monster_root_drift_curve'] = m_root_drift
    metrics['npc_root_drift_curve'] = n_root_drift
    for h in horizons:
        metrics[f'monster_root_drift@{h}f'] = m_root_drift[min(h, T) - 1]
        metrics[f'npc_root_drift@{h}f'] = n_root_drift[min(h, T) - 1]

    # Per-frame RA-MPJPE curve
    m_rp_pred_all = pred[:, 3:162].reshape(T, 53, 3)
    m_rp_gt_all = gt[:, 3:162].reshape(T, 53, 3)
    m_ra_per_frame = np.sqrt(np.sum((m_rp_pred_all - m_rp_gt_all) ** 2, axis=-1)).mean(axis=1)
    metrics['monster_ra_curve'] = m_ra_per_frame

    n_rp_pred_all = pred[:, 165:258].reshape(T, 31, 3)
    n_rp_gt_all = gt[:, 165:258].reshape(T, 31, 3)
    n_ra_per_frame = np.sqrt(np.sum((n_rp_pred_all - n_rp_gt_all) ** 2, axis=-1)).mean(axis=1)
    metrics['npc_ra_curve'] = n_ra_per_frame

    return metrics


# ============================================================
# Plotting
# ============================================================

def plot_error_curves(all_metrics, output_dir, seed_frames):
    """Plot error over time curves aggregated across segments."""
    os.makedirs(output_dir, exist_ok=True)

    # Collect per-frame curves, pad with nan
    max_T = max(len(m['monster_ra_curve']) for m in all_metrics)

    m_ra_curves = []
    n_ra_curves = []
    m_drift_curves = []
    n_drift_curves = []
    for m in all_metrics:
        c = m['monster_ra_curve']
        padded = np.full(max_T, np.nan)
        padded[:len(c)] = c
        m_ra_curves.append(padded)

        c = m['npc_ra_curve']
        padded = np.full(max_T, np.nan)
        padded[:len(c)] = c
        n_ra_curves.append(padded)

        c = m['monster_root_drift_curve']
        padded = np.full(max_T, np.nan)
        padded[:len(c)] = c
        m_drift_curves.append(padded)

        c = m['npc_root_drift_curve']
        padded = np.full(max_T, np.nan)
        padded[:len(c)] = c
        n_drift_curves.append(padded)

    m_ra_curves = np.array(m_ra_curves)
    n_ra_curves = np.array(n_ra_curves)
    m_drift_curves = np.array(m_drift_curves)
    n_drift_curves = np.array(n_drift_curves)

    frames = np.arange(max_T)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # RA-MPJPE over time
    ax = axes[0, 0]
    mean_m = np.nanmean(m_ra_curves, axis=0)
    std_m = np.nanstd(m_ra_curves, axis=0)
    ax.plot(frames, mean_m, 'r-', label='Monster', linewidth=2)
    ax.fill_between(frames, mean_m - std_m, mean_m + std_m, alpha=0.2, color='r')
    mean_n = np.nanmean(n_ra_curves, axis=0)
    std_n = np.nanstd(n_ra_curves, axis=0)
    ax.plot(frames, mean_n, 'b-', label='NPC', linewidth=2)
    ax.fill_between(frames, mean_n - std_n, mean_n + std_n, alpha=0.2, color='b')
    ax.set_xlabel('Generated Frame')
    ax.set_ylabel('RA-MPJPE')
    ax.set_title('Root-Aligned MPJPE over Time')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Root drift over time
    ax = axes[0, 1]
    mean_md = np.nanmean(m_drift_curves, axis=0)
    std_md = np.nanstd(m_drift_curves, axis=0)
    ax.plot(frames, mean_md, 'r-', label='Monster', linewidth=2)
    ax.fill_between(frames, mean_md - std_md, mean_md + std_md, alpha=0.2, color='r')
    mean_nd = np.nanmean(n_drift_curves, axis=0)
    std_nd = np.nanstd(n_drift_curves, axis=0)
    ax.plot(frames, mean_nd, 'b-', label='NPC', linewidth=2)
    ax.fill_between(frames, mean_nd - std_nd, mean_nd + std_nd, alpha=0.2, color='b')
    ax.set_xlabel('Generated Frame')
    ax.set_ylabel('Root Drift (accumulated)')
    ax.set_title('Root Position Drift over Time')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Comparison bar chart: RA-MPJPE vs Abs-MPJPE at key horizons
    ax = axes[1, 0]
    horizons_to_plot = [25, 50, 100]
    x = np.arange(len(horizons_to_plot))
    width = 0.18

    for i, (prefix, color, label) in enumerate([
        ('monster_ra_mpjpe', '#FF5722', 'Monster RA'),
        ('monster_abs_mpjpe', '#FF9800', 'Monster Abs'),
        ('npc_ra_mpjpe', '#2196F3', 'NPC RA'),
        ('npc_abs_mpjpe', '#64B5F6', 'NPC Abs'),
    ]):
        vals = []
        for h in horizons_to_plot:
            key = f'{prefix}@{h}f'
            v = [m[key] for m in all_metrics if key in m]
            vals.append(np.mean(v) if v else 0)
        ax.bar(x + i * width, vals, width, label=label, color=color)

    ax.set_xticks(x + 1.5 * width)
    ax.set_xticklabels([f'{h}f' for h in horizons_to_plot])
    ax.set_ylabel('MPJPE')
    ax.set_title('RA-MPJPE vs Absolute MPJPE')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')

    # V5 baseline comparison
    ax = axes[1, 1]
    v5_monster_ra = 1.88
    v5_npc_ra = 0.56
    cont_monster_ra = np.mean([m['monster_ra_mpjpe@100f'] for m in all_metrics])
    cont_npc_ra = np.mean([m['npc_ra_mpjpe@100f'] for m in all_metrics])
    cont_monster_abs = np.mean([m['monster_abs_mpjpe@100f'] for m in all_metrics])
    cont_npc_abs = np.mean([m['npc_abs_mpjpe@100f'] for m in all_metrics])

    labels = ['Monster RA', 'Monster Abs', 'NPC RA', 'NPC Abs']
    v5_vals = [v5_monster_ra, None, v5_npc_ra, None]  # Abs not available for V5
    cont_vals = [cont_monster_ra, cont_monster_abs, cont_npc_ra, cont_npc_abs]

    x = np.arange(len(labels))
    bars1 = ax.bar(x - 0.2, [v if v is not None else 0 for v in v5_vals], 0.35,
                   label='VQ-GPT V5', color='#9E9E9E', alpha=0.8)
    bars2 = ax.bar(x + 0.2, cont_vals, 0.35,
                   label='Continuous GPT', color='#4CAF50', alpha=0.8)
    # Add value labels
    for bar, val in zip(bars2, cont_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.2f}', ha='center', va='bottom', fontsize=9)
    for bar, val in zip(bars1, v5_vals):
        if val is not None:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{val:.2f}', ha='center', va='bottom', fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel('MPJPE @100f')
    ax.set_title('vs VQ-GPT V5 Baseline (@100 frames)')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    fig.suptitle(f'Continuous GPT Evaluation (seed={seed_frames}f)', fontsize=14, fontweight='bold')
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'eval_curves.png'), dpi=150)
    plt.close(fig)
    print(f"Saved eval curves: {output_dir}/eval_curves.png")


# ============================================================
# Side-by-side skeleton visualization
# ============================================================

def render_sidebyside_video(pred_276_denorm, gt_276_denorm, entity_configs,
                             output_path, fps=20, max_frames=200, title=""):
    """Render pred vs GT side-by-side skeleton video."""
    T = min(pred_276_denorm.shape[0], gt_276_denorm.shape[0])
    if max_frames > 0:
        T = min(T, max_frames)

    pred_m, pred_n, pred_w, _, _ = features276_to_positions(pred_276_denorm[:T])
    gt_m, gt_n, gt_w, _, _ = features276_to_positions(gt_276_denorm[:T])

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)

    print(f"Rendering {T} frames side-by-side...")

    with tempfile.TemporaryDirectory() as tmpdir:
        for t in range(T):
            fig = plt.figure(figsize=(18, 8), dpi=100)

            for panel_idx, (m_pos, n_pos, w_pos, panel_title) in enumerate([
                (pred_m[t], pred_n[t], pred_w[t], 'Predicted'),
                (gt_m[t], gt_n[t], gt_w[t], 'Ground Truth'),
            ]):
                ax = fig.add_subplot(1, 2, panel_idx + 1, projection='3d')

                positions = {'monster': m_pos, 'npc': n_pos, 'weapon': w_pos}
                entity_colors = {
                    'monster': ('#FF5722', '#E64A19'),
                    'npc': ('#2196F3', '#1976D2'),
                    'weapon': ('#9C27B0', '#7B1FA2'),
                }

                for ent_name, joints_raw in positions.items():
                    if ent_name not in entity_configs:
                        continue
                    names, parents = entity_configs[ent_name]
                    joints = _swap_yz(_fix_zero_joints(joints_raw, parents))
                    colors = entity_colors.get(ent_name, ('#666', '#444'))

                    for j in range(len(names)):
                        if parents[j] >= 0:
                            p = parents[j]
                            color = get_bone_color(names[j], names[p])
                            if ent_name == 'npc':
                                color = '#42A5F5' if 'L_' in names[j] else '#EF5350' if 'R_' in names[j] else '#66BB6A'
                            ax.plot([joints[j, 0], joints[p, 0]],
                                    [joints[j, 1], joints[p, 1]],
                                    [joints[j, 2], joints[p, 2]],
                                    color=color, linewidth=2.0, alpha=0.9)

                    ax.scatter(joints[:, 0], joints[:, 1], joints[:, 2],
                               c=colors[0], s=12, depthshade=True, alpha=0.8)

                # Set bounds
                bounds = compute_frame_bounds(positions, entity_configs, pad=3.0)
                ax.set_xlim(bounds['xmin'], bounds['xmax'])
                ax.set_ylim(bounds['ymin'], bounds['ymax'])
                ax.set_zlim(bounds['zmin'], bounds['zmax'])
                ax.set_xlabel('X', fontsize=7)
                ax.set_ylabel('Z', fontsize=7)
                ax.set_zlabel('Y (up)', fontsize=7)
                ax.view_init(elev=20, azim=-60 + t * 0.3)
                ax.set_title(panel_title, fontsize=12)

            fig.suptitle(f'{title}  frame {t}/{T}', fontsize=13)
            fig.tight_layout()
            fig.canvas.draw()
            w, h = fig.canvas.get_width_height()
            img = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)
            plt.imsave(os.path.join(tmpdir, f'frame_{t:05d}.png'), img)
            plt.close(fig)

            if (t + 1) % 50 == 0 or t == T - 1:
                print(f'  Rendered {t+1}/{T}')

        cmd = [
            'ffmpeg', '-y',
            '-framerate', str(fps),
            '-i', os.path.join(tmpdir, 'frame_%05d.png'),
            '-c:v', 'libx264',
            '-pix_fmt', 'yuv420p',
            '-crf', '20',
            output_path,
        ]
        print(f'Encoding: {output_path}')
        result = subprocess.run(cmd, capture_output=True)
        if result.returncode != 0:
            print(f'ffmpeg error: {result.stderr.decode()}')
        else:
            print(f'Saved video: {output_path} ({T} frames @ {fps}fps = {T/fps:.1f}s)')


# ============================================================
# Main
# ============================================================

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--seed_frames", type=int, default=64)
    parser.add_argument("--gen_frames", type=int, default=100)
    parser.add_argument("--num_evals", type=int, default=10)
    parser.add_argument("--output_dir", type=str, default="output/eval_continuous/")
    parser.add_argument("--vis_segments", type=int, default=3,
                        help="Number of segments to render videos for")
    parser.add_argument("--max_vis_frames", type=int, default=200)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    mc = ckpt["config"]["model"]
    model = ContinuousMotionGPT(
        feat_dim=mc["feat_dim"],
        embed_dim=mc["embed_dim"],
        block_size=mc["block_size"],
        num_layers=mc["num_layers"],
        n_head=mc["n_head"],
        drop_out_rate=mc["drop_out_rate"],
        fc_rate=mc["fc_rate"],
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    mean_276 = ckpt.get("mean", None)
    std_276 = ckpt.get("std", None)

    # Load data
    data = np.load(args.data_path, allow_pickle=True)
    num_seg = int(data["num_segments"])

    if mean_276 is None:
        mean_276 = slice_to_pos276(data["mean"])
        std_276 = slice_to_pos276(data["std"])

    # Load skeleton configs
    skel_dir = PROJECT_ROOT / "data" / "skeleton"
    entity_configs = {}
    for ent, skel_file in [('monster', 'em_19_pruned.json'),
                            ('npc', 'npc_pruned.json'),
                            ('weapon', 'wp_4.json')]:
        names, parents = parse_skeleton(str(skel_dir / skel_file))
        entity_configs[ent] = (names, parents)

    os.makedirs(args.output_dir, exist_ok=True)

    # Evaluate multiple segments
    eval_indices = list(range(min(args.num_evals, num_seg)))
    all_metrics = []
    vis_data = []

    for seg_idx in eval_indices:
        seg_780 = data[f"segment_{seg_idx}"]
        seg_276 = slice_to_pos276(seg_780).astype(np.float32)

        total_needed = args.seed_frames + args.gen_frames
        if seg_276.shape[0] < total_needed:
            continue

        seed = seg_276[:args.seed_frames]
        seed_tensor = torch.from_numpy(seed).float().unsqueeze(0).to(device)

        with torch.no_grad():
            full_seq = model.generate(seed_tensor, args.gen_frames)

        pred_all = full_seq.squeeze(0).cpu().numpy()
        pred_gen = pred_all[args.seed_frames:]
        gt_gen = seg_276[args.seed_frames:args.seed_frames + args.gen_frames]
        T_compare = min(pred_gen.shape[0], gt_gen.shape[0])

        if T_compare == 0:
            continue

        # Denormalize
        pred_denorm = denormalize_276(pred_gen[:T_compare], mean_276, std_276)
        gt_denorm = denormalize_276(gt_gen[:T_compare], mean_276, std_276)

        metrics = compute_all_metrics(pred_denorm, gt_denorm, horizons=[10, 25, 50, 100])
        metrics['segment'] = seg_idx
        all_metrics.append(metrics)

        if len(vis_data) < args.vis_segments:
            vis_data.append((seg_idx, pred_denorm, gt_denorm))

    # Print summary
    print(f"\n{'='*70}")
    print(f"Continuous GPT Evaluation: {len(all_metrics)} segments")
    print(f"Seed: {args.seed_frames}f, Generate: {args.gen_frames}f")
    print(f"{'='*70}")

    scalar_keys = [k for k in sorted(all_metrics[0].keys())
                   if k != 'segment' and not k.endswith('_curve')]
    for key in scalar_keys:
        vals = [m[key] for m in all_metrics]
        print(f"  {key:35s}: {np.mean(vals):.4f} ± {np.std(vals):.4f}")

    # Plot curves
    plot_error_curves(all_metrics, args.output_dir, args.seed_frames)

    # Render videos
    for seg_idx, pred_denorm, gt_denorm in vis_data:
        vid_path = os.path.join(args.output_dir, f'vis_seg{seg_idx}.mp4')
        render_sidebyside_video(
            pred_denorm, gt_denorm, entity_configs,
            vid_path, fps=20, max_frames=args.max_vis_frames,
            title=f'Segment {seg_idx}')

    print(f"\nAll outputs saved to {args.output_dir}")


if __name__ == "__main__":
    main()
