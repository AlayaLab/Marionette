#!/usr/bin/env python3
"""Render 3D skeleton animation as MP4 video.

Supports multi-entity visualization (monster + npc + weapon).
Uses matplotlib for rendering, ffmpeg for encoding.

Usage:
    # From processed data (npz with segments)
    python scripts/vis_skeleton_video.py --data data/processed/motion_data.npz \
        --segment 0 --start 0 --end 200 --out output/vis/sample.mp4

    # From rollout result
    python scripts/vis_skeleton_video.py --rollout output/rollout_result.npz \
        --out output/vis/rollout.mp4
"""

import argparse
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

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# Skeleton topology
# ============================================================

def parse_skeleton(json_path):
    with open(json_path) as f:
        tree = json.load(f)
    names, parents = [], []
    queue = deque([(tree, -1)])
    while queue:
        node, pidx = queue.popleft()
        idx = len(names)
        names.append(node['name'])
        parents.append(pidx)
        if node.get('childs'):
            for c in node['childs']:
                queue.append((c, idx))
    return names, parents


def get_bone_color(name_a, name_b):
    """Color bones: left=blue, right=red, center=green."""
    for n in [name_a, name_b]:
        if n.startswith('L_'):
            return '#2196F3'
        if n.startswith('R_'):
            return '#F44336'
    return '#4CAF50'


# ============================================================
# Feature -> positions reconstruction
# ============================================================

def features_to_positions(features, mean, std, joint_counts, initial_positions=None):
    """Reconstruct joint positions from feature vectors.

    Args:
        features: (T, D) normalized features
        mean, std: normalization stats
        joint_counts: dict with 'monster', 'npc', 'weapon' joint counts
        initial_positions: dict of initial (J, 3) absolute positions per entity, or None.
                           These should be the absolute world positions at frame 0.

    Returns:
        dict of entity -> (T+1, J, 3) positions in world coordinates
    """
    # Denormalize
    feat = features * std + mean
    T = feat.shape[0]

    m_J = joint_counts['monster']
    n_J = joint_counts['npc']
    w_J = joint_counts['weapon']

    # Split features
    idx = 0
    m_delta = feat[:, idx:idx + m_J*3].reshape(T, m_J, 3); idx += m_J*3
    m_rot6d = feat[:, idx:idx + m_J*6].reshape(T, m_J, 6); idx += m_J*6
    n_delta = feat[:, idx:idx + n_J*3].reshape(T, n_J, 3); idx += n_J*3
    n_rot6d = feat[:, idx:idx + n_J*6].reshape(T, n_J, 6); idx += n_J*6
    w_delta = feat[:, idx:idx + w_J*3].reshape(T, w_J, 3); idx += w_J*3

    # Accumulate positions
    def accumulate(delta, init_pos=None):
        T, J, _ = delta.shape
        pos = np.zeros((T + 1, J, 3))
        if init_pos is not None:
            pos[0] = init_pos
        for t in range(T):
            pos[t + 1] = pos[t] + delta[t]
        return pos

    m_init = initial_positions.get('monster') if initial_positions else None
    n_init = initial_positions.get('npc') if initial_positions else None
    w_init = initial_positions.get('weapon') if initial_positions else None

    return {
        'monster': accumulate(m_delta, m_init),
        'npc': accumulate(n_delta, n_init),
        'weapon': accumulate(w_delta, w_init),
    }


# ============================================================
# Rendering
# ============================================================

def _fix_zero_joints(positions, parents):
    """Replace joints stuck at origin (0,0,0) with their parent's position.

    Some joints (e.g. NPC L_Instep, R_Instep, L_Palm, R_Palm) have zero
    position data in the BIN source. Their deltas are also zero, so they
    stay at (0,0,0) while other joints are at world coordinates like (-800, 1300).
    This creates absurdly long bones. Fix by copying parent position.
    """
    pos = positions.copy()
    J = pos.shape[0]
    for j in range(J):
        if np.allclose(pos[j], 0.0, atol=0.01) and parents[j] >= 0:
            if not np.allclose(pos[parents[j]], 0.0, atol=0.01):
                pos[j] = pos[parents[j]]
    return pos


def _swap_yz(joints):
    """Swap Y and Z axes: game Y (vertical) -> matplotlib Z (vertical)."""
    out = joints.copy()
    out[..., 1], out[..., 2] = joints[..., 2].copy(), joints[..., 1].copy()
    return out


def render_frame(entities, frame_idx, total_frames, bounds, entity_configs, title=""):
    """Render a single frame with multiple entities.

    Args:
        entities: dict of entity_name -> (J, 3) positions for this frame
                  Coordinates are in game space (Y=up). Rendering swaps Y<->Z
                  so matplotlib Z axis = vertical.
        frame_idx: frame number
        total_frames: total frames
        bounds: dict with axis limits (already in render space, Y<->Z swapped)
        entity_configs: dict of entity_name -> (names, parents, color_base)
        title: plot title
    """
    fig = plt.figure(figsize=(10, 8), dpi=100)
    ax = fig.add_subplot(111, projection='3d')

    entity_colors = {
        'monster': ('#FF5722', '#E64A19', '#BF360C'),  # orange-red
        'npc': ('#2196F3', '#1976D2', '#0D47A1'),       # blue
        'weapon': ('#9C27B0', '#7B1FA2', '#4A148C'),    # purple
    }

    for ent_name, joints_raw in entities.items():
        if ent_name not in entity_configs:
            continue
        names, parents = entity_configs[ent_name]
        joints = _swap_yz(_fix_zero_joints(joints_raw, parents))
        colors = entity_colors.get(ent_name, ('#666', '#444', '#222'))
        J = len(names)

        # Draw bones
        for j in range(J):
            if parents[j] >= 0:
                p = parents[j]
                color = get_bone_color(names[j], names[p])
                # Tint based on entity
                if ent_name == 'npc':
                    color = '#42A5F5' if 'L_' in names[j] else '#EF5350' if 'R_' in names[j] else '#66BB6A'
                ax.plot([joints[j, 0], joints[p, 0]],
                        [joints[j, 1], joints[p, 1]],
                        [joints[j, 2], joints[p, 2]],
                        color=color, linewidth=2.0, solid_capstyle='round', alpha=0.9)

        # Draw joints
        ax.scatter(joints[:, 0], joints[:, 1], joints[:, 2],
                   c=colors[0], s=15, depthshade=True, alpha=0.8, zorder=5)

        # Mark root
        ax.scatter([joints[0, 0]], [joints[0, 1]], [joints[0, 2]],
                   c=colors[1], s=50, marker='o', depthshade=True, zorder=6,
                   label=ent_name)

    # Floor grid at minimum Z (vertical)
    floor_z = bounds['zmin']
    grid_range = max(bounds['xmax'] - bounds['xmin'],
                     bounds['ymax'] - bounds['ymin']) * 0.4
    cx = (bounds['xmax'] + bounds['xmin']) / 2
    cy = (bounds['ymax'] + bounds['ymin']) / 2
    for i in np.linspace(-grid_range, grid_range, 7):
        ax.plot([cx+i, cx+i], [cy-grid_range, cy+grid_range], [floor_z, floor_z],
                color='#BDBDBD', linewidth=0.3, alpha=0.4)
        ax.plot([cx-grid_range, cx+grid_range], [cy+i, cy+i], [floor_z, floor_z],
                color='#BDBDBD', linewidth=0.3, alpha=0.4)

    ax.set_xlim(bounds['xmin'], bounds['xmax'])
    ax.set_ylim(bounds['ymin'], bounds['ymax'])
    ax.set_zlim(bounds['zmin'], bounds['zmax'])
    ax.set_xlabel('X', fontsize=8)
    ax.set_ylabel('Z', fontsize=8)
    ax.set_zlabel('Y (up)', fontsize=8)

    elev = 20
    azim = -60 + frame_idx * 0.3
    ax.view_init(elev=elev, azim=azim)

    ax.legend(loc='upper right', fontsize=8)
    ax.set_title(f'{title}  frame {frame_idx}/{total_frames}', fontsize=11)

    fig.tight_layout()
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    img = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)
    plt.close(fig)
    return img


def compute_bounds(all_positions, pad=2.0):
    """Compute cubic bounds from multiple entity positions.

    Positions are in game space (Y=up). Bounds are computed in render space
    (Y<->Z swapped) so matplotlib Z = vertical.
    """
    all_pts = np.concatenate([p.reshape(-1, 3) for p in all_positions.values()], axis=0)
    # Swap Y<->Z for render space
    swapped = all_pts.copy()
    swapped[:, 1], swapped[:, 2] = all_pts[:, 2].copy(), all_pts[:, 1].copy()
    mins = swapped.min(axis=0)
    maxs = swapped.max(axis=0)
    center = (mins + maxs) / 2
    half_range = (maxs - mins).max() / 2 + pad
    return {
        'xmin': center[0] - half_range, 'xmax': center[0] + half_range,
        'ymin': center[1] - half_range, 'ymax': center[1] + half_range,
        'zmin': center[2] - half_range, 'zmax': center[2] + half_range,
    }


def compute_frame_bounds(entities, entity_configs=None, pad=3.0):
    """Compute bounds centered on all entities for a single frame."""
    pts_list = []
    for ent_name, j in entities.items():
        if entity_configs and ent_name in entity_configs:
            _, parents = entity_configs[ent_name]
            j = _fix_zero_joints(j, parents)
        pts_list.append(j.reshape(-1, 3))
    all_pts = np.concatenate(pts_list, axis=0)
    # Swap Y<->Z for render space
    swapped = all_pts.copy()
    swapped[:, 1], swapped[:, 2] = all_pts[:, 2].copy(), all_pts[:, 1].copy()
    mins = swapped.min(axis=0)
    maxs = swapped.max(axis=0)
    center = (mins + maxs) / 2
    half_range = max((maxs - mins).max() / 2 + pad, 5.0)  # minimum 5 units
    return {
        'xmin': center[0] - half_range, 'xmax': center[0] + half_range,
        'ymin': center[1] - half_range, 'ymax': center[1] + half_range,
        'zmin': center[2] - half_range, 'zmax': center[2] + half_range,
    }


def render_video(positions_dict, entity_configs, output_path, fps=20,
                 title="", max_frames=0):
    """Render full video with camera following the characters.

    Args:
        positions_dict: dict of entity -> (T, J, 3)
        entity_configs: dict of entity -> (names, parents)
        output_path: output MP4 path
        fps: frames per second
        title: video title
        max_frames: limit frames (0=all)
    """
    # Determine frame count
    T = min(p.shape[0] for p in positions_dict.values())
    if max_frames > 0:
        T = min(T, max_frames)

    # Trim positions
    trimmed = {k: v[:T] for k, v in positions_dict.items()}

    # Compute stable bounds over a sliding window for smooth camera
    all_bounds = []
    for t in range(T):
        frame_data = {k: v[t] for k, v in trimmed.items()}
        all_bounds.append(compute_frame_bounds(frame_data, entity_configs))

    # Smooth bounds with a window of 20 frames
    smooth_window = min(20, T)
    smoothed_bounds = []
    for t in range(T):
        start = max(0, t - smooth_window // 2)
        end = min(T, t + smooth_window // 2 + 1)
        avg_bounds = {}
        for key in ['xmin', 'xmax', 'ymin', 'ymax', 'zmin', 'zmax']:
            avg_bounds[key] = np.mean([all_bounds[i][key] for i in range(start, end)])
        smoothed_bounds.append(avg_bounds)

    print(f"Rendering {T} frames (camera follows characters)")

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)

    with tempfile.TemporaryDirectory() as tmpdir:
        for t in range(T):
            frame_data = {k: v[t] for k, v in trimmed.items()}
            img = render_frame(frame_data, t, T, smoothed_bounds[t], entity_configs, title)
            plt.imsave(os.path.join(tmpdir, f'frame_{t:05d}.png'), img)
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
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, help='motion_data.npz path')
    parser.add_argument('--segment', type=int, default=0)
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--end', type=int, default=200)
    parser.add_argument('--out', type=str, default='output/vis/sample.mp4')
    parser.add_argument('--fps', type=int, default=20)
    parser.add_argument('--title', type=str, default='')
    args = parser.parse_args()

    skel_dir = PROJECT_ROOT / "data" / "skeleton"
    entity_configs = {}
    joint_counts = {}
    for ent, skel_file in [('monster', 'em_19_pruned.json'),
                            ('npc', 'npc_pruned.json'),
                            ('weapon', 'wp_4.json')]:
        names, parents = parse_skeleton(str(skel_dir / skel_file))
        entity_configs[ent] = (names, parents)
        joint_counts[ent] = len(names)

    if args.data:
        data = np.load(args.data, allow_pickle=True)
        mean = data['mean']
        std = data['std']
        segment = data[f'segment_{args.segment}']

        # Load initial absolute positions if available
        init_key = f'init_pos_{args.segment}_monster'
        if init_key in data:
            init_positions = {
                'monster': data[f'init_pos_{args.segment}_monster'],
                'npc': data[f'init_pos_{args.segment}_npc'],
                'weapon': data[f'init_pos_{args.segment}_weapon'],
            }
            # If start > 0, accumulate deltas to get init pos at start frame
            if args.start > 0:
                pre = segment[:args.start] * std + mean
                pre_T = pre.shape[0]
                m_J = joint_counts['monster']
                n_J = joint_counts['npc']
                w_J = joint_counts['weapon']
                init_positions['monster'] = init_positions['monster'] + pre[:, :m_J*3].reshape(pre_T, m_J, 3).sum(axis=0)
                init_positions['npc'] = init_positions['npc'] + pre[:, m_J*9:m_J*9+n_J*3].reshape(pre_T, n_J, 3).sum(axis=0)
                init_positions['weapon'] = init_positions['weapon'] + pre[:, m_J*9+n_J*9:m_J*9+n_J*9+w_J*3].reshape(pre_T, w_J, 3).sum(axis=0)
        else:
            init_positions = None

        segment = segment[args.start:args.end]
        positions = features_to_positions(segment, mean, std, joint_counts, init_positions)

        title = args.title or f'Segment {args.segment} [{args.start}:{args.end}]'
        render_video(positions, entity_configs, args.out, args.fps, title)


if __name__ == '__main__':
    main()
