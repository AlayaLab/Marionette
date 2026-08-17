#!/usr/bin/env python3
"""Offline port of the collection-side camera_tracker.lua (sphere model + spring-arm
terrain-occlusion avoidance). Faithful port of camera_update() — same constants and
control flow — with the in-game raycast replaced by a caller-supplied `is_blocked`
callback (render-based terrain occlusion test in the offline pipeline).

camera_update(npc_pos, monster_pos, state, dt, is_blocked=None)
  npc_pos/monster_pos: (3,) world root positions (Y-up, metres)
  state: dict, all keys None on first call (mutated in place)
  is_blocked(from_pt, to_pt) -> (blocked: bool, clearance: float)
      from_pt = look_target (NPC head), to_pt = candidate eye.
      clearance: higher = less occluded (used to rank blocked candidates).
      If None, spring-arm is skipped (plain sphere camera).
  returns (eye(3,), focus(3,), fwd(3,), fov_deg)
"""
import math
import numpy as np

CFG = dict(
    ARM=6.15, CAM_HEIGHT_ABOVE_NPC=2.41, YAW_OFFSET_DEG=0.0,
    NPC_LOOK_HEIGHT=1.68, MON_LOOK_HEIGHT=1.68, FOCUS_SHIFT_MAX=3.0,
    FOV_DEG=42.0, FOV_WIDEN_START_DEG=20.0, FOV_WIDEN_K=1.0, FOV_MAX_DEG=75.0, FOV_SMOOTH_TAU=0.6,
    PITCH_FOLLOW=0.5, PITCH_DOWN_FOLLOW=0.3, PITCH_MAX_DEG=50.0, PITCH_BASELINE_DEG=-5.0,
    CAM_MAX_SPEED=40.0, CAM_MAX_ACCEL=80.0, FOCUS_MAX_SPEED=30.0, FOCUS_MAX_ACCEL=60.0,
    TELEPORT_SNAP_DIST=20.0,
    YAW_DEADZONE_DEG=12.0, YAW_RATE_DPS=150.0, YAW_SOFT_TAU=0.3,
    BATTLE_DIR_MIN_DIST=1.5, BATTLE_DIR_TAU=0.10,
    SPRING_ARM_ENABLED=True, SPRING_ARM_OFFSET=0.3,
    SPRING_ARM_YAW_STEP=15.0, SPRING_ARM_YAW_MAX=90.0,
    SPRING_ARM_PITCH_STEP=10.0, SPRING_ARM_PITCH_MAX=40.0,
)


def _v(x, y, z): return np.array([x, y, z], float)
def _norm(a):
    L = float(np.linalg.norm(a))
    return _v(0, 0, 1) if L < 1e-8 else a / L
def _clamp(x, lo, hi): return max(lo, min(hi, x))
def _ema(dt, tau): return 1.0 - math.exp(-dt / max(tau, 0.001))
def _rot_y(v, yaw):
    c, s = math.cos(yaw), math.sin(yaw)
    return _v(v[0] * c + v[2] * s, v[1], -v[0] * s + v[2] * c)
def _signed_yaw(a, b):
    cos_a = _clamp(float(np.dot(a, b)), -1.0, 1.0)
    cross_y = a[2] * b[0] - a[0] * b[2]
    return math.degrees(math.atan2(cross_y, cos_a))


def _smooth_move(cur, target, vel, dt, max_speed, max_accel):
    diff = target - cur
    dist = float(np.linalg.norm(diff))
    if dist < 0.001:
        return target.copy(), _v(0, 0, 0)
    d = diff / dist
    desired = min(max_speed, math.sqrt(2.0 * max_accel * dist))
    dv = d * desired - vel
    dvm = float(np.linalg.norm(dv))
    if dvm > max_accel * dt:
        dv *= (max_accel * dt) / dvm
    nv = vel + dv
    ns = float(np.linalg.norm(nv))
    if ns > max_speed:
        nv *= max_speed / ns
    npos = cur + nv * dt
    if float(np.dot(target - npos, diff)) <= 0:
        return target.copy(), _v(0, 0, 0)
    return npos, nv


def camera_update(npc_pos, monster_pos, st, dt, is_blocked=None, cfg=CFG):
    npc_pos = np.asarray(npc_pos, float); monster_pos = np.asarray(monster_pos, float)

    # STEP 0: teleport snap
    if st.get("last_npc_pos") is not None and cfg["TELEPORT_SNAP_DIST"] > 0:
        if float(np.linalg.norm(npc_pos - st["last_npc_pos"])) > cfg["TELEPORT_SNAP_DIST"]:
            for k in ("eye", "eye_vel", "focus", "focus_vel", "fwd_dir",
                      "battle_dir", "horiz_dir", "fov", "spring_lock_sign"):
                st[k] = None
    st["last_npc_pos"] = npc_pos.copy()

    # STEP 1: battle direction (XZ, monster->npc), low-pass smoothed
    diff = npc_pos - monster_pos
    diff_xz = _v(diff[0], 0.0, diff[2]); dist_xz = float(np.linalg.norm(diff_xz))
    if dist_xz < cfg["BATTLE_DIR_MIN_DIST"]:
        if st.get("battle_dir") is not None: bd_raw = st["battle_dir"].copy()
        elif dist_xz > 0.001: bd_raw = _norm(diff_xz)
        else: bd_raw = _v(0, 0, 1)
    else:
        bd_raw = _norm(diff_xz)
    if st.get("battle_dir") is None:
        battle_dir = bd_raw.copy()
    else:
        sm = st["battle_dir"] + (bd_raw - st["battle_dir"]) * _ema(dt, cfg["BATTLE_DIR_TAU"])
        L = float(np.linalg.norm(sm)); battle_dir = sm / L if L > 1e-6 else st["battle_dir"].copy()

    # STEP 2: orbit yaw (deadzone + rate-limit)
    ideal_horiz = _rot_y(battle_dir, math.radians(cfg["YAW_OFFSET_DEG"]))
    if st.get("horiz_dir") is None:
        horiz_dir = ideal_horiz.copy()
    else:
        ang = _signed_yaw(st["horiz_dir"], ideal_horiz); aa = abs(ang)
        if aa <= cfg["YAW_DEADZONE_DEG"]:
            horiz_dir = st["horiz_dir"].copy()
        else:
            excess = aa - cfg["YAW_DEADZONE_DEG"]
            step = min(cfg["YAW_RATE_DPS"] * dt,
                       excess * (1.0 - math.exp(-dt / max(cfg["YAW_SOFT_TAU"], 0.001))), excess)
            horiz_dir = _rot_y(st["horiz_dir"], math.radians((1.0 if ang > 0 else -1.0) * step))

    # STEP 3: pitch (sphere elevation) from height diff
    height_diff = monster_pos[1] - npc_pos[1]
    npc_to_mon_xz = float(np.linalg.norm(_v(monster_pos[0] - npc_pos[0], 0, monster_pos[2] - npc_pos[2])))
    ideal_pitch = math.atan2(height_diff, max(npc_to_mon_xz, 0.1))
    if height_diff < 0:
        pitch = abs(ideal_pitch) * cfg["PITCH_DOWN_FOLLOW"] + math.radians(cfg["PITCH_BASELINE_DEG"])
    else:
        pitch = ideal_pitch * cfg["PITCH_FOLLOW"] + math.radians(cfg["PITCH_BASELINE_DEG"])
    pm = math.radians(cfg["PITCH_MAX_DEG"]); pitch = _clamp(pitch, -pm, pm)

    # STEP 4: focus = npc anchor + clamped shift toward monster
    npc_anchor = npc_pos + _v(0, cfg["NPC_LOOK_HEIGHT"], 0)
    mon_anchor = monster_pos + _v(0, cfg["MON_LOOK_HEIGHT"], 0)
    to_mon = mon_anchor - npc_anchor; tl = float(np.linalg.norm(to_mon))
    look_target = npc_anchor + to_mon * (min(cfg["FOCUS_SHIFT_MAX"], tl) / tl) if tl > 0.01 else npc_anchor.copy()

    # STEP 5: sphere position
    arm = cfg["ARM"]; sphere_cy = npc_pos[1] + cfg["CAM_HEIGHT_ABOVE_NPC"]
    def make_eye(hdir, p=pitch):
        axz, ay = arm * math.cos(p), arm * math.sin(p)
        return _v(npc_pos[0] + hdir[0] * axz, sphere_cy + ay, npc_pos[2] + hdir[2] * axz)
    desired_eye = make_eye(horiz_dir)

    # STEP 6: spring-arm two-stage occlusion dodge (yaw scan -> pitch lift), side-locked
    if cfg["SPRING_ARM_ENABLED"] and is_blocked is not None:
        step_r = math.radians(cfg["SPRING_ARM_YAW_STEP"]); max_r = math.radians(cfg["SPRING_ARM_YAW_MAX"])
        center_eye = make_eye(horiz_dir)
        cblocked, _c = is_blocked(look_target, center_eye)
        best_eye, best_sign, found = None, None, False
        if not cblocked:
            st["spring_lock_sign"] = None; best_eye = center_eye; found = True
        else:
            signs = [st["spring_lock_sign"]] if st.get("spring_lock_sign") in (1, -1) else [1, -1]
            best_score = -9999.0
            for sign in signs:
                angle = step_r
                while angle <= max_r + 1e-4:
                    te = make_eye(_rot_y(horiz_dir, sign * angle))
                    blk, clr = is_blocked(look_target, te)
                    if not blk:
                        score = 10000.0 - math.degrees(angle); found = True
                    else:
                        score = clr
                    if score > best_score:
                        best_score, best_eye, best_sign = score, te, sign
                    angle += step_r
            if st.get("spring_lock_sign") is None and best_sign is not None:
                st["spring_lock_sign"] = best_sign
        if not found:  # Stage B: pitch lift on original horiz_dir
            ps = math.radians(cfg["SPRING_ARM_PITCH_STEP"]); pmx = math.radians(cfg["SPRING_ARM_PITCH_MAX"])
            lift = ps
            while lift <= pmx + 1e-4:
                lifted = pitch + lift
                if lifted >= math.radians(85): break
                te = make_eye(horiz_dir, lifted)
                blk, _c = is_blocked(look_target, te)
                if not blk:
                    best_eye = te; break
                lift += ps
        if best_eye is not None:
            desired_eye = best_eye

    # STEP 7: velocity-based smoothing
    ev = st.get("eye_vel"); fv = st.get("focus_vel")
    ev = _v(0, 0, 0) if ev is None else ev
    fv = _v(0, 0, 0) if fv is None else fv
    if st.get("eye") is None:
        eye, eye_vel = desired_eye.copy(), _v(0, 0, 0)
    else:
        eye, eye_vel = _smooth_move(st["eye"], desired_eye, ev, dt, cfg["CAM_MAX_SPEED"], cfg["CAM_MAX_ACCEL"])
    if st.get("focus") is None:
        focus, focus_vel = look_target.copy(), _v(0, 0, 0)
    else:
        focus, focus_vel = _smooth_move(st["focus"], look_target, fv, dt, cfg["FOCUS_MAX_SPEED"], cfg["FOCUS_MAX_ACCEL"])

    fwd_dir = _norm(eye - focus)  # RE: +Z away from target

    # STEP 7.6: dynamic FOV
    v_ang = math.degrees(math.atan2(abs(height_diff), max(npc_to_mon_xz, 0.1)))
    excess = max(0.0, v_ang - cfg["FOV_WIDEN_START_DEG"])
    fov_t = min(cfg["FOV_MAX_DEG"], cfg["FOV_DEG"] + excess * cfg["FOV_WIDEN_K"])
    fov = fov_t if st.get("fov") is None else st["fov"] + (fov_t - st["fov"]) * _ema(dt, cfg["FOV_SMOOTH_TAU"])

    st.update(eye=eye.copy(), eye_vel=eye_vel.copy(), focus=focus.copy(), focus_vel=focus_vel.copy(),
              fwd_dir=fwd_dir.copy(), battle_dir=battle_dir.copy(), horiz_dir=horiz_dir.copy(), fov=fov)
    return eye, focus, fwd_dir, fov
