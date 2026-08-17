#!/usr/bin/env python3
"""Turn the control demos into one PNG each: one row per variant, one column per moment.

The columns are sampled at **the centre of each commanded phase**, not at evenly spaced times.
A schedule of four five-second blocks wants four columns at 2.5/7.5/12.5/17.5 s; sampling five
evenly spaced points instead lands them off the blocks and mislabels every one. The demo's
schedule and this table have to agree, so both are written down here explicitly.

    python3 demos/make_control_sheet.py             # every demo present in demos/out/
    python3 demos/make_control_sheet.py 02_counterfactual
"""
import os
import subprocess
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")

try:
    import imageio_ffmpeg
    FF = imageio_ffmpeg.get_ffmpeg_exe()
except Exception:
    FF = "ffmpeg"

FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FONT_B = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
BG, FG, DIM, HOT = (10, 10, 13), (232, 232, 236), (150, 150, 158), (255, 122, 90)
CELL_W, PAD = 400, 10

# name: (title, subtitle, [(column label, seconds)], note)
DEMOS = {
    "01_free": (
        "No control", "both characters act on their own -- the baseline everything else differs from",
        [("t = 2s", 2), ("t = 6s", 6), ("t = 10s", 10), ("t = 14s", 14), ("t = 18s", 18)],
        "camera locked to the hunter, flat floor -- same framing as the action demo"),
    "02_action": (
        "Commanded actions",
        "one action id written into the hunter's stream every five seconds; flat floor, camera locked to the hunter",
        [("IDLE", 2.5), ("ATTACK", 7.5), ("DODGE", 12.5), ("HEAVY (re-trigger)", 17.5)],
        "the monster is NOT controlled -- it stays free and wanders through"),
    "03_counterfactual": (
        "One id apart",
        "same seed, window, RNG, camera and terrain; the hunter's held action changes from IDLE to ATTACK at 15 s",
        [("4s  same", 4), ("10s  same", 10), ("15.5s  +0.5", 15.5), ("17s", 17), ("19.5s", 19.5)],
        "the first 15 s are identical frame for frame -- verify with demos/check_counterfactual.py"),
    "04_move": (
        "Commanded movement",
        "root translation and yaw overwritten AND a gait commanded; four seconds spinning, then a heading every two",
        [("spin", 2), ("heading 0", 5), ("heading 90", 7), ("heading 180", 9), ("heading 270", 11)],
        "driving the root without commanding a gait makes the character slide -- both are needed"),
}


def decode(path):
    probe = subprocess.run([FF, "-i", path], capture_output=True, text=True).stderr
    w = h = None
    fps = 30.0
    for line in probe.split("\n"):
        if "Video:" in line:
            for tok in line.split(","):
                t = tok.strip()
                if t.endswith(" fps"):
                    try:
                        fps = float(t[:-4])
                    except ValueError:
                        pass
                t = t.split(" ")[0]
                if "x" in t and t.split("x")[0].isdigit():
                    w, h = (int(v) for v in t.split("x"))
            break
    if not w:
        raise SystemExit(f"cannot probe {path}")
    p = subprocess.run([FF, "-v", "error", "-i", path, "-f", "rawvideo",
                        "-pix_fmt", "rgb24", "-"], capture_output=True)
    buf = np.frombuffer(p.stdout, np.uint8)
    n = buf.size // (w * h * 3)
    return buf[: n * w * h * 3].reshape(n, h, w, 3), fps


def sheet(name):
    d = os.path.join(OUT, name)
    vids = sorted(f for f in os.listdir(d) if f.endswith(".mp4")) if os.path.isdir(d) else []
    if not vids:
        print(f"{name}: nothing in {d}, skipped")
        return
    title, sub, cols, note = DEMOS.get(name, (name, "", [(f"t{i}", i * 4) for i in range(5)], ""))

    rows = []
    for v in vids:
        arr, fps = decode(os.path.join(d, v))
        dur = len(arr) / fps
        picked = []
        for lab, t in cols:
            i = int(round(t * fps))
            if i >= len(arr):      # asking past the end is a schedule/horizon mismatch, not a
                i = len(arr) - 1   # rendering detail -- say so rather than silently clamping
                print(f"  {name}/{v}: column '{lab}' at {t}s is past the clip ({dur:.1f}s)")
            picked.append(arr[i])
        rows.append((os.path.splitext(v)[0], picked, dur))

    ncol = len(cols)
    fh = int(CELL_W * rows[0][1][0].shape[0] / rows[0][1][0].shape[1])
    head, label_h, gap = 74, 24, 8
    W = ncol * CELL_W + (ncol + 1) * PAD
    H = head + PAD + len(rows) * (label_h + fh + gap) + 22 + (20 if note else 0)
    im = Image.new("RGB", (W, H), BG)
    dr = ImageDraw.Draw(im)
    dr.text((PAD + 2, 16), title, font=ImageFont.truetype(FONT_B, 26), fill=FG)
    dr.text((PAD + 2, 48), sub, font=ImageFont.truetype(FONT, 15), fill=DIM)
    f_small = ImageFont.truetype(FONT, 14)
    f_col = ImageFont.truetype(FONT_B, 14)

    y = head + PAD
    for label, fr, dur in rows:
        dr.text((PAD + 2, y), f"{label}   ({dur:.1f}s)", font=f_small, fill=FG)
        yy = y + label_h
        for c, f in enumerate(fr):
            img = Image.fromarray(f).resize((CELL_W, fh), Image.LANCZOS)
            im.paste(img, (PAD + c * (CELL_W + PAD), yy))
        y = yy + fh + gap
    for c, (lab, _) in enumerate(cols):
        dr.text((PAD + c * (CELL_W + PAD) + 2, y + 2), lab, font=f_col, fill=DIM)
    if note:
        dr.text((PAD + 2, y + 22), note, font=f_small, fill=HOT)

    p = os.path.join(HERE, f"{name}.png")
    im.save(p, optimize=True)
    print(f"{name}: {len(rows)} variant(s) -> {p} ({os.path.getsize(p)/1e6:.2f} MB)")


if __name__ == "__main__":
    for n in (sys.argv[1:] or sorted(DEMOS)):
        sheet(n)
