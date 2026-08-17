#!/usr/bin/env python3
"""Check that the counterfactual pair is actually a controlled comparison.

Two videos that differ throughout prove nothing. The action stream is sampled, so two runs of
the *same* command already differ. What makes a difference attributable to the command is a
shared prefix that ends exactly where the command changes.

So this asserts the arithmetic the demo claims:

    frames before the commanded switch : identical
    the first differing frame          : the switch

    python3 demos/check_counterfactual.py            # exit 0 if the claim holds

Compares decoded pixels, not file bytes: the two are separate encodes, and an encoder is not
obliged to produce identical bytes for identical frames.
"""
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, "out", "03_counterfactual")
A, B = os.path.join(D, "stay.mp4"), os.path.join(D, "switch.mp4")
# Where the switch demo overrides the action, as a fraction of the horizon. Must match the
# bseqn block lengths in control_demos.sh (300 of 400 = 0.75).
SWITCH_FRAC = 0.75
FPS = 30            # output fps
TOL = 2             # per-channel tolerance: both sides go through a lossy encoder

try:
    import imageio_ffmpeg
    FF = imageio_ffmpeg.get_ffmpeg_exe()
except Exception:
    FF = "ffmpeg"


def decode(path):
    probe = subprocess.run([FF, "-i", path], capture_output=True, text=True).stderr
    w = h = None
    for line in probe.split("\n"):
        if "Video:" in line:
            for tok in line.split(","):
                tok = tok.strip().split(" ")[0]
                if "x" in tok and tok.split("x")[0].isdigit():
                    w, h = (int(v) for v in tok.split("x"))
                    break
            break
    if not w:
        sys.exit(f"cannot probe {path}")
    p = subprocess.run([FF, "-v", "error", "-i", path, "-f", "rawvideo",
                        "-pix_fmt", "rgb24", "-"], capture_output=True)
    buf = np.frombuffer(p.stdout, np.uint8)
    n = buf.size // (w * h * 3)
    return buf[: n * w * h * 3].reshape(n, h, w, 3)


def main():
    for p in (A, B):
        if not os.path.exists(p):
            sys.exit(f"missing {p}\nrun: ONLY=counterfactual bash demos/control_demos.sh")
    a, b = decode(A), decode(B)
    n = min(len(a), len(b))
    switch = int(round(n * SWITCH_FRAC))
    print(f"stay   : {len(a)} frames")
    print(f"switch : {len(b)} frames")
    print(f"commanded switch at frame {switch}  ({switch / FPS:.1f} s)\n")

    diff = np.array([int(np.abs(a[i].astype(np.int16) - b[i].astype(np.int16)).max())
                     for i in range(n)])
    first = next((i for i in range(n) if diff[i] > TOL), None)

    if first is None:
        print("the two videos never differ -- the forced action had no effect at all")
        return 2
    # Report the prefix STRICTLY BEFORE the first difference. Measuring "up to the commanded
    # switch" would include the first differing frame and print a large number next to the
    # claim that the prefix is identical.
    print(f"identical frames      : 0..{first - 1}  (max diff {int(diff[:first].max())})")
    print(f"first differing frame : {first}  ({first / FPS:.1f} s)")
    print(f"mean diff after it    : "
          f"{float(np.abs(a[first:n].astype(np.int16) - b[first:n].astype(np.int16)).mean()):.2f}")

    # One frame of slack: the rollout runs at 20 fps and the video is written at 30, so the
    # generated frame where the override begins lands between two output frames and the
    # resampler can blend it into the one before.
    ok = abs(first - switch) <= 1
    print()
    if ok:
        print(f"PASS: identical up to the commanded switch (expected frame {switch}, "
              f"observed {first}),")
        print("      then divergent. Whatever separates them was caused by that one action id.")
        return 0
    print(f"FAIL: they diverge at frame {first}, not at the commanded switch at {switch}.")
    print("      Something other than the forced action differs between the two runs;")
    print("      the comparison is not controlled and proves nothing.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
