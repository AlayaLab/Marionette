"""Naive-AR baseline rollout for Wan2.2-Fun-5B-Control (ckpt-3000, v1).

For each sample:
  chunk 0 : ref = GT[start_frame],            pose = GT_pose[start:start+L]
  chunk i>0: ref = generated[chunk_{i-1}][-1], pose = GT_pose[start+iL:start+(i+1)L]

No history conditioning beyond a single-frame ref → drift baseline for Helios.

Outputs (under SAVE_DIR/<seg>/):
  chunk_00.mp4 ... chunk_04.mp4      (each 81 frames)
  rollout.mp4                        (5×81 = 405 frames concat)
  gt.mp4                             (matching GT 405 frames)
  pose.mp4                           (input control video, 405 frames)
  comparison.mp4                     (GT | rollout side-by-side)
"""
import os, sys, json, gc, time, tempfile

import cv2
import imageio
import numpy as np
import torch
from PIL import Image
from omegaconf import OmegaConf
from decord import VideoReader, cpu
from diffusers import FlowMatchEulerDiscreteScheduler
from transformers import AutoTokenizer

CUR = os.path.abspath(__file__)
ROOTS = [os.path.dirname(CUR), os.path.dirname(os.path.dirname(CUR)), os.path.dirname(os.path.dirname(os.path.dirname(CUR)))]
for r in ROOTS:
    if r not in sys.path: sys.path.insert(0, r)

from videox_fun.dist import set_multi_gpus_devices
from videox_fun.models import (AutoencoderKLWan, AutoencoderKLWan3_8, AutoTokenizer, CLIPModel,
                               WanT5EncoderModel, Wan2_2Transformer3DModel)
from videox_fun.models.cache_utils import get_teacache_coefficients
from videox_fun.pipeline import Wan2_2FunControlPipeline
from videox_fun.utils.utils import (filter_kwargs, get_image_to_video_latent,
                                    get_video_to_video_latent, save_videos_grid)
from videox_fun.utils.fm_solvers import FlowDPMSolverMultistepScheduler
from videox_fun.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler


# ======================== config ==========================
# Defaults point at the weights and data shipped with this project, so the script runs with no
# environment set at all. MARIONETTE_ROOT overrides the project root.
_MROOT = os.environ.get("MARIONETTE_ROOT") or os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

CONFIG_PATH    = os.environ.get("WAN2_2_CONFIG_PATH", "config/wan2.2/wan_civitai_5b.yaml")
MODEL_NAME     = os.environ.get("WAN2_2_MODEL_PATH",
                                os.path.join(_MROOT, "weights/Wan2.2-Fun-5B-Control"))
# The checkpoint and the prompt template below are ONE pair and must be changed together: the
# released model was trained on text carrying `hunter appearance id N`, and feeding it the
# older prompt (or feeding an older checkpoint this one) puts the model out of distribution and
# monsters drop out of the rollout. This bit us once already.
CKPT_PATH      = os.environ.get("WAN2_2_TRANSFORMER_PATH",
                                os.path.join(_MROOT, "weights/observation/diffusion_pytorch_model.safetensors"))

DATASET_BASE   = os.environ.get("MH_DATASET_BASE", os.path.join(_MROOT, "samples"))
SAVE_DIR       = os.environ.get("WAN2_2_SAVE_PATH", "samples/ar_baseline_v2pose")

# Upstream defaults to three long clips from the development corpus, which is not part of the
# release. Default to the sample the demo produces, so running this file directly does
# something instead of failing on a missing directory.
SAMPLES = os.environ.get("MH_SAMPLES", "seg133_seed32").split(",")

# Start offset into each sample. The demo sample begins where the rollout begins, so 0.
_default_starts = "0"
START_FRAMES_STR = os.environ.get("MH_START_FRAMES", _default_starts)
START_FRAMES = [int(x) for x in START_FRAMES_STR.split(",")]
START_FRAME    = int(os.environ.get("MH_START_FRAME", "500"))  # legacy/single-value override
N_CHUNKS       = int(os.environ.get("MH_N_CHUNKS", "5").split(",")[0])  # default / single value
# per-sample chunk counts: "8,8,8,115" parallels MH_SAMPLES; falls back to N_CHUNKS
N_CHUNKS_LIST  = [int(x) for x in os.environ.get("MH_N_CHUNKS", str(N_CHUNKS)).split(",")]
CHUNK_LEN      = int(os.environ.get("MH_CHUNK_LEN", "81"))  # matches training
# The training resolution. Upstream defaults to 512x896, and inferring this checkpoint there
# does not merely lose detail -- it visibly wrecks the image. Changed here rather than left to
# the runner script, because the trap is invisible: 512 produces a plausible-looking video.
SAMPLE_H       = int(os.environ.get("WAN2_2_SAMPLE_H", "704"))
SAMPLE_W       = int(os.environ.get("WAN2_2_SAMPLE_W", "1280"))
FPS            = int(os.environ.get("WAN2_2_FPS", "30"))
NUM_STEPS      = int(os.environ.get("WAN2_2_NUM_INFERENCE_STEPS", "40"))
GUIDANCE       = float(os.environ.get("WAN2_2_GUIDANCE_SCALE", "6.0"))
SEED           = int(os.environ.get("WAN2_2_SEED", "43"))

PROMPT_TMPL    = os.environ.get(
    "WAN2_2_PROMPT",
    "Monster Hunter Wilds video game gameplay, stage 101, hunter appearance id 9 wielding "
    "weapon type 4, fighting monster id 19.")

# Train/infer consistency: if MH_MANIFEST is set, use each segment's EXACT
# training prompt (keyed by the segment directory name) instead of PROMPT_TMPL.
# This guarantees the npc-aware template (stage/appearance/weapon/monster ids)
# matches what the model saw during training for that very clip.
MANIFEST_PATH  = os.environ.get("MH_MANIFEST", "")
PROMPT_BY_SEG  = {}
if MANIFEST_PATH:
    with open(MANIFEST_PATH) as _f:
        for _e in json.load(_f):
            _name = os.path.basename(os.path.dirname(_e["file_path"]))
            PROMPT_BY_SEG[_name] = _e.get("text", "")
    print(f"loaded {len(PROMPT_BY_SEG)} per-segment prompts from {MANIFEST_PATH}")
NEG_PROMPT     = "色调艳丽，过曝，静态，细节模糊不清，字幕，静止，整体发灰，最差质量，低质量，JPEG压缩残留"

WEIGHT_DTYPE   = torch.bfloat16
sample_size    = [SAMPLE_H, SAMPLE_W]

# ============================================================


def video_read_range(path, start, n, ctx=cpu(0)):
    """Read n consecutive frames starting at `start` from a video. Returns [n, H, W, 3] uint8."""
    vr = VideoReader(path, ctx=ctx)
    total = len(vr)
    end = min(start + n, total)
    idxs = np.arange(start, end, dtype=int)
    frames = vr.get_batch(idxs).asnumpy()
    if frames.shape[0] < n:
        # pad with last frame
        pad = np.tile(frames[-1:], (n - frames.shape[0], 1, 1, 1))
        frames = np.concatenate([frames, pad], axis=0)
    return frames


def save_video(frames_np, out_path, fps=30):
    """frames_np: [T,H,W,3] uint8 RGB."""
    writer = imageio.get_writer(out_path, fps=fps, codec="libx264", pixelformat="yuv420p", quality=8)
    for f in frames_np:
        writer.append_data(f)
    writer.close()


def side_by_side(gt_np, rollout_np, out_path, fps=30):
    """concat horizontally [T,H,W,3] each."""
    T = min(gt_np.shape[0], rollout_np.shape[0])
    H = max(gt_np.shape[1], rollout_np.shape[1])
    writer = imageio.get_writer(out_path, fps=fps, codec="libx264", pixelformat="yuv420p", quality=8)
    for t in range(T):
        a = gt_np[t]; b = rollout_np[t]
        if a.shape[0] != H: a = cv2.resize(a, (a.shape[1] * H // a.shape[0], H))
        if b.shape[0] != H: b = cv2.resize(b, (b.shape[1] * H // b.shape[0], H))
        writer.append_data(np.hstack([a, b]))
    writer.close()


# ===================== load model ==========================
print("=" * 60)
print(f"naive-AR baseline rollout")
print(f"  CKPT: {CKPT_PATH}")
print(f"  samples: {len(SAMPLES)}  chunks_per: {N_CHUNKS}  chunk_len: {CHUNK_LEN}")
print(f"  resolution: {sample_size}  fps: {FPS}  steps: {NUM_STEPS}  cfg: {GUIDANCE}")
print(f"  start_frame: {START_FRAME}")
print("=" * 60)

device = set_multi_gpus_devices(1, 1)
config = OmegaConf.load(CONFIG_PATH)
boundary = config['transformer_additional_kwargs'].get('boundary', 0.875)

transformer = Wan2_2Transformer3DModel.from_pretrained(
    os.path.join(MODEL_NAME, config['transformer_additional_kwargs'].get('transformer_low_noise_model_subpath', 'transformer')),
    transformer_additional_kwargs=OmegaConf.to_container(config['transformer_additional_kwargs']),
    low_cpu_mem_usage=True,
    torch_dtype=WEIGHT_DTYPE,
)
transformer_2 = None  # 5B single-DiT

if CKPT_PATH:
    print(f"Loading ckpt from: {CKPT_PATH}")
    from safetensors.torch import load_file
    sd = load_file(CKPT_PATH) if CKPT_PATH.endswith(".safetensors") else torch.load(CKPT_PATH, map_location="cpu")
    sd = sd.get("state_dict", sd) if isinstance(sd, dict) else sd
    m, u = transformer.load_state_dict(sd, strict=False)
    print(f"  missing: {len(m)}  unexpected: {len(u)}")

Chosen_VAE = {"AutoencoderKLWan": AutoencoderKLWan, "AutoencoderKLWan3_8": AutoencoderKLWan3_8}[
    config['vae_kwargs'].get('vae_type', 'AutoencoderKLWan')]
vae = Chosen_VAE.from_pretrained(
    os.path.join(MODEL_NAME, config['vae_kwargs'].get('vae_subpath', 'vae')),
    additional_kwargs=OmegaConf.to_container(config['vae_kwargs']),
).to(WEIGHT_DTYPE)

tokenizer = AutoTokenizer.from_pretrained(
    os.path.join(MODEL_NAME, config['text_encoder_kwargs'].get('tokenizer_subpath', 'tokenizer')))
text_encoder = WanT5EncoderModel.from_pretrained(
    os.path.join(MODEL_NAME, config['text_encoder_kwargs'].get('text_encoder_subpath', 'text_encoder')),
    additional_kwargs=OmegaConf.to_container(config['text_encoder_kwargs']),
    low_cpu_mem_usage=True, torch_dtype=WEIGHT_DTYPE,
).eval()

scheduler = FlowMatchEulerDiscreteScheduler(
    **filter_kwargs(FlowMatchEulerDiscreteScheduler, OmegaConf.to_container(config['scheduler_kwargs'])))

pipeline = Wan2_2FunControlPipeline(
    transformer=transformer, transformer_2=transformer_2, vae=vae,
    tokenizer=tokenizer, text_encoder=text_encoder, scheduler=scheduler,
).to(device=device)

# round video_length to vae temporal compression
VL = (CHUNK_LEN - 1) // vae.config.temporal_compression_ratio * vae.config.temporal_compression_ratio + 1
assert VL == CHUNK_LEN, f"chunk_len {CHUNK_LEN} not VAE-aligned, got {VL}"


# ===================== rollout loop ==========================
def run_chunk(prompt, pose_frames_np, ref_pil_or_path, seed=SEED):
    """One denoise call. pose_frames_np: [T,H,W,3] uint8; ref: PIL or path."""
    gen = torch.Generator(device=device).manual_seed(seed)
    inpaint_video, inpaint_video_mask, clip_image = get_image_to_video_latent(
        ref_pil_or_path, None, video_length=CHUNK_LEN, sample_size=sample_size)
    # control video: pass numpy list directly to avoid temp-file roundtrip
    input_video, input_video_mask, _, _ = get_video_to_video_latent(
        list(pose_frames_np), video_length=CHUNK_LEN, sample_size=sample_size, fps=FPS, ref_image=None)
    sample = pipeline(
        prompt, num_frames=CHUNK_LEN,
        negative_prompt=NEG_PROMPT,
        height=sample_size[0], width=sample_size[1],
        generator=gen,
        guidance_scale=GUIDANCE, num_inference_steps=NUM_STEPS,
        video=inpaint_video, mask_video=inpaint_video_mask,
        control_video=input_video, control_camera_video=None,
        ref_image=None, boundary=boundary, shift=5,
    ).videos  # [1, 3, T, H, W] in [0,1]
    out = sample[0].permute(1, 2, 3, 0).clamp(0, 1).float().cpu().numpy()  # [T, H, W, 3]
    out_uint8 = (out * 255.0).round().astype(np.uint8)
    return out_uint8


os.makedirs(SAVE_DIR, exist_ok=True)
summary = {"samples": [], "config": {
    "ckpt": CKPT_PATH, "n_chunks": N_CHUNKS, "chunk_len": CHUNK_LEN,
    "sample_size": sample_size, "fps": FPS, "num_steps": NUM_STEPS,
    "guidance_scale": GUIDANCE, "seed": SEED, "start_frame": START_FRAME,
}}

for si, seg in enumerate(SAMPLES):
    seg = seg.strip()
    # SAMPLES entries may be a bare segment name (joined with DATASET_BASE) or
    # an absolute segment-directory path (lets one run mix processed/processed_v2).
    seg_dir = seg if os.path.isabs(seg) else os.path.join(DATASET_BASE, seg)
    seg_name = os.path.basename(seg_dir.rstrip("/"))
    rgb_path = os.path.join(seg_dir, "rgb.mp4")
    pose_path = os.path.join(seg_dir, "pose.mp4")
    if not (os.path.isfile(rgb_path) and os.path.isfile(pose_path)):
        print(f"[skip] missing files for {seg}")
        continue

    # exact training prompt for THIS segment when a manifest was provided
    seg_prompt = PROMPT_BY_SEG.get(seg_name, PROMPT_TMPL)
    print(f"  prompt: {seg_prompt}")

    # Per-sample start frame + chunk count (allows visual diversity + long stress runs)
    start_frame = START_FRAMES[si] if si < len(START_FRAMES) else START_FRAME
    n_chunks = N_CHUNKS_LIST[si] if si < len(N_CHUNKS_LIST) else N_CHUNKS_LIST[-1]
    print(f"\n[{seg_name}] start={start_frame}, {n_chunks}×{CHUNK_LEN} frames")
    # suffix start+nchunks so the same clip can appear twice (e.g. 8-chunk + stress)
    out_dir = os.path.join(SAVE_DIR, f"{seg_name}_s{start_frame}_n{n_chunks}")
    os.makedirs(out_dir, exist_ok=True)

    t0 = time.time()
    # GT frames for comparison
    n_total = n_chunks * CHUNK_LEN
    gt_full   = video_read_range(rgb_path,  start_frame, n_total)   # [n_total,H,W,3]
    pose_full = video_read_range(pose_path, start_frame, n_total)
    save_video(gt_full,   os.path.join(out_dir, "gt.mp4"),   fps=FPS)
    save_video(pose_full, os.path.join(out_dir, "pose.mp4"), fps=FPS)
    print(f"  loaded GT/pose ({n_total} frames each) in {time.time()-t0:.1f}s")

    rollout_chunks = []
    # get_image_to_video_latent expects either a file path OR a list of PIL images
    ref_for_next = [Image.fromarray(gt_full[0])]   # chunk 0 uses GT first frame
    for ci in range(n_chunks):
        s = ci * CHUNK_LEN
        pose_chunk = pose_full[s:s+CHUNK_LEN]
        tc = time.time()
        gen_uint8 = run_chunk(seg_prompt, pose_chunk, ref_for_next, seed=SEED + ci)
        print(f"  chunk {ci}: gen {gen_uint8.shape} in {time.time()-tc:.1f}s")
        save_video(gen_uint8, os.path.join(out_dir, f"chunk_{ci:02d}.mp4"), fps=FPS)
        rollout_chunks.append(gen_uint8)
        ref_for_next = [Image.fromarray(gen_uint8[-1])]   # last frame → next ref
        del gen_uint8
        torch.cuda.empty_cache(); gc.collect()

    rollout_full = np.concatenate(rollout_chunks, axis=0)   # [405,H,W,3]
    save_video(rollout_full, os.path.join(out_dir, "rollout.mp4"), fps=FPS)

    # Side-by-side comparison
    side_by_side(gt_full, rollout_full, os.path.join(out_dir, "comparison.mp4"), fps=FPS)

    print(f"  {seg} done in {time.time()-t0:.1f}s → {out_dir}")
    summary["samples"].append({"seg": seg_name, "prompt": seg_prompt, "start": start_frame, "n_chunks": n_chunks, "wall_s": time.time() - t0})

with open(os.path.join(SAVE_DIR, "rollout_summary.json"), "w") as f:
    json.dump(summary, f, indent=2)
print(f"\nALL DONE → {SAVE_DIR}")
