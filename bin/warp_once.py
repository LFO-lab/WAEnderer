#!/usr/bin/env python3
"""
Encode a single WAV with the Stable Audio VAE, apply a uniform time-warp to the latent
trajectory, then decode back to audio.

Usage:
  python bin/warp_once.py --input path/to/file.wav --output path/to/out.wav --warp 0.8
"""
import argparse
import numpy as np
import soundfile as sf
from stable_audio_wanderer.vae.sae import load_vae, load_wav, encode_full, decode_window
from stable_audio_wanderer.config import SR


def warp_latents(z: np.ndarray, factor: float) -> np.ndarray:
    """
    Reparameterize the latent timeline by a constant factor (no new points invented).
      factor > 1.0: faster playback (shorter output)
      factor < 1.0: slower playback (longer output)
      factor = 1.0: no change
    """
    factor = float(factor)
    if factor <= 0.0:
        raise ValueError("--warp must be > 0")
    if z.size == 0 or factor == 1.0:
        return z.copy()

    n, d = z.shape
    # New length inversely scales with speed.
    new_len = max(1, int(np.ceil(n / factor)))
    src_idx = np.arange(new_len, dtype=np.float32) * factor
    src_idx = np.clip(src_idx, 0.0, float(max(n - 1, 0)))

    t = np.arange(n, dtype=np.float32)
    warped = np.zeros((new_len, d), dtype=np.float32)
    for k in range(d):
        warped[:, k] = np.interp(src_idx, t, z[:, k])
    return warped


def main():
    ap = argparse.ArgumentParser(description="Encode → warp latents → decode once.")
    ap.add_argument("--input", required=True, help="Input WAV path.")
    ap.add_argument("--output", required=True, help="Destination WAV path.")
    ap.add_argument("--warp", type=float, default=1.0, help="Time-warp factor (>1 faster, <1 slower).")
    ap.add_argument("--pretrained", default="stabilityai/stable-audio-open-1.0", help="VAE repo or path.")
    args = ap.parse_args()

    print(f"[info] Loading audio: {args.input}")
    wav = load_wav(args.input, target_sr=SR)
    print(f"[info] Loaded audio shape: {wav.shape}, sr={SR}")

    print("[info] Loading VAE…")
    ae = load_vae(args.pretrained)

    print("[info] Encoding to latents…")
    z = encode_full(ae, wav)
    print(f"[info] Latents shape: {z.shape}")

    print(f"[info] Applying warp factor {args.warp}…")
    z_warp = warp_latents(z, args.warp)
    print(f"[info] Warped latents shape: {z_warp.shape}")

    print("[info] Decoding warped latents…")
    audio = decode_window(ae, z_warp)
    print(f"[info] Decoded audio shape: {audio.shape}")

    print(f"[info] Writing to {args.output}")
    sf.write(args.output, audio, SR)
    print("[done] Warp/decode complete.")


if __name__ == "__main__":
    main()
