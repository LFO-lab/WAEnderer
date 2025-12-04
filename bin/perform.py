#!/usr/bin/env python3
import os, argparse, threading
import numpy as np
from stable_audio_wanderer.config import SR
from stable_audio_wanderer.io.corpus_io import find_latest, load_corpus, load_latents_bundle
from stable_audio_wanderer.vae.sae import load_vae, encode_full, load_wav
from stable_audio_wanderer.runtime.player import Player
from stable_audio_wanderer.runtime.osc_server import run_server

def make_bundle_loader(bundle_path, ae):
    lat_bundle = None
    def loader(file_id: int, wav_path: str):
        nonlocal lat_bundle
        if lat_bundle is None and bundle_path is not None:
            lat_bundle = load_latents_bundle(bundle_path)
        if lat_bundle is not None:
            key = f"z_{file_id}"
            if key in lat_bundle.files:
                return lat_bundle[key]
        # fallback: encode from WAV
        return encode_full(ae, load_wav(wav_path))
    return loader

def main():
    ap = argparse.ArgumentParser(description="Player temps réel (OSC → VAE decode).")
    ap.add_argument("--corpus_dir", required=True, help="Dossier contenant le corpus & latents")
    ap.add_argument("--pretrained", default="stabilityai/stable-audio-open-1.0")
    ap.add_argument("--osc_ip", default="127.0.0.1")
    ap.add_argument("--osc_port", type=int, default=9000)
    ap.add_argument("--win_sec", type=float, default=0.2,
                    help="Durée fenêtre décodée (grain).")
    ap.add_argument("--hop_sec", type=float, default=0.05,
                    help="Pas de sortie (latence/cadence).")
    ap.add_argument("--beta_target", type=float, default=0.2,
                    help="Gravité temporelle vers cible kNN (0..1).")
    ap.add_argument("--jump_thresh", type=int, default=64,
                    help="Seuil (frames latentes) de saut immédiat.")
    ap.add_argument("--micro_jitter", type=int, default=0,
                    help="±1 frame de jitter pour réduire artefacts de bouclage (0/1).")
    ap.add_argument("--kernel_blend", action="store_true",
                    help="Active le blending local (noyau gaussien) dans l'espace latent VAE.")
    ap.add_argument("--kernel_k", type=int, default=4,
                    help="Nombre de voisins en géométrie de projection pour le blending (k).")
    ap.add_argument("--kernel_sigma", type=float, default=-1.0,
                    help="Sigma du noyau gaussien dans l'espace de projection (<=0 pour sigma auto local).")
    ap.add_argument("--kernel_sigma_scale", type=float, default=1.0,
                    help="Facteur multiplicatif appliqué au sigma auto (si utilisé).")
    ap.add_argument("--kernel_target_norm", type=float, default=0.0,
                    help="Norme L2 cible dans l'espace latent normalisé (<=0 pour désactiver la renormalisation).")
    args = ap.parse_args()

    # Resolve files from folder
    corpus_npz = find_latest(args.corpus_dir, "*_corpus_*.npz")
    print(f"[info] Using corpus: {corpus_npz}")
    data = load_corpus(corpus_npz)

    ZZ    = data["ZZ"].astype(np.float32)
    meta  = data["meta"]
    paths = list(map(str, data["paths"]))
    Z_mean= data["Z_mean"].astype(np.float32)
    Z_std = data["Z_std"].astype(np.float32)

    # Bundle path (fallback to latest latents file in folder)
    if "latent_bundle_path" in data.files and os.path.isfile(str(data["latent_bundle_path"])):
        bundle_path = str(data["latent_bundle_path"])
    else:
        bundle_path = find_latest(os.path.dirname(corpus_npz), "*_latents_*.npz")

    print(f"[info] Using latents bundle: {bundle_path}")

    ae = load_vae(args.pretrained)
    loader = make_bundle_loader(bundle_path, ae)

    player = Player(
        ae, ZZ, meta, paths, Z_mean, Z_std,
        latent_bundle_loader=loader,
        beta_target=args.beta_target,
        jump_thresh=args.jump_thresh,
        micro_jitter=args.micro_jitter,
        win_sec=args.win_sec,
        hop_sec=args.hop_sec,
        kernel_blend=args.kernel_blend,
        kernel_k=args.kernel_k,
        kernel_sigma=args.kernel_sigma,
        kernel_sigma_scale=args.kernel_sigma_scale,
        kernel_target_norm=args.kernel_target_norm
    )

    t = threading.Thread(target=player.run, daemon=True)
    t.start()
    print(threading.enumerate())
    try:
        run_server(player, ip=args.osc_ip, port=args.osc_port)
    except KeyboardInterrupt:
        pass
    finally:
        player.stop()
        t.join()

if __name__ == "__main__":
    main()
