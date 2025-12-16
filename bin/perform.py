#!/usr/bin/env python3
import os, argparse, threading, torch
import numpy as np
from stable_audio_wanderer.config import SR, DEVICE
from stable_audio_wanderer.io.corpus_io import find_latest, load_corpus, load_latents_bundle
from stable_audio_wanderer.vae.sae import load_vae, encode_full, load_wav
from stable_audio_wanderer.runtime.player import Player
from stable_audio_wanderer.runtime.osc_server import run_server
from stable_audio_wanderer.models.latent_ar import load_ar_model_with_projector

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
    ap.add_argument("--ar_drive", action="store_true",
                    help="Génère l'audio via le modèle AR latent entraîné (si présent dans le corpus).")
    ap.add_argument("--ar_noise", type=float, default=0.0,
                    help="Ecart-type du bruit gaussien ajouté aux prédictions AR (>=0).")
    # Inference stabilization options
    ap.add_argument("--ar_use_projection", action=argparse.BooleanOptionalAction, default=True,
                    help="Utilise le projecteur de manifold pour corriger la dérive (default: on si disponible).")
    ap.add_argument("--ar_clamp_std", type=float, default=3.0,
                    help="Seuil de clampage adaptatif en écarts-types (0 pour désactiver).")
    ap.add_argument("--ar_target_norm", type=float, default=0.0,
                    help="Norme cible pour projection sphérique (0 pour désactiver).")
    ap.add_argument("--ar_reanchor_interval", type=int, default=0,
                    help="Intervalle (frames) pour re-ancrage au corpus (0 pour désactiver).")
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
    ar_model = None
    ar_projector = None
    ar_context = None
    ar_path = None
    ar_noise_std = max(0.0, float(args.ar_noise))
    ar_clamp_std = float(args.ar_clamp_std) if args.ar_clamp_std > 0 else None
    ar_target_norm = float(args.ar_target_norm) if args.ar_target_norm > 0 else None
    ar_reanchor_interval = max(0, int(args.ar_reanchor_interval))

    if args.ar_drive:
        if "ar_model_path" in data.files:
            candidate = str(data["ar_model_path"])
            if candidate and os.path.isfile(candidate):
                ar_path = candidate
        if ar_path is None:
            print("[warn] --ar_drive demandé mais aucun modèle AR trouvé dans le corpus; retour au mode kNN.")
            args.ar_drive = False
        else:
            device = torch.device(DEVICE)
            # Load AR model and projector together
            ar_model, ar_projector, ar_meta = load_ar_model_with_projector(ar_path, device=device)
            ctx_from_meta = ar_meta.get("context", 0) if isinstance(ar_meta, dict) else 0
            ar_context = int(ctx_from_meta) if ctx_from_meta else None
            if ar_context is None and "ar_context" in data.files:
                try:
                    ar_context = int(data["ar_context"])
                except Exception:
                    ar_context = None
            if ar_context is None or ar_context <= 0:
                print("[warn] Impossible de récupérer le contexte AR; retour au mode kNN.")
                args.ar_drive = False
                ar_model = None
                ar_projector = None
            else:
                proj_status = "avec projecteur" if ar_projector is not None else "sans projecteur"
                print(f"[info] Modèle AR chargé ({ar_path}), contexte={ar_context}, {proj_status}")
                
                # Get target norm from training if not specified
                if ar_target_norm is None or ar_target_norm <= 0:
                    if "target_norm" in ar_meta:
                        ar_target_norm = float(ar_meta["target_norm"])
                        print(f"[info] Utilisation de la norme cible d'entraînement: {ar_target_norm:.4f}")

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
        kernel_target_norm=args.kernel_target_norm,
        ar_model=ar_model,
        ar_context=ar_context,
        ar_drive=args.ar_drive,
        ar_noise_std=ar_noise_std,
        # Inference stabilization
        ar_projector=ar_projector,
        ar_use_projection=bool(args.ar_use_projection),
        ar_clamp_std=ar_clamp_std,
        ar_reanchor_interval=ar_reanchor_interval,
        ar_target_norm=ar_target_norm,
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
