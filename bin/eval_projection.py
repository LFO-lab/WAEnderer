#!/usr/bin/env python3
import os, argparse, numpy as np, matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr
import soundfile as sf
import librosa

from stable_audio_wanderer.config import SR, LATENT_HZ
from stable_audio_wanderer.io.corpus_io import find_latest, load_corpus
from stable_audio_wanderer.features.mfcc import segment_mfcc_no_c0
from sklearn.manifold import trustworthiness

def segment_audio(paths, meta_row):
    fid, t_lat, win_lat = int(meta_row[0]), int(meta_row[1]), int(meta_row[2])
    path = paths[fid]
    x, sr = sf.read(path, always_2d=True)
    if sr != SR:
        x = librosa.resample(x.T, orig_sr=sr, target_sr=SR, axis=1).T
    # segment samples from latent time
    seg_sec = win_lat / LATENT_HZ
    samp_start = int(round((t_lat / LATENT_HZ) * SR))
    seg_len = int(round(seg_sec * SR))
    T = x.shape[0]
    s = np.clip(samp_start, 0, max(0, T-1))
    e = np.clip(s + seg_len, 0, T)
    seg = x[s:e]
    if seg.shape[0] < seg_len:
        pad = np.zeros((seg_len - seg.shape[0], seg.shape[1]), dtype=np.float32)
        seg = np.concatenate([seg.astype(np.float32), pad], axis=0)
    return seg.astype(np.float32)

def spectral_features(seg_np, sr=SR):
    mono = seg_np.mean(axis=1)
    S = np.abs(librosa.stft(mono, n_fft=2048, hop_length=512, center=False)) + 1e-9
    freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
    # centroid (Hz)
    sc = librosa.feature.spectral_centroid(S=S, sr=sr).mean()
    # brightness: energy ratio above cutoff (e.g. 3000 Hz)
    cutoff = 3000.0
    idx = freqs >= cutoff
    bright = (S[idx, :].sum() / S.sum()).item()
    # rms (time-domain)
    rms = float(np.sqrt((mono**2).mean()))
    return sc, bright, rms

def continuity(X_high, X_low, n_neighbors=10):
    # Implementation per Venna & Kaski: continuity = 1 - 2/(n*k*(2n-3k-1)) * sum over points of sum over neighbors in high-d not in low-d of (r_i(j) - k)
    # We compute ranks in both spaces.
    from sklearn.neighbors import NearestNeighbors
    n = X_high.shape[0]
    k = n_neighbors
    # neighbors in high-d
    nh = NearestNeighbors(n_neighbors=n-1, metric='euclidean').fit(X_high)
    dist_h, ind_h = nh.kneighbors(X_high)  # includes all points; first neighbor is the closest (not self)
    # neighbors in low-d
    nl = NearestNeighbors(n_neighbors=n-1, metric='euclidean').fit(X_low)
    dist_l, ind_l = nl.kneighbors(X_low)
    # Build rank matrices (index -> rank position)
    ranks_l = np.empty((n, n), dtype=np.int32); ranks_l.fill(n+1)
    for i in range(n):
        ranks_l[i, ind_l[i]] = np.arange(1, n, dtype=np.int32)  # ranks 1..n-1
    c_sum = 0.0
    for i in range(n):
        neigh_hi = ind_h[i, :k]  # top-k in high-d
        for j_rank, j in enumerate(neigh_hi, start=1):
            r_ij = ranks_l[i, j]
            if r_ij > k:
                c_sum += (r_ij - k)
    denom = n * k * (2.0*n - 3.0*k - 1.0)
    return 1.0 - (2.0 * c_sum) / denom

def main():
    ap = argparse.ArgumentParser(description="Eval projection: trustworthiness, continuity, distance correlation, visuals.")
    ap.add_argument("--corpus_dir", required=True)
    ap.add_argument("--k", type=int, default=10, help="neighbors for trustworthiness/continuity")
    ap.add_argument("--max_points", type=int, default=2000, help="subsample for heavy metrics/plots")
    ap.add_argument("--outdir", default="./eval_out", help="where to save plots")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    corpus_npz = find_latest(args.corpus_dir, "*_corpus_*.npz")
    print(f"[info] Using corpus: {corpus_npz}")
    data = load_corpus(corpus_npz)

    ZZ = data["ZZ"].astype(np.float32)       # [N, D]
    meta = data["meta"]                      # [N, 3]
    paths = list(map(str, data["paths"]))

    N = ZZ.shape[0]
    idx = np.arange(N)
    if N > args.max_points:
        np.random.seed(0)
        idx = np.random.choice(N, size=args.max_points, replace=False)
        idx.sort()
        ZZs = ZZ[idx]
        metas = meta[idx]
    else:
        ZZs = ZZ
        metas = meta

    print(f"[info] Evaluating on {ZZs.shape[0]} points (D={ZZs.shape[1]})")

    # Recompute high-d MFCC features (same windowing as preprocess)
    feats = []
    sc_list, br_list, rms_list = [], [], []
    for m in tqdm(metas, desc="Features"):
        seg = segment_audio(paths, m)  # [L, 2]
        feats.append(segment_mfcc_no_c0(seg, 0, seg.shape[0], add_rms=False))  # start at 0 within seg
        sc, br, rms = spectral_features(seg)
        sc_list.append(sc); br_list.append(br); rms_list.append(rms)
    F = np.vstack(feats).astype(np.float32)

    # Metrics
    print("[metric] Trustworthiness…")
    tw = trustworthiness(F, ZZs, n_neighbors=args.k, metric="euclidean")
    print(f"  trustworthiness (k={args.k}): {tw:.4f}")

    print("[metric] Continuity…")
    cont = continuity(F, ZZs, n_neighbors=args.k)
    print(f"  continuity (k={args.k}): {cont:.4f}")

    print("[metric] Distance correlation (Spearman) …")
    # Subsample again if needed for O(N^2)
    M = min(800, ZZs.shape[0])
    sel = np.random.choice(ZZs.shape[0], size=M, replace=False)
    D_high = squareform(pdist(F[sel], metric="euclidean"))
    D_low  = squareform(pdist(ZZs[sel], metric="euclidean"))
    rho, p = spearmanr(D_high.ravel(), D_low.ravel())
    print(f"  spearman rho (pairwise distances): {rho:.4f} (p={p:.2e})")

    # Visuals (only if at least 2D)
    if ZZs.shape[1] >= 2:
        x, y = ZZs[:,0], ZZs[:,1]
        def plot_scatter(c, title, fname):
            plt.figure(figsize=(6,5))
            plt.scatter(x, y, c=c, s=8, alpha=0.9)
            plt.title(title)
            plt.xlabel("dim 0"); plt.ylabel("dim 1")
            plt.colorbar()
            plt.tight_layout()
            plt.savefig(os.path.join(args.outdir, fname), dpi=160)
            plt.close()

        # Normalize features for color
        sc_arr = np.asarray(sc_list)
        br_arr = np.asarray(br_list)
        rms_arr= np.asarray(rms_list)

        plot_scatter(sc_arr, "Spectral Centroid (Hz)", "scatter_centroid.png")
        plot_scatter(br_arr, "Brightness (energy > 3kHz)", "scatter_brightness.png")
        plot_scatter(rms_arr, "RMS", "scatter_rms.png")
        print(f"[plots] Saved to {args.outdir}/scatter_*.png")

    # Summary
    with open(os.path.join(args.outdir, "metrics.txt"), "w") as f:
        f.write(f"trustworthiness(k={args.k}): {tw:.6f}\n")
        f.write(f"continuity(k={args.k}): {cont:.6f}\n")
        f.write(f"distance_spearman_rho(M={M}): {rho:.6f}  p={p:.3e}\n")
    print(f"[metrics] Saved {args.outdir}/metrics.txt")

if __name__ == "__main__":
    main()
