#!/usr/bin/env python3
"""
Preprocess audio files: MFCC→PCA + VAE latents + contrastive projector + grain rendering.
"""
import os, argparse, datetime, glob, numpy as np
from typing import List, Dict, Tuple
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.neighbors import NearestNeighbors

from stable_audio_wanderer.config import SR, LATENT_HZ, DEVICE
from stable_audio_wanderer.vae.sae import load_vae, load_wav, encode_full
from stable_audio_wanderer.features.mfcc import segment_mfcc_no_c0, ensure_2d_stack
from stable_audio_wanderer.dr.pca import fit_transform
from stable_audio_wanderer.io.corpus_io import save_latents_bundle, save_corpus, save_grain_manifest
from stable_audio_wanderer.io.audio_io import save_wav


class LatentDataset(Dataset):
    def __init__(self, latents: np.ndarray, return_index: bool = False):
        self.latents = torch.from_numpy(latents)
        self.return_index = return_index

    def __len__(self):
        return self.latents.shape[0]

    def __getitem__(self, idx: int):
        if self.return_index:
            return self.latents[idx], idx
        return self.latents[idx]


class LatentProjector(nn.Module):
    def __init__(self, input_dim: int = 64, proj_dim: int = 64):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, proj_dim)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.fc1(x))
        x = self.fc2(x)
        return F.normalize(x, dim=-1, eps=1e-8)


def augment_latents(x: torch.Tensor, noise_std: float, drop_prob: float, mix_alpha: float) -> torch.Tensor:
    y = x.clone()
    if noise_std > 0.0:
        y = y + noise_std * torch.randn_like(y)
    if drop_prob > 0.0:
        keep = (torch.rand_like(y) > drop_prob).to(y.dtype)
        y = y * keep
    if mix_alpha > 0.0 and y.size(0) > 1:
        perm = torch.randperm(y.size(0), device=y.device)
        y = (1.0 - mix_alpha) * y + mix_alpha * y[perm]
    return y


def build_latent_knn(latents_np: np.ndarray, k: int) -> List[List[int]]:
    n = latents_np.shape[0]
    if n <= 1:
        return [[]]
    k_eff = max(1, min(int(k), n - 1))
    nn = NearestNeighbors(n_neighbors=min(k_eff + 1, n), metric="euclidean")
    nn.fit(latents_np)
    _, indices = nn.kneighbors(latents_np, return_distance=True)
    neighbors: List[List[int]] = []
    for i, row in enumerate(indices):
        neigh = [int(j) for j in row if int(j) != i][:k_eff]
        neighbors.append(neigh)
    return neighbors


def sample_positive_indices(batch_indices: torch.Tensor, neighbor_graph: List[List[int]]) -> torch.Tensor:
    pos = []
    for idx in batch_indices.tolist():
        neigh = neighbor_graph[idx]
        if neigh:
            choice = np.random.choice(neigh)
        else:
            choice = idx
        pos.append(int(choice))
    return torch.tensor(pos, device=batch_indices.device, dtype=torch.long)


def uniformity_loss(u: torch.Tensor) -> torch.Tensor:
    if u.size(0) <= 1:
        return u.new_tensor(0.0)
    # For unit vectors: ||u_i - u_j||^2 = 2 - 2 cos(theta)
    sim = torch.matmul(u, u.transpose(0, 1))
    mask = ~torch.eye(u.size(0), dtype=torch.bool, device=u.device)
    dist2 = torch.clamp(2.0 - 2.0 * sim, min=0.0)
    pairs = dist2[mask]
    if pairs.numel() == 0:
        return u.new_tensor(0.0)
    return torch.log(torch.exp(-2.0 * pairs).mean() + 1e-12)


def covariance_regularizer(
    u: torch.Tensor, gamma: float, var_weight: float, decorr_weight: float
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if u.size(0) <= 1:
        zeros = u.new_tensor(0.0)
        return zeros, zeros, zeros
    mu = u.mean(dim=0, keepdim=True)
    zc = u - mu
    cov = torch.matmul(zc.transpose(0, 1), zc) / float(u.size(0))
    var = torch.diag(cov)
    var_term = F.relu(gamma - var).pow(2)
    var_loss = var_term.sum() / max(var.numel(), 1)

    # Decorrelate off-diagonal terms.
    off_diag = cov - torch.diag(var)
    off_denom = max(off_diag.numel() - var.numel(), 1)
    decorr_loss = off_diag.pow(2).sum() / off_denom

    cov_loss = var_weight * var_loss + decorr_weight * decorr_loss
    return cov_loss, var_loss, decorr_loss


def info_nce_loss(u: torch.Tensor, v: torch.Tensor, tau: float) -> torch.Tensor:
    logits = torch.matmul(u, v.transpose(0, 1)) / tau
    labels = torch.arange(u.size(0), device=u.device)
    loss_uv = F.cross_entropy(logits, labels)
    loss_vu = F.cross_entropy(logits.transpose(0, 1), labels)
    return 0.5 * (loss_uv + loss_vu)


def train_contrastive_projector(
    latents_np: np.ndarray,
    proj_dim: int,
    tau: float,
    epochs: int,
    batch_size: int,
    lr: float,
    device: torch.device,
    noise_std: float,
    drop_prob: float,
    mix_alpha: float,
    uniformize: bool,
    knn_k: int,
    lambda_unif: float,
    lambda_cov: float,
    cov_var_weight: float,
    cov_decorr_weight: float,
    cov_gamma: float,
    mean_weight: float,
):
    dataset = LatentDataset(latents_np, return_index=uniformize)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    model = LatentProjector(input_dim=latents_np.shape[1], proj_dim=proj_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.999))
    loss_history: List[Dict[str, float]] = []

    neighbor_graph = build_latent_knn(latents_np, k=knn_k) if uniformize else None
    latents_all = torch.from_numpy(latents_np).to(device) if uniformize else None

    for epoch in range(epochs):
        model.train()
        totals = {
            "total": 0.0,
            "sim": 0.0,
            "unif": 0.0,
            "cov": 0.0,
            "cov_var": 0.0,
            "cov_decorr": 0.0,
            "mean": 0.0,
        }
        total_samples = 0

        for batch in loader:
            if uniformize:
                batch, batch_idx = batch
                batch = batch.to(device)
                batch_idx = batch_idx.to(device)
                pos_idx = sample_positive_indices(batch_idx, neighbor_graph)
                anchor = augment_latents(batch, noise_std=noise_std, drop_prob=drop_prob, mix_alpha=0.0)
                pos_latents = latents_all[pos_idx]
                positive = augment_latents(pos_latents, noise_std=noise_std, drop_prob=drop_prob, mix_alpha=0.0)
            else:
                batch = batch.to(device)
                anchor = batch
                positive = augment_latents(batch, noise_std=noise_std, drop_prob=drop_prob, mix_alpha=mix_alpha)

            optimizer.zero_grad(set_to_none=True)
            u = model(anchor)
            v = model(positive)

            sim_loss = info_nce_loss(u, v, tau=tau)
            total_loss = sim_loss

            unif_loss = u.new_tensor(0.0)
            cov_loss = u.new_tensor(0.0)
            cov_var_loss = u.new_tensor(0.0)
            cov_decorr_loss = u.new_tensor(0.0)
            mean_loss = u.new_tensor(0.0)

            if uniformize:
                combined = torch.cat([u, v], dim=0)
                unif_loss = uniformity_loss(combined)
                cov_loss, cov_var_loss, cov_decorr_loss = covariance_regularizer(
                    combined, gamma=cov_gamma, var_weight=cov_var_weight, decorr_weight=cov_decorr_weight
                )
                total_loss = total_loss + lambda_unif * unif_loss + lambda_cov * cov_loss
                if mean_weight > 0.0:
                    mean_loss = combined.mean(dim=0).pow(2).sum()
                    total_loss = total_loss + mean_weight * mean_loss

            total_loss.backward()
            optimizer.step()

            bs = anchor.size(0)
            total_samples += bs
            totals["total"] += float(total_loss.item()) * bs
            totals["sim"] += float(sim_loss.item()) * bs
            totals["unif"] += float(unif_loss.item()) * bs
            totals["cov"] += float(cov_loss.item()) * bs
            totals["cov_var"] += float(cov_var_loss.item()) * bs
            totals["cov_decorr"] += float(cov_decorr_loss.item()) * bs
            totals["mean"] += float(mean_loss.item()) * bs

        denom = max(total_samples, 1)
        epoch_stats = {k: v / denom for k, v in totals.items()}
        loss_history.append(epoch_stats)
        print(
            f"[g] epoch {epoch + 1}/{epochs} - "
            f"total: {epoch_stats['total']:.4f} | sim: {epoch_stats['sim']:.4f} | "
            f"unif: {epoch_stats['unif']:.4f} | cov: {epoch_stats['cov']:.4f}"
        )
    return model, loss_history


def project_latents(model: LatentProjector, latents_np: np.ndarray, device: torch.device, batch_size: int = 4096):
    model.eval()
    outputs: List[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, latents_np.shape[0], batch_size):
            chunk = torch.from_numpy(latents_np[start:start + batch_size]).to(device)
            proj = model(chunk).cpu().numpy()
            outputs.append(proj)
    return np.concatenate(outputs, axis=0)


def compute_segment_latents(latents_dict: dict, meta: np.ndarray, fallback_len: int) -> np.ndarray:
    seg_latents = []
    for fid, start, seg_len in meta:
        key = f"z_{int(fid)}"
        z_full = latents_dict[key]
        start = int(start)
        seg_len = int(seg_len) if int(seg_len) > 0 else int(fallback_len)
        end = min(z_full.shape[0], start + seg_len)
        z_slice = z_full[start:end]
        if z_slice.shape[0] == 0:
            z_slice = np.repeat(z_full[:1], seg_len, axis=0)
        if z_slice.shape[0] < seg_len:
            pad = np.repeat(z_slice[-1:], seg_len - z_slice.shape[0], axis=0)
            z_slice = np.concatenate([z_slice, pad], axis=0)
        seg_latents.append(z_slice.mean(axis=0))
    return np.stack(seg_latents, axis=0).astype(np.float32)


def render_grains(
    wav_dict: Dict[int, np.ndarray],
    meta: np.ndarray,
    grain_sec: float,
    out_dir: str,
    sr: int = SR,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """
    Render pre-baked grains with Hann envelope for each segment.
    
    Args:
        wav_dict: Dict mapping file_id -> loaded waveform [T, 2]
        meta: Segment metadata [N_seg, 3] with (file_id, t_lat, win_lat)
        grain_sec: Grain duration in seconds (longer grains with envelope)
        out_dir: Output directory for grain WAV files
        sr: Sample rate
        
    Returns:
        offsets: Sample offset into concatenated grain buffer [N_grains]
        lengths: Length in samples for each grain [N_grains]
        file_ids: Source file ID for each grain [N_grains]
        segment_ids: Maps each grain to corpus segment index [N_grains]
        grain_paths: Path to each file's concatenated grain WAV [N_files]
    """
    grain_len_samp = int(round(grain_sec * sr))
    
    # Create Hann window for envelope
    envelope = np.hanning(grain_len_samp).astype(np.float32)
    envelope = envelope[:, None]  # [grain_len, 1] for broadcasting to stereo
    
    # Group segments by file
    file_segments: Dict[int, List[Tuple[int, int]]] = {}  # file_id -> [(seg_idx, t_lat), ...]
    for seg_idx, (fid, t_lat, _) in enumerate(meta):
        fid = int(fid)
        if fid not in file_segments:
            file_segments[fid] = []
        file_segments[fid].append((seg_idx, int(t_lat)))
    
    offsets_list: List[int] = []
    lengths_list: List[int] = []
    file_ids_list: List[int] = []
    segment_ids_list: List[int] = []
    grain_paths: List[str] = []
    
    grains_dir = os.path.join(out_dir, "grains")
    os.makedirs(grains_dir, exist_ok=True)
    
    # For each source file, render grains and concatenate
    for fid in sorted(file_segments.keys()):
        wav = wav_dict[fid]
        segments = file_segments[fid]
        
        grain_buffer_list: List[np.ndarray] = []
        current_offset = 0
        
        for seg_idx, t_lat in segments:
            # Convert latent time to sample position
            sec_start = t_lat / LATENT_HZ
            samp_start = int(round(sec_start * sr))
            
            # Extract grain with envelope
            samp_end = min(samp_start + grain_len_samp, wav.shape[0])
            grain = wav[samp_start:samp_end].copy()
            
            # Pad if grain is shorter than expected
            if grain.shape[0] < grain_len_samp:
                pad_len = grain_len_samp - grain.shape[0]
                pad = np.zeros((pad_len, grain.shape[1]), dtype=np.float32)
                grain = np.concatenate([grain, pad], axis=0)
            
            # Apply Hann envelope
            grain = grain * envelope
            
            # Record metadata
            offsets_list.append(current_offset)
            lengths_list.append(grain.shape[0])
            file_ids_list.append(fid)
            segment_ids_list.append(seg_idx)
            
            grain_buffer_list.append(grain)
            current_offset += grain.shape[0]
        
        # Concatenate all grains for this file
        if grain_buffer_list:
            file_grain_buffer = np.concatenate(grain_buffer_list, axis=0)
            grain_path = os.path.join(grains_dir, f"file_{fid:04d}.wav")
            save_wav(grain_path, file_grain_buffer, sr)
            grain_paths.append(grain_path)
        else:
            grain_paths.append("")
    
    return (
        np.array(offsets_list, dtype=np.int64),
        np.array(lengths_list, dtype=np.int64),
        np.array(file_ids_list, dtype=np.int32),
        np.array(segment_ids_list, dtype=np.int32),
        grain_paths,
    )


def main():
    ap = argparse.ArgumentParser(description="Préprocess: MFCC→PCA + bundle latents + contrastive projector g.")
    ap.add_argument("--audio_dir", required=True)
    ap.add_argument("--out_prefix", required=True)
    ap.add_argument("--pretrained", default="stabilityai/stable-audio-open-1.0")
    ap.add_argument("--seg_sec", type=float, default=0.2)
    ap.add_argument("--hop_sec", type=float, default=0.05)
    ap.add_argument("--pca_dim", type=int, default=2)
    ap.add_argument("--add_rms", type=int, default=0)
    ap.add_argument("--proj_dim", type=int, default=64, help="Output dimension d ≤ 64 for g(z).")
    ap.add_argument("--proj_epochs", type=int, default=10, help="Number of epochs for contrastive training.")
    ap.add_argument("--proj_batch", type=int, default=512, help="Batch size for contrastive projector training.")
    ap.add_argument("--proj_tau", type=float, default=0.1, help="InfoNCE temperature (0.05–0.2).")
    ap.add_argument("--proj_lr", type=float, default=1e-3, help="Learning rate for projector training.")
    ap.add_argument("--proj_noise", type=float, default=0.05, help="Gaussian noise std for latent augmentation.")
    ap.add_argument("--proj_drop", type=float, default=0.1, help="Drop probability for latent augmentation.")
    ap.add_argument(
        "--proj_mix",
        type=float,
        default=0.15,
        help="Mixup coefficient for latent augmentation (ignored when --uniform_proj is enabled).",
    )
    ap.add_argument(
        "--uniform_proj",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable spherical uniformization + VAE-KNN positives for the projector (default: on).",
    )
    ap.add_argument(
        "--proj_knn",
        type=int,
        default=10,
        help="Number of nearest neighbors (in VAE latent space) used as positives when uniform_proj is on.",
    )
    ap.add_argument("--proj_lambda_unif", type=float, default=1.0, help="Weight for spherical uniformity loss.")
    ap.add_argument("--proj_lambda_cov", type=float, default=0.5, help="Weight for covariance regularizer.")
    ap.add_argument("--proj_cov_var", type=float, default=0.5, help="Weight for the variance floor term inside cov reg.")
    ap.add_argument(
        "--proj_cov_decorr",
        type=float,
        default=1.0,
        help="Weight for the decorrelation term inside covariance regularizer.",
    )
    ap.add_argument(
        "--proj_cov_gamma",
        type=float,
        default=-1.0,
        help="Minimum per-dim variance target for covariance regularizer (<= 1/proj_dim). Use -1 for auto.",
    )
    ap.add_argument(
        "--proj_mean_weight",
        type=float,
        default=0.0,
        help="Optional weight for ||mean||^2 penalty on projector outputs (helps recentring if needed).",
    )
    # Grain rendering arguments
    ap.add_argument(
        "--render_grains",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Render pre-baked grain audio buffers for runtime playback (default: on).",
    )
    ap.add_argument(
        "--grain_sec",
        type=float,
        default=1.0 / LATENT_HZ,  # ~0.0465s, matches one latent frame and playback grain duration
        help="Grain duration in seconds (default: ~0.0465s, matching one latent frame).",
    )
    args = ap.parse_args()

    if not (1 <= args.proj_dim <= 64):
        ap.error("--proj_dim must be in [1, 64].")
    if args.proj_epochs < 1:
        ap.error("--proj_epochs must be ≥ 1.")
    if args.proj_batch < 1:
        ap.error("--proj_batch must be ≥ 1.")
    tau = float(np.clip(args.proj_tau, 0.05, 0.2))
    if abs(tau - args.proj_tau) > 1e-6:
        print(f"[warn] Clamped --proj_tau to {tau:.3f}")
    args.proj_tau = tau
    if args.proj_knn < 1:
        ap.error("--proj_knn must be ≥ 1.")

    # Clamp weights to sane ranges.
    args.proj_lambda_unif = max(0.0, float(args.proj_lambda_unif))
    args.proj_lambda_cov = max(0.0, float(args.proj_lambda_cov))
    args.proj_cov_var = max(0.0, float(args.proj_cov_var))
    args.proj_cov_decorr = max(0.0, float(args.proj_cov_decorr))
    args.proj_mean_weight = max(0.0, float(args.proj_mean_weight))
    max_gamma = 1.0 / float(args.proj_dim)
    if args.proj_cov_gamma <= 0.0 or args.proj_cov_gamma > max_gamma:
        if args.proj_cov_gamma > max_gamma:
            print(f"[warn] Clamped --proj_cov_gamma to {max_gamma:.4f} to keep targets feasible on the unit sphere.")
        args.proj_cov_gamma = max_gamma
    if args.uniform_proj and args.proj_mix > 0.0:
        print("[info] --proj_mix is ignored when --uniform_proj is enabled (using VAE-space neighbors as positives).")

    prefix = os.path.basename(args.out_prefix)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(os.getcwd(), "corpus", f"{prefix}_{ts}")
    os.makedirs(out_dir, exist_ok=True)

    seg_len_samp = int(round(args.seg_sec * SR))
    assert seg_len_samp >= 2048, "seg_sec must be >= 2048/44100"

    ae = load_vae(args.pretrained)

    paths = sorted(glob.glob(os.path.join(args.audio_dir, "*.wav")))
    if not paths:
        raise FileNotFoundError("No WAV files in --audio_dir")

    win_lat = max(1, int(round(args.seg_sec * LATENT_HZ)))
    hop_lat = max(1, int(round(args.hop_sec * LATENT_HZ)))

    feat_list, meta_list = [], []
    latent_sequences: List[np.ndarray] = []
    latents_dict = {}
    wav_dict: Dict[int, np.ndarray] = {}  # Store wavs for grain rendering

    print("Encoding + feature extraction…")
    for fid, p in enumerate(tqdm(paths)):
        wav = load_wav(p)
        wav_dict[fid] = wav  # Store for grain rendering
        z_full = encode_full(ae, wav).astype(np.float32)  # [T_lat, 64]
        z_full = np.ascontiguousarray(z_full)
        latents_dict[f"z_{fid}"] = z_full
        latent_sequences.append(z_full)

        T_lat = z_full.shape[0]
        starts = np.arange(0, max(1, T_lat - win_lat + 1), hop_lat, dtype=int)
        for t_lat in starts:
            sec_start = t_lat / LATENT_HZ
            samp_start = int(round(sec_start * SR))
            feats = segment_mfcc_no_c0(wav, samp_start, seg_len_samp, add_rms=bool(args.add_rms))
            feat_list.append(feats)
            meta_list.append((fid, t_lat, win_lat))

    if not latent_sequences:
        raise RuntimeError("No latent sequences were extracted.")

    F = ensure_2d_stack(feat_list)
    meta = np.array(meta_list, dtype=np.int32)
    paths_arr = np.array(paths)

    latent_stack = np.concatenate(latent_sequences, axis=0).astype(np.float32)
    Z_mean = latent_stack.mean(axis=0).astype(np.float32)
    Z_var = latent_stack.var(axis=0).astype(np.float32)
    Z_std = np.sqrt(Z_var + 1e-6).astype(np.float32)

    latent_stack_norm = (latent_stack - Z_mean[None, :]) / Z_std[None, :]
    latent_stack_norm = np.ascontiguousarray(latent_stack_norm.astype(np.float32))
    if latent_stack_norm.shape[0] < 2:
        raise RuntimeError("Need at least two latent frames to train the contrastive projector.")

    proj_batch = min(args.proj_batch, latent_stack_norm.shape[0])
    proj_batch = max(2, proj_batch)
    device = torch.device(DEVICE)

    print(f"Training contrastive projector g… (uniform_proj={bool(args.uniform_proj)})")
    projector, g_loss_history = train_contrastive_projector(
        latent_stack_norm,
        args.proj_dim,
        args.proj_tau,
        args.proj_epochs,
        proj_batch,
        args.proj_lr,
        device,
        args.proj_noise,
        args.proj_drop,
        args.proj_mix,
        uniformize=bool(args.uniform_proj),
        knn_k=int(args.proj_knn),
        lambda_unif=float(args.proj_lambda_unif),
        lambda_cov=float(args.proj_lambda_cov),
        cov_var_weight=float(args.proj_cov_var),
        cov_decorr_weight=float(args.proj_cov_decorr),
        cov_gamma=float(args.proj_cov_gamma),
        mean_weight=float(args.proj_mean_weight),
    )
    g_losses = [float(entry["total"]) for entry in g_loss_history]

    seg_latents = compute_segment_latents(latents_dict, meta, win_lat)
    seg_latents_norm = (seg_latents - Z_mean[None, :]) / Z_std[None, :]
    seg_latents_norm = np.ascontiguousarray(seg_latents_norm.astype(np.float32))

    g_batch = max(proj_batch, 1024)
    g_embeddings = project_latents(projector, seg_latents_norm, device, batch_size=g_batch)

    g_min = g_embeddings.min(axis=0).astype(np.float32)
    g_max = g_embeddings.max(axis=0).astype(np.float32)
    g_range = np.maximum(g_max - g_min, 1e-6)
    GG = ((g_embeddings - g_min[None, :]) / g_range[None, :]).astype(np.float32)
    GG = np.clip(GG, 0.0, 1.0)

    projector_state = {k: v.detach().cpu() for k, v in projector.state_dict().items()}
    projector_path = os.path.join(out_dir, f"{prefix}_projector_{ts}.pt")
    torch.save(
        {
            "state_dict": projector_state,
            "input_dim": 64,
            "proj_dim": int(args.proj_dim),
            "tau": float(args.proj_tau),
            "augment": {
                "noise_std": float(args.proj_noise),
                "drop_prob": float(args.proj_drop),
                "mix_alpha": float(args.proj_mix),
            },
            "uniformize": bool(args.uniform_proj),
            "knn_k": int(args.proj_knn),
            "lambda_unif": float(args.proj_lambda_unif),
            "lambda_cov": float(args.proj_lambda_cov),
            "cov_var_weight": float(args.proj_cov_var),
            "cov_decorr_weight": float(args.proj_cov_decorr),
            "cov_gamma": float(args.proj_cov_gamma),
            "mean_weight": float(args.proj_mean_weight),
            "loss_history": g_losses,
            "loss_breakdown": g_loss_history,
        },
        projector_path,
    )

    ZZ, dr_meta = fit_transform(F, args.pca_dim)

    latents_path = os.path.join(out_dir, f"{prefix}_latents_{ts}.npz")
    save_latents_bundle(latents_path, latents_dict, paths_arr)

    corpus_path = os.path.join(out_dir, f"{prefix}_corpus_{ts}.npz")
    dr_arrays = {k: (v if v is None else np.asarray(v, dtype=np.float32)) for k, v in dr_meta.items()}
    save_corpus(
        corpus_path,
        ZZ=ZZ,
        GG=GG,
        pca_dim=np.array(int(args.pca_dim), dtype=np.int32),
        g_dim=np.array(int(args.proj_dim), dtype=np.int32),
        meta=meta,
        paths=paths_arr,
        Z_mean=Z_mean,
        Z_std=Z_std,
        g_min=g_min,
        g_max=g_max,
        g_loss_history=np.asarray(g_losses, dtype=np.float32),
        g_model_path=np.array(projector_path),
        g_tau=np.array(float(args.proj_tau), dtype=np.float32),
        g_train_epochs=np.array(int(args.proj_epochs), dtype=np.int32),
        g_train_batch=np.array(int(proj_batch), dtype=np.int32),
        g_train_lr=np.array(float(args.proj_lr), dtype=np.float32),
        g_aug_noise=np.array(float(args.proj_noise), dtype=np.float32),
        g_aug_drop=np.array(float(args.proj_drop), dtype=np.float32),
        g_aug_mix=np.array(float(args.proj_mix), dtype=np.float32),
        g_uniformize=np.array(bool(args.uniform_proj)),
        g_knn_k=np.array(int(args.proj_knn), dtype=np.int32),
        g_lambda_unif=np.array(float(args.proj_lambda_unif), dtype=np.float32),
        g_lambda_cov=np.array(float(args.proj_lambda_cov), dtype=np.float32),
        g_cov_var_weight=np.array(float(args.proj_cov_var), dtype=np.float32),
        g_cov_decorr_weight=np.array(float(args.proj_cov_decorr), dtype=np.float32),
        g_cov_gamma=np.array(float(args.proj_cov_gamma), dtype=np.float32),
        g_mean_weight=np.array(float(args.proj_mean_weight), dtype=np.float32),
        g_loss_breakdown=np.array(g_loss_history, dtype=object),
        latent_bundle_path=np.array(latents_path),
        **dr_arrays,
    )

    # --- Grain Rendering ---
    grain_manifest_path = None
    if args.render_grains:
        print(f"\nRendering grains (grain_sec={args.grain_sec})…")
        offsets, lengths, file_ids, segment_ids, grain_paths = render_grains(
            wav_dict=wav_dict,
            meta=meta,
            grain_sec=args.grain_sec,
            out_dir=out_dir,
            sr=SR,
        )
        
        grain_manifest_path = os.path.join(out_dir, "grains", "manifest.npz")
        save_grain_manifest(
            path=grain_manifest_path,
            offsets=offsets,
            lengths=lengths,
            file_ids=file_ids,
            segment_ids=segment_ids,
            grain_paths=grain_paths,
            grain_sec=args.grain_sec,
            grain_hop_sec=args.hop_sec,  # Use segment hop as grain hop reference
            sr=SR,
        )
        print(f"  Grains manifest: {grain_manifest_path}")
        print(f"  Grain files: {len(grain_paths)} files in {os.path.join(out_dir, 'grains')}")
    
    # Clear wav_dict to free memory
    wav_dict.clear()

    print("\nSaved:")
    print("  Latents   :", latents_path)
    print("  Projector :", projector_path)
    print("  Corpus    :", corpus_path)
    if grain_manifest_path:
        print("  Grains    :", grain_manifest_path)
    print("  Folder    :", out_dir)


if __name__ == "__main__":
    main()
