import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import RobustScaler

def fit_transform(F: np.ndarray, pca_dim: int):
    scaler = RobustScaler(with_centering=True, with_scaling=True, quantile_range=(25.0, 75.0))
    F_scaled = scaler.fit_transform(F).astype(np.float32)

    pca = PCA(n_components=int(pca_dim), whiten=True, random_state=0).fit(F_scaled)
    ZZ = pca.transform(F_scaled).astype(np.float32)

    # min–max par dimension → [0,1]
    min_vals = ZZ.min(axis=0, keepdims=True)
    max_vals = ZZ.max(axis=0, keepdims=True)
    ZZ01 = (ZZ - min_vals) / (max_vals - min_vals + 1e-8)

    meta = {
        "scaler_center": getattr(scaler, "center_", None),
        "scaler_scale": getattr(scaler, "scale_", None),
        "scaler_quantile_range": scaler.quantile_range,
        "pca_components_": pca.components_.astype(np.float32),
        "pca_mean_": pca.mean_.astype(np.float32),
        "pca_whiten": bool(pca.whiten),
        "pca_min": min_vals.squeeze().astype(np.float32),
        "pca_max": max_vals.squeeze().astype(np.float32),
    }
    return ZZ01.astype(np.float32), meta
