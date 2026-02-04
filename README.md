# Stable Audio Wanderer

Stable Audio Wanderer is a 64D latent navigation instrument for exploring sound corpora. Audio is encoded with the Stable Audio Open VAE, segmented, and transformed into a latent corpus with geometry (kNN + PCA). A learned policy moves through this space, and a manifold-constrained generator produces latents for real-time VAE decoding. The performance UI is a WebSocket-based p5.js visualization.

**Pipeline**
1. Preprocess audio into a geometry-enabled latent corpus.
2. Train a latent navigation policy on the corpus.
3. Perform with OSC control and the WebSocket UI (realtime decoding).

**Commands**
Preprocess (build corpus + geometry):
```bash
python bin/preprocess.py --audio_dir /path/to/wavs --out_prefix my_corpus
```

Train policy (latent-only):
```bash
python bin/train_policy.py --corpus_dir corpus/my_corpus_YYYYMMDD_HHMMSS
```

Perform (OSC + WebSocket UI):
```bash
python bin/perform.py --corpus_dir corpus/my_corpus_YYYYMMDD_HHMMSS
```
Open `web/index.html` in a browser to view the visualization.

**Requirements**
- Python 3.9+
- Core dependencies: `torch`, `torchaudio`, `numpy`, `soundfile`, `scikit-learn`, `scipy`, `faiss-cpu`
- Runtime audio: `sounddevice`
- Control + UI: `python-osc`, `websockets`
- VAE encoding/decoding: `diffusers` (Stable Audio Open VAE)

Install core requirements:
```bash
pip install -r requirements.txt
```

**Notes**
- Corpora must be generated with geometry (`geom_*` fields). Legacy corpora are not supported.
- The VAE is used for offline preprocessing and runtime decoding.
