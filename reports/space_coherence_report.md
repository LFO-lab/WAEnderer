# Space Coherence Report

Generated: `2026-03-28T17:23:24-04:00`

## Motivation
This evaluation tests the claim that raw VAE latent proximity is a poor proxy for perceptual timbral similarity, and that the descriptor-derived navigation space used by Stable Audio Wanderer yields more coherent local neighborhoods.

## Method
We sampled `256` anchor frames from the corpus and retrieved `k=8` nearest neighbors in two spaces: raw latent space (`Z_concat`) and the descriptor-driven navigation space (`manual_embed_points (umap)`).
Timbral similarity was evaluated using source-audio descriptors already extracted during preprocessing at the latent frame rate. For the selected features, this uses stored MFCC, centroid, loudness, flatness, and rolloff descriptors rather than distances from either search space.
The composite timbral distance is the mean of per-feature distances after robust scaling by the global median absolute deviation for each feature.
Selected features: mfcc, centroid, loudness, flatness, rolloff.

## Environment
- Corpus directory: `/Users/dthibault/Documents/GitHub/Stable-Audio-Wanderer/corpus/Rehab_20260328_150359`
- Corpus file: `/Users/dthibault/Documents/GitHub/Stable-Audio-Wanderer/corpus/Rehab_20260328_150359/corpus.npz`
- Frames: 4530
- Sample rate: 44100 Hz
- Latent rate: 21.5000 Hz
- Navigation reducer: `umap`
- Navigation source: `manual_embed_points (umap)`
- Host: `MacBook-Pro-de-Dominic-2.local`
- Platform: `macOS-15.6.1-arm64-arm-64bit`
- Device: `cpu` / `arm`
- CSV: `/Users/dthibault/Documents/GitHub/Stable-Audio-Wanderer/eval_out/space_coherence_validation/space_coherence_pairs.csv`
- JSON summary: `/Users/dthibault/Documents/GitHub/Stable-Audio-Wanderer/eval_out/space_coherence_validation/space_coherence_summary.json`
- Correlation figure: `/Users/dthibault/Documents/GitHub/Stable-Audio-Wanderer/eval_out/space_coherence_validation/space_distance_vs_timbral_distance.png`
- Boxplot figure: `/Users/dthibault/Documents/GitHub/Stable-Audio-Wanderer/eval_out/space_coherence_validation/space_neighbor_timbral_boxplots.png`

## Summary Table

| Space | Composite Mean | Composite Median | MFCC Distance Mean | Centroid Distance Mean | Loudness Distance Mean | Flatness Distance Mean | Rolloff Distance Mean | Composite rho |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Raw Latent Space | 2.9749 | 2.4213 | 46.4049 | 0.6976 | 0.3531 | 0.0018 | 0.9301 | 0.0829 |
| Descriptor Navigation Space | 0.9613 | 0.8991 | 27.1697 | 0.1835 | 0.1935 | 0.0002 | 0.2796 | 0.2916 |

## Significance
Paired anchor-level comparisons use a one-sided Wilcoxon signed-rank test with the alternative hypothesis that navigation-space neighbors have smaller timbral distances than latent-space neighbors.

| Metric | Mean Delta (latent - navigation) | Wilcoxon p |
| --- | ---: | ---: |
| MFCC Distance | 19.2351 | 5.55e-44 |
| Centroid Distance | 0.5141 | 5.817e-44 |
| Loudness Distance | 0.1596 | 1.119e-31 |
| Flatness Distance | 0.0016 | 1.353e-43 |
| Rolloff Distance | 0.6505 | 1.575e-43 |
| Composite Timbral Distance | 2.0136 | 4.82e-44 |

## Figure References
- Figure 1: `/Users/dthibault/Documents/GitHub/Stable-Audio-Wanderer/eval_out/space_coherence_validation/space_distance_vs_timbral_distance.png` shows space distance vs composite timbral distance.
- Figure 2: `/Users/dthibault/Documents/GitHub/Stable-Audio-Wanderer/eval_out/space_coherence_validation/space_neighbor_timbral_boxplots.png` compares neighbor timbral distances for each space.

## Interpretation
In this run, descriptor/navigation-space neighbors were more timbrally coherent than raw-latent neighbors on the composite metric, with lower mean and median timbral distance and a positive anchor-level mean delta (latent minus navigation).
The more important diagnostic is whether local distance in a space tracks audio-domain timbral differences. If raw latent space shows weaker correlation or higher neighbor feature distances, that supports the claim that direct latent navigation is not a reliable perceptual strategy.
Because the navigation space is built from audio descriptors and dimensionality reduction, it is explicitly optimized for perceptual organization, whereas the raw latent space is optimized for reconstruction.

## DAFx-Ready Paragraph
We evaluated local neighborhood coherence in raw VAE latent space and in the descriptor-derived navigation space used by Stable Audio Wanderer. Using latent-rate source-audio descriptors as a timbral reference, navigation-space neighbors showed lower robustly scaled composite timbral distance than raw-latent neighbors (mean 0.9613 vs 2.9749), supporting the claim that raw latent proximity is not a reliable perceptual navigation strategy. This motivates the paper's separation between perceptual control space and latent reconstruction space.
