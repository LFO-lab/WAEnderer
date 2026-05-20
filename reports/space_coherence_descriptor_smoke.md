# Space Coherence Report

Generated: `2026-03-28T17:23:46-04:00`

## Motivation
This evaluation tests the claim that raw VAE latent proximity is a poor proxy for perceptual timbral similarity, and that the descriptor-derived navigation space used by Stable Audio Wanderer yields more coherent local neighborhoods.

## Method
We sampled `8` anchor frames from the corpus and retrieved `k=2` nearest neighbors in two spaces: raw latent space (`Z_concat`) and the descriptor-driven navigation space (`manual_desc_weighted`).
Timbral similarity was evaluated using source-audio descriptors already extracted during preprocessing at the latent frame rate. For the selected features, this uses stored MFCC, centroid, loudness, flatness, and rolloff descriptors rather than distances from either search space.
The composite timbral distance is the mean of per-feature distances after robust scaling by the global median absolute deviation for each feature.
Selected features: mfcc, centroid, loudness, flatness, rolloff.

## Environment
- Corpus directory: `/Users/dthibault/Documents/GitHub/Stable-Audio-Wanderer/corpus/Rehab_20260328_150359`
- Corpus file: `/Users/dthibault/Documents/GitHub/Stable-Audio-Wanderer/corpus/Rehab_20260328_150359/corpus.npz`
- Frames: 4530
- Sample rate: 44100 Hz
- Latent rate: 21.5000 Hz
- Navigation reducer: `weighted_descriptor_space`
- Navigation source: `manual_desc_weighted`
- Host: `MacBook-Pro-de-Dominic-2.local`
- Platform: `macOS-15.6.1-arm64-arm-64bit`
- Device: `cpu` / `arm`
- CSV: `/Users/dthibault/Documents/GitHub/Stable-Audio-Wanderer/eval_out/space_coherence_descriptor_smoke/space_coherence_pairs.csv`
- JSON summary: `/Users/dthibault/Documents/GitHub/Stable-Audio-Wanderer/eval_out/space_coherence_descriptor_smoke/space_coherence_summary.json`
- Correlation figure: `/Users/dthibault/Documents/GitHub/Stable-Audio-Wanderer/eval_out/space_coherence_descriptor_smoke/space_distance_vs_timbral_distance.png`
- Boxplot figure: `/Users/dthibault/Documents/GitHub/Stable-Audio-Wanderer/eval_out/space_coherence_descriptor_smoke/space_neighbor_timbral_boxplots.png`

## Summary Table

| Space | Composite Mean | Composite Median | MFCC Distance Mean | Centroid Distance Mean | Loudness Distance Mean | Flatness Distance Mean | Rolloff Distance Mean | Composite rho |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Raw Latent Space | 3.7236 | 2.8174 | 54.1657 | 0.6184 | 0.3299 | 0.0023 | 1.1696 | 0.4206 |
| Descriptor Navigation Space | 0.7743 | 0.5947 | 24.1457 | 0.1455 | 0.1563 | 0.0001 | 0.2792 | 0.7735 |

## Significance
Paired anchor-level comparisons use a one-sided Wilcoxon signed-rank test with the alternative hypothesis that navigation-space neighbors have smaller timbral distances than latent-space neighbors.

| Metric | Mean Delta (latent - navigation) | Wilcoxon p |
| --- | ---: | ---: |
| MFCC Distance | 30.0201 | 0.003906 |
| Centroid Distance | 0.4729 | 0.03906 |
| Loudness Distance | 0.1737 | 0.01953 |
| Flatness Distance | 0.0022 | 0.007812 |
| Rolloff Distance | 0.8904 | 0.01562 |
| Composite Timbral Distance | 2.9493 | 0.003906 |

## Figure References
- Figure 1: `/Users/dthibault/Documents/GitHub/Stable-Audio-Wanderer/eval_out/space_coherence_descriptor_smoke/space_distance_vs_timbral_distance.png` shows space distance vs composite timbral distance.
- Figure 2: `/Users/dthibault/Documents/GitHub/Stable-Audio-Wanderer/eval_out/space_coherence_descriptor_smoke/space_neighbor_timbral_boxplots.png` compares neighbor timbral distances for each space.

## Interpretation
In this run, descriptor/navigation-space neighbors were more timbrally coherent than raw-latent neighbors on the composite metric, with lower mean and median timbral distance and a positive anchor-level mean delta (latent minus navigation).
The more important diagnostic is whether local distance in a space tracks audio-domain timbral differences. If raw latent space shows weaker correlation or higher neighbor feature distances, that supports the claim that direct latent navigation is not a reliable perceptual strategy.
Because the navigation space is built from audio descriptors and dimensionality reduction, it is explicitly optimized for perceptual organization, whereas the raw latent space is optimized for reconstruction.

## DAFx-Ready Paragraph
We evaluated local neighborhood coherence in raw VAE latent space and in the descriptor-derived navigation space used by Stable Audio Wanderer. Using latent-rate source-audio descriptors as a timbral reference, navigation-space neighbors showed lower robustly scaled composite timbral distance than raw-latent neighbors (mean 0.7743 vs 3.7236), supporting the claim that raw latent proximity is not a reliable perceptual navigation strategy. This motivates the paper's separation between perceptual control space and latent reconstruction space.
