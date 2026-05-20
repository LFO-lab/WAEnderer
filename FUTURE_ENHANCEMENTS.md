# Future Enhancements

This file tracks improvements that are intentionally deferred.

## 1) Policy Navigation Preset System

Add named preset recall/save for policy controls so meaningful parameter sets can be switched instantly.

Where this is useful:
- Live performance: fast A/B between behavior profiles without manual slider setup.
- Corpus-specific workflows: quickly recall known-good settings for a given dataset.
- Iteration/testing: reproducible control setups across sessions.

## 2) Improved Morphological Unit Segmentation

Replace the current heuristic unit splitting with a more robust morphology-aware segmentation method.

Where this is useful:
- Better V2 training data: cleaner unit boundaries improve learned trajectory quality.
- More coherent recomposition: transitions happen at structurally meaningful points.
- Cross-corpus reliability: segmentation behavior is less arbitrary and more consistent.

## 3) Corpus Augmentation For Latent Overlap

Augment source audio during dataset preparation to encourage denser overlap in the VAE latent space.

Possible augmentations:
- Time-stretching: create nearby temporal variants without changing the source identity too drastically.
- Transposition: add pitch-shifted versions so related material occupies more connected regions.
- Gain variation: expose the encoder to level differences that should still map to similar content.
- Other light transformations: small EQ, filtering, or dynamic changes where musically appropriate.

Where this is useful:
- Better corpus connectivity: more neighboring examples can reduce isolated latent pockets.
- Smoother navigation: random/reorganized traversal may find more plausible bridges between materials.
- More robust retrieval: related audio states can stay closer together despite minor acoustic variation.

## 4) Expose MLP Training Parameters

Make the reorganized transition-model MLP training parameters more explicit and easier to tune from the training workflow.

Parameters worth exposing/documenting clearly:
- Hidden dimension: adjust model capacity for larger or smaller corpora.
- Layer count: control MLP depth without editing code.
- Dropout: tune regularization for more or less aggressive fitting.
- Learning rate and weight decay: improve optimizer control during transition-model training.
- Batch size, epoch count, validation split, and seed: make runs easier to reproduce and compare.

Where this is useful:
- Faster iteration: easier hyperparameter sweeps when the reorganized model underfits or overfits.
- Corpus adaptation: different datasets may need different MLP capacity and regularization.
- Reproducibility: training settings are easier to track and revisit later.
