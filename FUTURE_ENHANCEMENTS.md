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
- Smoother navigation: wander/reorganized traversal may find more plausible bridges between materials.
- More robust retrieval: related audio states can stay closer together despite minor acoustic variation.
