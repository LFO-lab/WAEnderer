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
