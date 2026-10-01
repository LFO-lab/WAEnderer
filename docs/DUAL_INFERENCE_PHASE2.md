# Phase 2 — Décodeur natif SAME-S

Implémentation du moteur terminée le 1er octobre 2026. Validation numérique réelle réussie sur CPU et Apple MPS. CUDA est implémenté mais non qualifié sur matériel ; la validation perceptuelle des extraits reste à effectuer. Le sélecteur Web et la fabrique du pipeline restent en phases 3 et 4.

## API et comportement

```python
import numpy as np
from stable_audio_wanderer.vae import SameSTorchDecoder

decoder = SameSTorchDecoder(device="mps", local_files_only=True)
decoded = decoder.decode(np.zeros((2, 256), dtype=np.float32))
print(decoded.audio.shape)       # (8192, 2), PCM CPU float32 contigu
print(decoded.decode_time_ms)    # attente éventuelle + calcul + transferts + validation
print(decoder.info)              # périphérique réel, empreintes, révisions, warm-up
```

- Périphérique obligatoire : `mps`, `cuda`/`cuda:N` ou `cpu` explicitement pour diagnostic. Pas de `auto`, pas de repli CPU ni ONNX. La variable globale `config.DEVICE` n’est pas modifiée.
- Le modèle utilise le checkpoint original `stabilityai/SAME-S`, à la révision `fbeb3dcf53a326e5682f38e22e7f740202d44232`. Le résolveur vérifie les empreintes config/poids consignées en phase 0.
- Le chargement reste sur CPU et utilise `load_state_dict(..., strict=True)` : clés manquantes, inattendues et dimensions incorrectes entraînent un refus. Les 244 tenseurs du checkpoint réel correspondent au modèle.
- L’encodeur est retiré de l’instance dédiée avant le transfert. Bottleneck, décodeur et prétransform sont conservés, soit 218 062 948 octets de paramètres float32 (~208 Mio). Ce chiffre ne représente pas la mémoire totale ou le pic de chargement.
- Le modèle est figé en évaluation. Chaque appel utilise `torch.inference_mode()` et `decode_audio(..., chunked=False)` ; aucun changement de précision ou du bruit natif n’est appliqué.
- Les 16 fenêtres paires T2 à T32 sont échauffées et leurs sorties vérifiées avant que le constructeur retourne. Pas de padding, de normalisation du corpus ou d’overlap-add dans ce moteur.
- Entrée : latents bruts dénormalisés, float32 `[T,256]`, finis. Sortie : PCM CPU float32 `[T*4096,2]`, fini, contigu et possédé par l’appelant. Les entrées non contiguës ou en lecture seule sont acceptées sans mutation.
- Un verrou par instance sérialise les appels pour respecter le contrat du transport à deux producteurs. Aucun reset de graine n’est fait par le runtime.
- `decode_time_ms` comprend l’attente du verrou, le transfert des latents, l’inférence, le retour CPU bloquant et la préparation/validation du PCM.
- `NativeDecoderLoadError` identifie les problèmes de préparation ; `DecoderRuntimeError` couvre les requêtes et échecs d’inférence. Les erreurs ne déclenchent pas de changement de moteur.

Le mode natif n’ouvre ni session ONNX ni graphe ONNX. Les poids natifs sont sa seule ressource de modèle. Les tests de comparaison utilisent ONNX séparément comme référence.

## Installation indépendante de l’environnement existant

Une installation propre a été réalisée dans `/private/tmp/waenderer-phase2-native-env`, sans réutiliser les paquets de `.venv`, sur le Mac M1 Max de l’audit. Elle utilise **Torch et Torchaudio 2.7.1**, comme demandé par le commit amont, sans `--no-deps` ni remplacement de l’environnement actuel. `uv pip check` confirme la compatibilité des 71 paquets installés.

Depuis la racine du dépôt, pour reproduire le profil macOS arm64 / Python 3.12 :

```sh
uv venv --python 3.12 .venv-native
uv pip install --python .venv-native/bin/python -r requirements-same-s-native-macos.lock
uv pip check --python .venv-native/bin/python
```

Le [profil d’installation](../requirements-same-s-native.txt) fixe le commit de `stable-audio-3` et ses versions Torch. Le [verrou de référence macOS](../requirements-same-s-native-macos.lock) enregistre également toutes les versions transitives et les outils de validation installés. C’est un profil spécifique à la validation native, pas un remplacement de `uv.lock` ni un verrou CUDA multiplateforme.

Versions principales vérifiées : Python 3.12.13, Torch/Torchaudio 2.7.1, `stable-audio-3` 0.1.0 au commit `779434a908193105335fd8d833418603625b2859`, ONNX Runtime 1.30.0, NumPy 2.5.3, safetensors 0.8.0. Flash Attention n’est pas installé ni nécessaire pour les essais réalisés.

Les poids étaient déjà en cache et n’ont pas été téléchargés pendant les tests. Le constructeur est hors ligne par défaut. Pour obtenir les fichiers épinglés si nécessaire, passer explicitement `local_files_only=False` après configuration normale de l’accès Hugging Face. Une erreur d’accès ou d’identité reste visible ; aucun autre checkpoint en cache n’est substitué.

Pour MPS, démarrer le processus avec `PYTORCH_ENABLE_MPS_FALLBACK` absent ou égal à `0`. Un processus sans accès au GPU échoue explicitement. Les essais réels MPS ont été exécutés hors bac à sable avec ce repli désactivé.

## Validation

### Tests automatisés

**182 tests Python réussis, 1 optionnel ignoré**, dans l’environnement propre, couvrant le nouveau moteur, le contrat commun, ONNX, le pipeline, les transports, le lecteur, l’overlap-add, les fenêtres, WebSocket, l’export et OSC/Erae. Trois suites Web JavaScript réussies. Les avertissements existants concernent l’API WebSocket dépréciée.

Les nouveaux tests couvrent notamment : chargement strict avant retrait de l’encodeur, empreintes/révision/cache hors ligne, absence de fallback, choix du périphérique, bibliothèque manquante, état d’évaluation/inférence, toutes les fenêtres d’échauffement, identité de la référence livrée, dispositions des tenseurs, entrées immuables, sorties invalides, erreurs d’opérations, concurrence sérialisée et absence de réinitialisation RNG.

Le test ignoré exige un bundle utilisateur via `SAW_REAL_SAME_S_BUNDLE`. Il ne remplace pas les essais réels du graphe applicatif et du modèle natif effectués ci-dessous.

### Comparaisons sur les modèles réels

Le script [validate_torch_decoder.py](../eval_scripts/validate_torch_decoder.py) effectue huit comparaisons par fenêtre sur les 16 fenêtres, avec entrées synthétiques d’écart-type 0,05 et graines NumPy `T*1000+seed`. Il vérifie aussi deux appels concurrents au décodeur réel. L’ONNX est chargé avec les paramètres du runtime applicatif.

| Périphérique natif | Comparaisons réussies | RMSE min–max | SNR min–max |
| --- | ---: | --- | --- |
| CPU explicite | 128 / 128 | 0,001945–0,002213 | 23,53–24,68 dB |
| MPS | 128 / 128 | 0,001955–0,002228 | 23,63–24,80 dB |

Tous les essais respectent le seuil synthétique historique **SNR > 20 dB, RMSE < 0,005**. Les différences restent compatibles avec la stochasticité constatée en phase 0 ; cette tolérance n’est pas une mesure universelle de qualité perceptuelle.

Rapports bruts : [CPU](dual_inference_phase2_cpu.json), [MPS](dual_inference_phase2_mps.json). Les médianes natives MPS observées sont d’environ 24,3 ms à T2, 25,0 ms à T8 et 47,1 ms à T32. Le chargement incluant l’échauffement a pris environ 6,2 secondes. Il s’agit de mesures courtes de décodage, pas d’une qualification de latence p99, d’underruns ou de performance du transport intégré.

Reproduction :

```sh
PYTHONPATH=. .venv-native/bin/python eval_scripts/validate_torch_decoder.py \
  --device cpu --output /tmp/native-cpu.json
PYTHONPATH=. .venv-native/bin/python eval_scripts/validate_torch_decoder.py \
  --device mps --output /tmp/native-mps.json --audio-dir /tmp/native-listening
```

### Extraits pour l’écoute

Les fichiers locaux `eval_out/dual_inference_phase2/source.wav`, `onnx.wav` et `native.wav` ont été générés. Ils sont reproductibles via `--audio-dir` et restent dans le répertoire d’évaluation ignoré par Git.

La source stéréo synthétique contient des harmoniques à enveloppes percussives, encodées par le SAME-S original épinglé. Les mêmes fenêtres T8 et hops de quatre latents sont ensuite décodés et assemblés par overlap-add dans les deux moteurs, sans normalisation ou clipping supplémentaire. Les deux rendus ont la même longueur de 245 760 échantillons et une différence de RMSE 0,000885 / SNR 37,53 dB sur cette entrée.

**L’écoute humaine n’a pas été validée.** Les fichiers permettent de comparer niveau, attaques, queues et raccords ; les chiffres seuls ne permettent pas de déclarer cette étape perceptuelle terminée.

## Périmètre restant

- CUDA : code de sélection/transfert/erreur et tests de résolution présents ; exécuter le script sur un GPU NVIDIA avant de déclarer le support matériel qualifié.
- Écoute : valider les deux extraits, puis les corpus représentatifs de la phase 5.
- Phases 3/4 : intégrer le cycle de vie au pipeline et le sélecteur à l’interface. Le Web continue à utiliser uniquement ONNX CPU.
- Phase 5 : sessions prolongées, changements de génération/fenêtre, latence p99 et absence d’underruns sur le transport réel.

Le code de phase 2 est disponible et testé ; sa qualification complète conserve explicitement les réserves CUDA et écoute ci-dessus.
