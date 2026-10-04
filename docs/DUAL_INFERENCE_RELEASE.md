# Livraison — Deux moteurs SAME-S

## Formats de livraison

L’application Web est livrée dans le dépôt ou l’archive source (`sdist`), qui inclut `bin/`, `web/`, les scripts d’évaluation, les profils de dépendances et les rapports. Exécuter les commandes depuis cette racine. Le wheel contient la bibliothèque Python et les ressources du décodeur ; il n’est pas présenté comme un exécutable Web autonome. Aucun corpus utilisateur ni WAV d’évaluation n’entre dans l’archive source.

## Installer et utiliser

Le parcours existant ONNX CPU garde les dépendances de `pyproject.toml` / `uv.lock`. Torch reste requis par les composants historiques de navigation/prétraitement ; ONNX n’exige pas la bibliothèque native `stable-audio-3` ni ses poids. Le graphe applicatif SAME-S doit être préparé/livré selon le README et la checklist existante.

Pour le GPU natif, la décision de livraison est de conserver **un environnement séparé**, défini par `requirements-same-s-native.txt` et son verrou `requirements-same-s-native-macos.lock`. Aucun extra mélangeant des versions Torch incompatibles n’est ajouté au verrou principal. Ce profil décrit la qualification historique. Le contrat d’installation actuel et le profil natif combiné sont décrits dans [INSTALLATION_PROFILES.md](INSTALLATION_PROFILES.md).

Profil macOS arm64 / Python 3.12 déjà vérifié :

```sh
uv venv --python 3.12 .venv-native
uv pip install --python .venv-native/bin/python -r requirements-same-s-native-macos.lock
uv pip check --python .venv-native/bin/python
```

Ce verrou n’est pas un profil Windows/Linux/CUDA qualifié. Le profil source épingle Torch/Torchaudio 2.7.1 et `stable-audio-3` au commit `779434a908193105335fd8d833418603625b2859`. Voir le [rapport d’installation](DUAL_INFERENCE_PHASE2.md).

Pour obtenir les poids natifs, avec votre accès Hugging Face configuré :

```sh
.venv-native/bin/python -c 'from stable_audio_wanderer.vae.same_s_weights import resolve_same_s_weights; print(resolve_same_s_weights(local_files_only=False))'
PYTORCH_ENABLE_MPS_FALLBACK=0 .venv-native/bin/python bin/serve.py
```

Le téléchargement est explicite. Le runtime Web utilise ensuite le cache hors ligne. Le résolveur vérifie la révision et les empreintes ; aucun remplacement automatique de modèle. Le moteur natif n’a pas besoin du graphe ONNX pour fonctionner.

Ouvrir `http://localhost:8080`, choisir le corpus puis **Perform → Decode with**. **Refresh availability** distingue présence du matériel, dépendances et poids ; **Start Perform** valide et prépare le moteur. Les choix sont **ONNX · CPU** ou **PyTorch · GPU · MPS/CUDA:N**. Pour changer : **Stop Perform to change decoder → choix → Start Perform**. Stop Decode reste une pause. Après une mise à jour du code, redémarrer le serveur et recharger la page sans cache.

## Statut matériel et fenêtres

| Parcours | Fenêtres | Preuves | Limites |
| --- | --- | --- | --- |
| ONNX CPU | T2, T4, …, T32 | Régression, graphe réel, parité et 10 minutes Core Audio sans incident | Scénario BurntMemory sur le M1 Max testé |
| PyTorch CPU diagnostic | T2, T4, …, T32 | Contrat et parité numérique réelle | Option API, pas une option GPU dans le menu |
| PyTorch MPS, Apple M1 Max | T2, T4, …, T32 | Parité, mémoire, écoute comparative et 10 minutes Core Audio sans incident | Scénario BurntMemory ; ONNX perçu légèrement plus bruiteux |
| PyTorch CUDA:N | T2, T4, …, T32 annoncées après warm-up | Implémentation et tests simulés | Aucune machine NVIDIA qualifiée ici |

Les deux moteurs sont stochastiques. Les tolérances numériques sont des détecteurs de régression, pas une preuve perceptuelle. Aucun gain de vitesse GPU n’est garanti. Les versions et mesures sont conservées dans les rapports des phases 0 à 5.

## Matrice de release

La livraison du code et la qualification audio sont distinctes. Avant d’annoncer un scénario matériel pris en charge, conserver :

- Le résultat de la suite Python complète et des trois suites Web, l’environnement exact et le commit du produit.
- Le rapport de parité synthétique et celui du corpus représentatif (empreintes incluses), les répétitions de chaque moteur et les rendus après overlap-add.
- Une session de dix minutes minimum sur **le périphérique audio réel**, pour chaque mode/fenêtre déclaré pris en charge : latences médiane/p95/p99, transitions, charge et mémoire, aucun underrun.
- Les cycles arrêt/changement/reprise et démarrage réseau désactivé.
- Une fiche d’écoute datée avec auditeur, corpus, verdict et anomalies ; préciser écouteurs/enceintes, attaques, queues, raccords et niveaux lorsque renseignés, sans inventer les informations manquantes. Conserver les WAV localement ; ne pas publier le corpus utilisateur.
- Un essai autonome `bin/perform.py` et un démarrage du serveur ONNX sans bibliothèque native.

Les commandes et limites de la campagne sont dans [le rapport de phase 5](DUAL_INFERENCE_PHASE5.md). La checklist globale reste [RELEASE_CHECKLIST.md](../RELEASE_CHECKLIST.md). L’absence de verdict d’écoute ou de validation matérielle bloque une annonce de qualification complète, pas l’accès aux moteurs expérimentaux.

## Audit de livraison du 2 octobre 2026

- `uv pip check` : les 71 paquets de l’environnement natif épinglé sont compatibles.
- Smoke du pipeline avec le graphe ONNX réel, corpus BurntMemory, imports `stable_audio_3` interdits : préparation, PCM T2 valide, arrêt et fermeture réussis. [Preuve](dual_inference_phase6_cpu.json).
- Le test CLI historique `bin/perform.py --help` passe avec les imports natifs interdits ; le serveur expose également son CLI. La lecture autonome sur matériel fait l’objet du rapport distinct ci-dessous.
- `SAW_RELEASE_BUILD=1 uv build --offline` : archive source et wheel construits, contrôle d’identité/provenance du modèle passé. Inspection des archives : interface, CLI, profils et rapports présents dans la source ; deux décodeurs et graphe ONNX présents dans le wheel. Aucun corpus ni WAV inclus.
- Contrôle de conformité du dépôt réussi ; régression complète : 252 tests réussis, 1 optionnel ignoré, trois suites Web réussies.

**Conclusion : phase 6 terminée pour la livraison ONNX CPU / MPS documentée.** Le contrôle de phase 5 et la lecture autonome physique passent. CUDA demeure expérimental et non qualifié. Aucun déploiement ni publication externe n’a été effectué.

## Contrôle automatique des preuves

Le contrôle suivant vérifie les rapports du scénario explicitement déclaré ; il ne qualifie aucun périphérique par extrapolation :

```sh
PYTHONPATH=. .venv/bin/python bin/check_dual_inference_qualification.py docs/dual_inference_qualification.json
```

Il refuse une horloge simulée, des politiques de replay, une durée insuffisante, des frames manquantes, un underrun, des fenêtres/modes non exercés, un corpus différent entre les preuves ou l’absence de verdict d’écoute. Le manifeste limite la qualification à ONNX CPU et MPS sur la sortie Core Audio mesurée. CUDA reste une option expérimentale non qualifiée, exclue de cette matrice faute de matériel NVIDIA.

Le smoke autonome est reproductible avec `eval_scripts/validate_standalone.py --corpus corpus/BurntMemory_20260915_211501 --output docs/dual_inference_phase6_standalone.json`. Il démarre le véritable `bin.perform` en mode manuel, sortie muette, poids en cache, vérifie vingt secondes de statistiques audio puis demande l’arrêt propre du processus.

## Lecture autonome et correction d’arrêt

Le [smoke physique final](dual_inference_phase6_standalone.json) utilise les haut-parleurs MacBook Pro à volume nul, les poids en cache et `HF_HUB_OFFLINE=1` : 22 relevés audio, 21 avec décodage effectif, zéro underrun et sortie normale du processus après SIGINT.

Le [premier essai](dual_inference_phase6_standalone_baseline.json) avait zéro incident pendant la lecture mais 41 pendant l’arrêt : le lecteur continuait à consommer le tampon alors que les producteurs étaient arrêtés et que le programme attendait les threads. `TransportController.stop()` arrête désormais la sortie audio avant ces attentes, puis efface les buffers après leur fermeture. Un test vérifie cet ordre ; le second essai physique passe. Le chargement du modèle et le décodage historique restent identiques.

Les sorties Web et autonomes sont mesurées muettes ; l’écoute comparative porte sur les WAV de phase 5. Le périphérique et les détails d’écoute non fournis par l’utilisateur sont indiqués comme inconnus dans sa fiche. Les résultats ne constituent pas une qualification de tous les périphériques ou de toutes les charges système.
