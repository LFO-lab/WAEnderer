# Phase 0 — Audit de faisabilité du double moteur SAME-S

Date : 1er octobre 2026. Référence examinée : `d5c758a70cf14f8924dab2668e607d43da5e6a9d` (`Fixing elements for performance`). Arbre Git propre au début de cet audit ; les modifications locales mentionnées dans la première roadmap sont désormais intégrées à ce commit.

**Conclusion : feu vert pour la phase 1.** Le décodage natif correspondant au graphe livré a été identifié et essayé sur CPU et MPS pour toutes les fenêtres paires T2 à T32. Aucun moteur applicatif ni transport n’a été modifié. La validation CUDA, l’écoute et la qualification temps réel restent des travaux ultérieurs.

## 1. Référence fonctionnelle

Résultats des tests existants : **150 réussites, 1 ignoré**, avec trois avertissements de dépréciation WebSocket. Les suites `test_web_onnx_ui.js`, `test_web_wander_ui.js` et `test_web_erae_ui.js` passent également.

Commande Python utilisée depuis la racine du dépôt :

```sh
.venv/bin/python -m pytest -q \
  tests/test_onnx_decoder.py tests/test_app_decoder_resource.py \
  tests/test_pipeline_onnx.py tests/test_onnx_transport.py \
  tests/test_decoder_player.py tests/test_overlap_add.py \
  tests/test_manual_windows.py tests/test_wander_windows.py \
  tests/test_window_controls.py tests/test_ws_server.py \
  tests/test_erae_osc.py tests/test_erae_visual.py tests/test_export_web_decoder.py
node test_web_onnx_ui.js
node test_web_wander_ui.js
node test_web_erae_ui.js
```

Le test ignoré est `test_opt_in_real_same_s_bundle`, qui nécessite `SAW_REAL_SAME_S_BUNDLE`. Les essais réels décrits ci-dessous utilisent directement la ressource ONNX applicative et les poids natifs en cache ; ils ne remplacent pas ce test de bundle.

Le premier passage dans le bac à sable a produit 143 réussites, 1 test ignoré et 7 échecs/erreurs liés aux sockets OSC interdites. Le passage hors bac à sable donne le résultat de référence ci-dessus. Ce n’est pas une régression du dépôt.

Comportements à préserver, couverts par les suites examinées :

- Décodage Web exclusivement SAME-S, fenêtres paires, PCM stéréo 44,1 kHz, 4 096 échantillons par latent.
- Générations active/candidate avec leurs états de planification et overlap-add distincts ; rejet des résultats obsolètes.
- Continuité des transitions, changements de fenêtres et contrôles adaptatifs ; provenance des frames et curseur de présentation.
- Arrêt attendant les décodages en cours avant remise à zéro des buffers ; erreur persistante visible jusqu’au prochain démarrage.
- Contrats WebSocket, commandes OSC/Erae et affichage synchronisé.

Deux détails de cycle de vie à prendre en compte en phase 3 : le VAE de prétraitement est déjà libéré à la fin du prétraitement, et arrêter le transport ne remet pas à lui seul le pipeline en phase `idle`. Réutiliser le VAE de prétraitement ou reconfigurer uniquement après un simple `stop` ne peut donc pas être supposé fonctionnel.

## 2. Provenance des modèles et choix du chargeur

| Élément | Valeur vérifiée |
| --- | --- |
| Dépôt des poids | `stabilityai/SAME-S` |
| Révision des poids | `fbeb3dcf53a326e5682f38e22e7f740202d44232` |
| SHA-256 ONNX livré | `333dc948f52fa59e9a80a066ffd2ca801d3a54b2fb07c93aa87cac845c547872` |
| SHA-256 `model_config.json` en cache | `c329dd0a6f61d0b3ea4f23930059a6c00437005692fed4310924eb286253303a` |
| SHA-256 `model.safetensors` en cache | `c19698ce3a0b462acb967ee495e9eb7945221f236968c50206cce8cf22b3d305` |
| Taille du checkpoint natif | 433 007 588 octets |
| Commit de `stable-audio-3` installé | `779434a908193105335fd8d833418603625b2859` |

Ces empreintes ont été calculées sur les fichiers locaux. Les empreintes config/poids constituent une référence de cet audit, pas une preuve de provenance supplémentaire indépendante du cache et de la révision indiquée. Aucun téléchargement ni réexport n’a été effectué.

**Décision :** reprendre le mécanisme de résolution épinglée de `_load_pinned_same_s_autoencoder` dans `bin/export_vst_bundle.py`, puis le déplacer dans une couche de chargement partagée lors de l’implémentation. Le runtime ne doit pas dépendre d’un script d’export. Utiliser les deux fichiers de la révision explicite, contrôler leur identité et échouer clairement s’ils sont indisponibles.

Le chargeur historique `AutoencoderModel.from_pretrained("same-s")` n’est pas la référence de provenance : sa résolution peut préférer un checkpoint Stable Audio 3 complet déjà en cache et ne reçoit pas cette révision explicite. Son enveloppe ne fournit pas non plus les méthodes `.to()`/`.eval()` que l’adaptateur cherche conditionnellement ; passer le périphérique explicitement est nécessaire.

Le chargeur amont `copy_state_dict` tolère des clés absentes/incompatibles. La future intégration devra vérifier les tenseurs attendus pour le décodage, au lieu de considérer un retour sans exception comme preuve de chargement complet.

## 3. Chemin de décodage natif retenu

Source inspectée : fichiers installés de `stable_audio_3`, au commit ci-dessus.

```text
AutoencoderModel.decode(..., chunked=False)
    → AudioAutoencoder.decode_audio(..., chunked=False)
    → AudioAutoencoder.decode(...)
    → bottleneck.decode → decoder → pretransform.decode → soft_clip éventuel
```

Le chemin d’export appelle directement `decode_audio(..., chunked=False)`. Dans cette version installée, le wrapper public ne rajoute ni normalisation, ni padding, ni découpage lorsque `chunked=False`. Le décodage interne du modèle conserve toutefois ses propres opérations de padding/segmentation. La dénormalisation des latents du corpus reste à effectuer avant l’entrée dans ce chemin.

Vérification expérimentale à T2 sur CPU, avec les mêmes poids et la même graine : sortie du wrapper public et sortie du chemin d’export **exactement égales**.

**Décision :** appeler explicitement `AudioAutoencoder.decode_audio(latents, chunked=False)` en évaluation et sous `torch.inference_mode()`. Conserver le bottleneck et la reconstruction par le prétransform ; appeler uniquement le sous-module `decoder` serait incorrect.

## 4. Retrait de l’encodeur et mémoire

Le chargeur amont construit systématiquement l’encodeur et le décodeur. Il n’offre pas d’option de chargement direct du décodeur seul dans la version inspectée.

Le décodage n’utilise pas l’encodeur. Dans une instance isolée chargée pour l’audit, retirer sa référence donne la même sortie T2 sur CPU à graine identique. Cette instance réduite a ensuite décodé les 16 fenêtres sur CPU et MPS.

| Composant | Octets de paramètres float32 |
| --- | ---: |
| Encodeur | 214 915 168 |
| Décodeur | 218 060 896 |
| Bottleneck | 2 052 |
| Prétransform | 0 |
| Total complet | 432 978 116 (~412,9 Mio) |
| Total après retrait de l’encodeur | 218 062 948 (~208,0 Mio) |

**Décision pour la première version :** charger l’autoencodeur épinglé sur CPU, retirer l’encodeur de cette instance exclusivement dédiée à la lecture, puis transférer les composants restants sur le périphérique choisi. Garder l’enveloppe `AudioAutoencoder` pour préserver la sémantique de décodage. Un chargement sélectif des seuls tenseurs utiles pourra être une optimisation ultérieure.

Cette stratégie économise environ 205 Mio de paramètres résidents mais ne supprime pas le pic mémoire du chargement CPU complet ni les copies temporaires du checkpoint. Les paramètres ne représentent pas toute la consommation GPU : à la fin du probe MPS, PyTorch rapporte 222 789 376 octets alloués aux tenseurs et 1 183 481 856 octets alloués par le driver. Ce dernier chiffre comprend d’autres allocations/caches et n’est pas une mesure du pic mémoire ni d’une fuite.

## 5. Stochasticité : correction importante du protocole de parité

Le modèle ajoute du bruit **même en évaluation** :

- `SoftNormBottleneck.decode` utilise `randn_like` avec un facteur de bruit de `1e-3` en mode évaluation lorsque `noise_regularize=true`.
- Le décodeur contient aussi une injection de bruit sur les tokens (`mask_noise=0.01`).
- Le graphe ONNX contient deux nœuds `RandomNormalLike` au premier niveau.

Les appels successifs avec les mêmes latents diffèrent donc dans les deux moteurs. Remettre la graine PyTorch à zéro reproduit exactement les sorties testées au sein de chaque périphérique, mais ne synchronise pas les générateurs aléatoires de PyTorch et ONNX Runtime.

| Comparaison | RMSE observée, toutes fenêtres |
| --- | --- |
| PyTorch CPU / ONNX | 0,001959 à 0,002131 |
| PyTorch MPS / ONNX | 0,001978 à 0,002161 |
| PyTorch CPU / nouvel appel CPU sans réinitialiser la graine | 0,001944 à 0,002091 |
| PyTorch MPS / nouvel appel MPS sans réinitialiser la graine | 0,001987 à 0,002100 |
| ONNX / nouvel appel ONNX, sur les deux probes | 0,001942 à 0,002117 |

**Interprétation :** les écarts entre moteurs sont du même ordre que la variabilité entre deux appels d’un même moteur. Cela ne prouve pas l’absence de toute erreur de conversion, mais interdit d’attribuer tout l’écart à ONNX ou au GPU.

**Décision :** conserver la stochasticité native. Ne pas désactiver le bruit, modifier le graphe ou réinitialiser la graine à chaque fenêtre du runtime pour obtenir artificiellement une égalité. Tester séparément la variabilité interne et les écarts entre moteurs ; les appels seront initialement sérialisés côté GPU.

## 6. Fenêtres et résultats du probe

Les métadonnées applicatives annoncent les 16 fenêtres paires T2 à T32. Le rapport historique `decoder_parity.json` couvre seulement T2/T4/T8/T16/T32 ; `even_window_validation.json` complète la couverture des 16 fenêtres pour la même empreinte ONNX et la même révision de poids.

La campagne de phase 0 a décodé chacune des 16 fenêtres avec les poids réels. Pour chaque fenêtre : entrée normale de moyenne 0, écart-type 0,05, graine NumPy égale à T, sortie stéréo float32 finie de forme `[1, 2, T × 4096]`.

| T | Parité historique | Rapport fenêtres paires | CPU natif / MPS : forme et valeurs finies | SNR MPS / ONNX (dB) | Médiane native MPS indicative (ms) |
| ---: | :---: | :---: | :---: | ---: | ---: |
| 2 | Oui | Oui | OK / OK | 23,48 | 13,0 |
| 4 | Oui | Oui | OK / OK | 24,52 | 16,5 |
| 6 | — | Oui | OK / OK | 24,15 | 16,0 |
| 8 | Oui | Oui | OK / OK | 24,25 | 17,4 |
| 10 | — | Oui | OK / OK | 24,15 | 21,2 |
| 12 | — | Oui | OK / OK | 24,10 | 18,8 |
| 14 | — | Oui | OK / OK | 24,31 | 21,9 |
| 16 | Oui | Oui | OK / OK | 24,43 | 22,6 |
| 18 | — | Oui | OK / OK | 24,11 | 23,5 |
| 20 | — | Oui | OK / OK | 24,45 | 25,7 |
| 22 | — | Oui | OK / OK | 24,32 | 27,2 |
| 24 | — | Oui | OK / OK | 24,14 | 29,2 |
| 26 | — | Oui | OK / OK | 24,15 | 30,0 |
| 28 | — | Oui | OK / OK | 24,26 | 31,1 |
| 30 | — | Oui | OK / OK | 24,42 | 33,8 |
| 32 | Oui | Oui | OK / OK | 24,40 | 36,0 |

Le seuil synthétique existant, **SNR > 20 dB et RMSE < 0,005**, est respecté par les 32 comparaisons natif/ONNX CPU et MPS. Il est retenu comme premier détecteur de régression sur ces entrées synthétiques, pas comme certificat de qualité perceptuelle ni comme seuil universel sur les corpus réels.

Les mesures natives sont des médianes de trois appels après préparation, incluant transfert des latents et retour PCM sur CPU. Elles ne comprennent ni navigation, ni overlap-add, ni concurrence, ni sortie audio. Aucun p99 exploitable, résultat d’underrun ou avantage chiffré sur ONNX ne peut être déduit de cet échantillon. La session ONNX de ce probe reprend le style du script de validation existant : un thread intra-op et les optimisations ORT par défaut ; le runtime applicatif impose `ORT_ENABLE_BASIC` et un thread inter-op. La future campagne de performance devra utiliser les paramètres exacts du runtime.

## 7. Matériel et environnement

- Machine locale : MacBookPro18,2, Apple M1 Max, 64 Gio, macOS 26.6.2 arm64.
- MPS : disponible hors bac à sable, tenseur et décodeur réellement exécutés sur `mps`; variable `PYTORCH_ENABLE_MPS_FALLBACK` absente.
- CUDA : indisponible sur cette machine. Aucun GPU NVIDIA ni hôte CUDA n’a été validé ; prévoir une machine dédiée avant d’annoncer le support CUDA.
- Python de travail : `.venv/bin/python`, 3.12.13. Le `python3` global est en 3.10 et ne constitue pas l’environnement du projet.

| Dépendance observée | Version |
| --- | --- |
| torch | 2.13.0 |
| torchaudio | 2.11.0 |
| onnxruntime | 1.28.0 |
| onnx | 1.22.0 |
| numpy | 2.5.2 |
| stable-audio-3 | 0.1.0, commit épinglé ci-dessus |
| huggingface-hub | 1.27.0 |
| safetensors | 0.8.0 |
| pytest | 9.1.1 |

Ces versions décrivent l’environnement qui a réussi le probe ; elles ne constituent pas une nouvelle configuration d’installation recommandée. `stable-audio-3` déclare `torch==2.7.1` et `torchaudio==2.7.1`, différents de l’environnement présent, et n’est pas verrouillé comme dépendance normale par `uv.lock`. Le README utilise une installation `--no-deps`. La résolution d’un environnement propre et compatible reste à vérifier en phase 2/6 ; aucun paquet ni verrouillage n’a été changé ici. Flash Attention est absent et la bibliothèque utilise ses chemins alternatifs.

## 8. Reproduction et protocole retenu pour la suite

Le script [audit_dual_inference_phase0.py](../eval_scripts/audit_dual_inference_phase0.py) est un outil d’évaluation isolé. Il force le mode hors ligne, vérifie l’empreinte ONNX, utilise la révision de poids explicite et écrit uniquement le rapport demandé. Les poids doivent déjà être en cache. Les graines ne garantissent pas une reproduction identique des tirages ONNX entre processus.

```sh
PYTHONPATH=. .venv/bin/python eval_scripts/audit_dual_inference_phase0.py \
  --device cpu --output docs/dual_inference_phase0_cpu.json
PYTHONPATH=. .venv/bin/python eval_scripts/audit_dual_inference_phase0.py \
  --device mps --output docs/dual_inference_phase0_mps.json
```

L’essai MPS nécessite un processus ayant accès au GPU. Les rapports conservés sont [CPU](dual_inference_phase0_cpu.json) et [MPS](dual_inference_phase0_mps.json). `snr_db: null` accompagné de `exact_equal: true` signifie une erreur nulle, donc un SNR infini non représenté en JSON.

Pour les phases suivantes :

1. Conserver poids, bibliothèque, fenêtres, paramètres de session et empreintes dans chaque rapport.
2. Rejouer toutes les fenêtres paires sur au moins huit graines synthétiques et des fenêtres brutes dénormalisées de corpus représentatifs, en comparant aussi les répétitions internes de chaque moteur.
3. Appliquer le seuil historique uniquement aux entrées synthétiques correspondantes ; analyser séparément silence/faible énergie où le SNR est peu informatif. Fixer les limites propres aux corpus avant la campagne d’acceptation.
4. Comparer gain, spectre, formes, valeurs finies et rendu après overlap-add ; effectuer l’écoute prévue en phase 5.
5. Mesurer p95/p99, mémoire, transitions, commandes et interruptions avec le transport réel, pendant au moins 10 minutes par scénario déclaré pris en charge. Aucun fallback GPU→CPU implicite.
6. Répéter sur une machine CUDA et documenter ses versions avant de qualifier ce périphérique.

## 9. Décisions transmises à l’implémentation

- **Phase 1 :** extraire le contrat commun, ONNX seul ; ne pas changer l’audio ou son caractère stochastique.
- **Phase 2 :** chargement épinglé sur CPU, retrait de l’encodeur de l’instance dédiée, transfert explicite, décodage complet non découpé en float32. MPS est le premier périphérique matériel disponible pour valider l’intégration ; CUDA reste conditionnel à ses essais.
- **Phase 3 :** appels GPU sérialisés initialement, gestion explicite de la sortie/reconfiguration de `perform`, drainage avant libération et remise à zéro des buffers.
- **Validation :** liste commune T2 à T32 paires, seuil synthétique existant conservé, contrôle de la stochasticité et qualification audio/temps réel distincts.

La phase 0 est terminée : référence, provenance, chemin natif, stratégie mémoire, couverture de fenêtres, matériel disponible et protocole sont établis. Elle ne valide pas encore la lecture GPU intégrée, CUDA, l’installation depuis un environnement vierge ni la qualité à l’écoute.
