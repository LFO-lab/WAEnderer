# Phase 0 — Inventaire et diagnostic du menu

Statut : **terminée le 2 octobre 2026**. Aucun nouvel export ONNX, téléchargement, chargement EAR ni changement d’environnement. Le graphe SAME-S existant est conservé. Les modifications multi-VAE antérieures et le nouvel environnement `.venv-ear` ne sont pas remplacés.

## Diagnostic

Les réponses du véritable `PipelineManager` ont été enregistrées pour les deux corpus dans [l’inventaire JSON](multi_vae_phase0_inventory.json), puis rejouées dans le code JavaScript de production avec un DOM simulé.

| Corpus | Choix observés dans `.venv`, après correction CPU |
| --- | --- |
| BurntMemory, `same_s` | ONNX CPU, PyTorch CPU, PyTorch MPS ; CUDA visible mais indisponible |
| Rack, `stable_audio_open` | PyTorch CPU, PyTorch MPS ; CUDA visible mais indisponible ; aucun ONNX |

La disparition de SAME-S ONNX **avec Rack** est le filtrage attendu : son graphe ne peut pas décoder les latents Stable Audio Open. L’option réapparaît en revenant à BurntMemory. Dans cet état du code, aucune disparition n’a été reproduite avec un corpus SAME-S. Il n’est donc pas établi qu’une ancienne observation avec SAME-S provenait d’un cache navigateur ou d’un serveur obsolète.

Les tests rejouent deux clients neufs (rechargement), une déconnexion/reconnexion, la conservation du choix PyTorch CPU, le passage SAME-S → Rack → SAME-S, et une réponse où le graphe manque. Dans ce dernier cas ONNX reste visible mais désactivé. **Ce sont des tests du protocole et du DOM simulé, pas l’observation de la session navigateur personnelle de l’utilisateur.**

Un défaut distinct était certain : SAME-S PyTorch CPU était accepté par le chargeur mais absent de la découverte. L’entrée est maintenant ajoutée ; sa disponibilité dépend des dépendances et poids natifs, pas d’une garantie de temps réel. Les GPU restent inchangés. La matrice complète des futures options ONNX non-SAME-S sera réalisée dans les phases suivantes : elle n’est pas simulée comme disponible aujourd’hui.

## Inventaire vérifié

| VAE | Sources natives | ONNX local |
| --- | --- | --- |
| SAME-S | Config/checkpoint épinglés en cache ; empreintes enregistrées | Graphe applicatif présent, empreinte conforme au manifeste, chargement CPU vérifié |
| Stable Audio Open | Config/checkpoint en cache ; snapshot et empreintes enregistrés | Aucun dans les ressources applicatives ; exporteur VST existant, intégration Web à réaliser |
| EAR 44,1 kHz | `EAR_VAE/pretrained_weight/ear_vae_44k.pyt`, 591 453 838 octets | Aucun ; export à réaliser |
| EAR 48 kHz | `EAR_VAE/pretrained_weight/ear_vae_v2_48k.pyt`, 673 753 246 octets | Aucun ; export à réaliser |

Chemins absolus, SHA-256 des poids/configurations et réponses de découverte : [inventaire](multi_vae_phase0_inventory.json). Les fichiers EAR sont lus pour calculer leurs empreintes, sans désérialiser ni exécuter les checkpoints.

Les configurations sélectionnées par la convention actuelle sont `config/model_config.json` pour EAR 44,1 kHz et `config/ear_vae_v2.json` pour EAR 48 kHz, dans `/Users/dthibault/Documents/GitHub/EAR_VAE`. Elles annoncent 64 dimensions latentes et respectivement 1024 et 960 échantillons par latent, d’après le produit des strides. **Ces valeurs restent à confirmer par le chargement et le décodage réels en phase 3.** L’adaptateur peut notamment réconcilier la présence du transformer selon les clés du checkpoint.

## Environnements

L’inventaire a été exécuté hors du sandbox pour que la découverte voie MPS, avec `HF_HUB_OFFLINE=1` et sans sortie audio.

| Dépendance | `.venv` utilisé par l’audit | `.venv-ear` fourni |
| --- | --- | --- |
| Torch | 2.13.0 | 2.14.1 |
| ONNX Runtime | 1.28.0 | 1.26.0 |
| Diffusers | 0.35.1 | 0.35.1 |
| stable-audio-3 | 0.1.0 | absent |
| descript-audio-codec | absent | 1.0.0 |

Ainsi, EAR possède bien ses poids sur disque, mais la découverte depuis `.venv` annonce ses dépendances manquantes. L’inventaire de paquets dans `.venv-ear` ne prouve pas encore le bon chargement de ces checkpoints ni leur exportabilité. Aucun mélange automatique des environnements n’est effectué.

## Preuves et limites

- Graphe SAME-S : SHA-256 `333dc948f52fa59e9a80a066ffd2ca801d3a54b2fb07c93aa87cac845c547872`, identique au manifeste existant.
- Chargement ONNX Runtime CPU hors ligne et 16 décodages T2–T32 : PCM float32 fini, formes attendues `[T × 4096, 2]`.
- 13 tests Python ciblés réussis (découverte et ressource ONNX), dont matrice CPU des quatre VAE et distinction présence/validation des poids EAR.
- Trois suites Web réussies, incluant le rejeu des réponses réelles de l’audit pour le menu.
- Aucun verdict de parité, d’écoute, de temps réel ou d’export EAR n’est déduit de ces vérifications. La phase 1 n’est pas commencée.

## Reproduire

Depuis `WAEnderer_python` :

```sh
HF_HUB_OFFLINE=1 PYTHONPATH=. .venv/bin/python eval_scripts/audit_multi_vae_phase0.py \
  --same-s-corpus corpus/BurntMemory_20260915_211501 \
  --sao-corpus corpus/Rack_20260428_181107 \
  --ear-44k /Users/dthibault/Documents/GitHub/EAR_VAE/pretrained_weight/ear_vae_44k.pyt \
  --ear-48k /Users/dthibault/Documents/GitHub/EAR_VAE/pretrained_weight/ear_vae_v2_48k.pyt \
  --environment-python .venv-ear/bin/python \
  --output docs/multi_vae_phase0_inventory.json
.venv/bin/python -m pytest -q tests/test_decoder_availability.py tests/test_app_decoder_resource.py
node test_web_onnx_ui.js
node test_web_wander_ui.js
node test_web_erae_ui.js
```

Pour voir l’entrée SAME-S PyTorch CPU ajoutée, redémarrer le serveur et recharger la page. Aucune sélection du VAE ne remplace celui du corpus.
