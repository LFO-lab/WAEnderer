# Réactivation des VAE existants dans Perform

L’erreur `Corpus VAE 'stable_audio_open' is incompatible; expected 'same_s'` venait du chemin Web imposé lors de l’intégration ONNX. La première roadmap de double inférence était limitée à SAME-S ; ses phases terminées ne réactivaient pas les autres adaptateurs. Le corpus Rack n’a pas besoin d’être réencodé.

## Chemins disponibles

| VAE du corpus | Décodage Web | Préparation |
| --- | --- | --- |
| SAME-S | ONNX CPU ou PyTorch SAME-S épinglé | Parcours existant conservé |
| Stable Audio Open | Adaptateur PyTorch `AutoencoderOobleck` existant | Config et poids en cache Hugging Face, sans dépendance à `stable-audio-3` |
| EAR 44.1 kHz / 48 kHz | Adaptateur PyTorch EAR existant | Chemin `.pyt` dans son dépôt EAR, configuration et dépendances EAR dont `descript-audio-codec` |

Le VAE est lu dans `corpus.npz`, jamais remplacé par celui choisi pour un autre corpus. L’interface recharge les choix compatibles au changement de corpus et ignore les réponses concernant un ancien corpus. ONNX n’est proposé que pour SAME-S. Le périphérique choisi est explicite ; aucun repli après une erreur d’inférence. Pour un nouveau corpus non-SAME-S, un GPU disponible est proposé, sinon CPU ; le choix reste visible et modifiable avant Start Perform.

Le transport lit la dimension des latents et le nombre d’échantillons par hop dans les métadonnées du décodeur. Les adaptateurs sont chargés sur le périphérique demandé sans modifier le `DEVICE` global. Le décodage est sérialisé, en float32/inference mode, retourne le PCM sur CPU et conserve le transport/OLA commun. Chaque fenêtre paire T2–T32 est échauffée et ses sorties contrôlées avant publication des capacités. Les instances sont distinguées par VAE, périphérique et empreintes des poids/configurations.

Stable Audio Open produit **2048 échantillons par latent**. Les corpus historiques déclarent `latent_hz=21.5` ; cette valeur est acceptée, mais la durée PCM vient de la sortie mesurée et non de cet arrondi. SAME-S conserve 4096 échantillons par latent. Les modèles EAR utilisent leur configuration et la sortie mesurée, y compris à 48 kHz.

## Utilisation

Redémarrer le serveur puis recharger la page sans cache. Choisir le corpus, ouvrir Perform, vérifier le VAE et le périphérique dans **Decode with**, puis **Start Perform**. Les adaptateurs réactivés démarrent à **T8** ; SAME-S conserve T2. Les autres fenêtres deviennent disponibles après préparation. Pour EAR, renseigner **VAE weights (EAR .pyt)**. Pour changer de modèle/périphérique, sortir de Perform avant de redémarrer.

## Vérification sur Rack

Corpus fourni : `corpus/Rack_20260428_181107/corpus.npz`, Stable Audio Open, 152 463 frames × 64 dimensions, 44,1 kHz. Les politiques Wander et Reorganized et l’artefact Manual existants sont utilisés.

- [Essai du pipeline complet](multi_vae_rack_pipeline.json), environnement habituel Torch 2.13.0, MPS, haut-parleurs MacBook Pro à volume nul : chargement via `PipelineManager`, préparation des 16 fenêtres, puis 10 secondes à T8 dans chacun des trois modes. **Zéro underrun**, 1243 callbacks, fermeture et libération effectuées.
- [Essai de changements rapides](multi_vae_rack_mps.json), environnement natif séparé Torch 2.7.1, MPS, 180 secondes, changement toutes les 3 secondes, T2–T32 et adaptive : **534 sous-alimentations PCM**, zéro incident de périphérique. Ce stress test ne passe pas le critère temps réel. Son ancien calcul de budget SAME-S a été corrigé explicitement dans le rapport pour le ratio SAO ; les mesures brutes et compteurs sont conservés.
- Les environnements et scénarios diffèrent : le succès à T8 ne démontre pas une cause unique aux sous-alimentations. T8 est un réglage initial prudent ; la capacité numérique à décoder T2–T32 ne garantit pas le temps réel à toutes les fenêtres.
- **265 tests Python réussis, 1 optionnel ignoré** ; les trois suites Web passent. Tests ajoutés : dimensions/rates, cycle de vie, EAR 44/48 kHz simulés, identité de cache, sélection depuis le corpus, choix Web et chemin EAR.

Aucune qualification audio matérielle EAR, CUDA ou écoute comparative supplémentaire n’est revendiquée. Le contrôle Rack est un smoke test de lecture muette ; il ne remplace pas une campagne prolongée ni une écoute. Les qualifications SAME-S antérieures restent propres à leur périmètre.

Reproduire le contrôle Rack dans l’environnement habituel :

```sh
PYTHONPATH=. .venv/bin/python eval_scripts/validate_corpus_decoder.py \
  --corpus corpus/Rack_20260428_181107 --device mps --window 8 \
  --audio-device "Haut-parleurs MacBook Pro" \
  --output docs/multi_vae_rack_pipeline.json
```
