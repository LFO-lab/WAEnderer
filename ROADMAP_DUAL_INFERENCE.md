# Roadmap — Inférence SAME-S sélectionnable : ONNX CPU / PyTorch GPU

Statut : planification uniquement. Aucune implémentation du double moteur à ce stade.

## Objectif et périmètre

Permettre de choisir le moteur de décodage audio dans l’interface Web, tout en conservant un transport commun : navigation, assemblage des latents, overlap-add, buffers, transitions et synchronisation visuelle.

Première version :

- SAME-S uniquement, avec les mêmes poids et la même révision source pour les deux moteurs.
- ONNX Runtime sur CPU reste le choix par défaut et le comportement des configurations existantes.
- PyTorch sur CUDA ou MPS devient une option explicite, disponible seulement après validation du périphérique concerné.
- Un seul moteur de décodage est actif pour la performance. « Deux méthodes en parallèle » désigne leur coexistence dans l’application.
- Sélection avant le démarrage de la performance ; changement après arrêt et drainage des tâches de décodage.
- Aucun repli automatique silencieux vers un autre moteur ou périphérique.

Hors périmètre initial : bascule sans interruption pendant la lecture, décodage simultané CPU/GPU, ONNX sur GPU, généralisation à d’autres VAE et remplacement du mode autonome historique.

## État de départ vérifié

Le commit `177b0f4` a introduit le décodeur SAME-S ONNX pour le Web. Le code local contient également des évolutions non commitées ; cette roadmap s’appuie sur cet état de travail et devra être rapprochée de la branche retenue avant implémentation.

| Élément | Situation actuelle | Conséquence |
| --- | --- | --- |
| `runtime/pipeline_server.py` | Charge et conserve un décodeur ONNX applicatif | Ajouter une sélection et un cycle de vie par configuration |
| `bin/serve.py` | Configure le transport Web pour SAME-S | Conserver la contrainte SAME-S, retirer la dépendance au moteur ONNX |
| `runtime/onnx_transport.py` | Appelle un décodeur séparé du lecteur audio | Généraliser cette frontière sans reconstruire le transport |
| `vae/onnx_decoder.py` | Fournit audio, métadonnées de fenêtre, capacités et mesures | Extraire un contrat partagé en préservant les validations ONNX |
| `bin/perform.py`, `vae/decoder.py` | Le chemin PyTorch historique existe toujours | Référence utile, mais pas un transport à recopier dans le Web |
| `vae/adapters/same_s.py` | Charge l’autoencodeur natif sur le périphérique global | Définir un chargement et un périphérique propres au décodage |
| `bin/export_vst_bundle.py` | Exporte `decode_audio(latents, chunked=False)` | Vérifier l’équivalence avec le chemin natif, qui appelle actuellement `decode` |
| `web/pipeline.js`, `web/index.html` | Présentent un décodeur ONNX CPU imposé | Ajouter sélection, disponibilité et état réellement actif |

## Architecture cible

```text
Corpus SAME-S → navigation → fenêtres de latents bruts
                                     ↓
                           contrat de décodage commun
                            ↙                     ↘
                    ONNX Runtime CPU        PyTorch CUDA / MPS
                            ↘                     ↙
                         PCM + métadonnées + mesures
                                     ↓
                        overlap-add → buffers → audio
```

Le contrat commun doit exposer :

- L’identité du VAE et du modèle, le moteur et le périphérique effectifs.
- Les fenêtres prises en charge, la fenêtre par défaut et leurs métadonnées temporelles.
- Une entrée NumPy `float32` de forme `[T, 256]`, en latents bruts dénormalisés.
- Une sortie PCM NumPy `float32` de forme `[samples, channels]`, les métadonnées de fenêtre et la durée du décodage.
- Les opérations de préparation et de libération, ainsi qu’une politique explicite de concurrence.

Les noms exacts des classes et champs restent à fixer. Configuration envisagée : `decoder_backend` et `decoder_device`, séparés du choix du VAE. En leur absence, appliquer ONNX CPU.

## Phase 0 — Établir la référence et lever les incertitudes

- [ ] Relever les comportements et tests de référence du transport actuel, y compris les modifications locales à conserver.
- [ ] Identifier précisément les poids et la révision utilisés par l’ONNX livré ; choisir une résolution identique des poids natifs.
- [ ] Comparer les chemins `decode` et `decode_audio(..., chunked=False)` : normalisation, découpage, padding et forme de sortie.
- [ ] Déterminer si le décodeur natif peut être chargé seul. Sinon, documenter la mémoire de l’autoencodeur complet et le compromis pour la première version.
- [ ] Inventorier les fenêtres annoncées et les preuves de validation disponibles. Ne pas assimiler plage déclarée et couverture des rapports de parité existants.
- [ ] Définir les machines CUDA/MPS de validation et les versions de dépendances reproductibles.

**Critère de sortie :** chemin natif choisi, provenance des poids identifiée, matrice de fenêtres et protocole de comparaison documentés. Aucun gain de performance présumé.

## Phase 1 — Généraliser la frontière de décodage

- [ ] Extraire les types et métadonnées communs des éléments spécifiques aux bundles et sessions ONNX.
- [ ] Adapter le décodeur ONNX au contrat sans modifier ses validations de modèle, corpus, dtype, dimensions et valeurs finies.
- [ ] Faire dépendre le transport du contrat partagé ; généraliser les noms et messages liés au moteur lorsque nécessaire.
- [ ] Conserver les comportements de navigation, provenance des frames, overlap-add, fenêtres adaptatives et transitions.
- [ ] Préserver les imports et points d’entrée existants si leur renommage risque de casser les scripts ou tests.

**Critère de sortie :** ONNX CPU fonctionne seul à travers la nouvelle interface, avec les tests existants pertinents toujours satisfaits et sans changement utilisateur.

## Phase 2 — Ajouter le moteur PyTorch SAME-S

- [ ] Charger les poids correspondant à la référence ONNX ; tracer leur identité dans les diagnostics.
- [ ] Passer explicitement le périphérique au chargeur de décodage sans modifier le `DEVICE` global des autres composants.
- [ ] Utiliser le mode évaluation et le mode inférence, avec `float32` comme référence initiale ; différer les optimisations de précision.
- [ ] Effectuer les conversions de disposition et transferts nécessaires, puis retourner un PCM CPU conforme au contrat.
- [ ] Valider les sorties et échauffer les fenêtres annoncées avant d’autoriser la lecture.
- [ ] Mesurer le temps jusqu’à disponibilité réelle du PCM sur CPU, transferts et synchronisation GPU compris.
- [ ] Vérifier séparément CUDA et MPS ; produire une erreur explicite en cas de périphérique, dépendance, poids ou opération indisponibles.

**Critère de sortie :** décodage natif conforme au contrat sur chaque périphérique annoncé comme pris en charge, avec comparaison numérique et écoute sur les mêmes entrées que l’ONNX.

## Phase 3 — Intégrer le cycle de vie et la concurrence

- [ ] Remplacer le chargement ONNX imposé par une fabrique de décodeurs basée sur la configuration validée.
- [ ] Valider le corpus pour chaque démarrage, y compris lorsqu’une instance est réutilisée.
- [ ] Définir la réutilisation selon moteur, périphérique et identité des poids ; ne jamais réutiliser une instance incompatible.
- [ ] Auditer la libération du VAE de prétraitement et les références conservées afin d’éviter la double occupation mémoire involontaire.
- [ ] Commencer avec une politique GPU sérialisée, puis n’autoriser davantage de concurrence que sur preuve de correction et de bénéfice mesuré.
- [ ] Préserver la préparation des générations active et candidate pendant les transitions, sans tâches obsolètes publiées ni file de calcul non bornée.
- [ ] Définir la séquence de changement : arrêt audio, fin des tâches, remise à zéro du transport, libération de l’ancien moteur, chargement et échauffement du nouveau.
- [ ] En cas d’échec, rester dans un état arrêté cohérent avec une erreur visible et une possibilité de réessayer ; ne pas annoncer le nouveau moteur comme actif.
- [ ] Distinguer l’arrêt du transport de la sortie de la phase `perform` : ajouter le chemin de reconfiguration nécessaire si la machine d’états actuelle ne le permet pas.

**Critère de sortie :** démarrages, arrêts, échecs et changements répétés ne laissent ni audio obsolète, ni worker actif après libération, ni accumulation de mémoire attribuable aux instances conservées.

## Phase 4 — Exposer la sélection dans le Web

- [ ] Ajouter « ONNX · CPU » et « PyTorch · GPU », avec identification CUDA/MPS lorsque pertinent.
- [ ] Distinguer détection du matériel, présence des dépendances/poids et validation effective du décodeur.
- [ ] Transmettre la configuration au pipeline et afficher le moteur ainsi que le périphérique réellement actifs.
- [ ] Verrouiller le changement pendant la lecture ; rendre le parcours arrêt → sélection → redémarrage explicite.
- [ ] Afficher les étapes de chargement/échauffement et les erreurs exploitables, sans figer durablement l’interface.
- [ ] Préserver ONNX CPU pour les anciennes configurations et adapter les textes actuellement codés en dur.
- [ ] Maintenir les états utilisés par WebSocket, OSC et l’intégration Erae ; aucune commande de sélection distante supplémentaire n’est requise dans cette version.

**Critère de sortie :** l’utilisateur peut identifier, sélectionner et démarrer chaque option disponible sans ambiguïté entre choix demandé et moteur actif.

## Phase 5 — Valider la qualité audio et le temps réel

### Vérifications automatisées

- [ ] Contrat partagé : formes, dtype, valeurs finies, fenêtres et métadonnées, erreurs sur entrées invalides.
- [ ] Pipeline : valeur par défaut, sélection, incompatibilités, réutilisation, libération et récupération après échec.
- [ ] Transport : transitions, changements de fenêtre, politique de concurrence et absence de publication d’anciens résultats.
- [ ] UI : configuration transmise, disponibilité, verrouillage et affichage du moteur effectif.
- [ ] Régression : suites existantes du décodeur, pipeline, transport, lecteur, overlap-add et interface ; vérifier également les consommateurs OSC/Erae concernés.
- [ ] Séparer les tests sans GPU des essais matériels ; un test simulé ne constitue pas une validation CUDA/MPS.

### Campagne comparative reproductible

Utiliser les mêmes poids, corpus, séquences de latents, graines, fenêtres et réglages de transport. Tester toutes les fenêtres proposées, les modes de navigation, les changements de fenêtre fixes/adaptatifs et les transitions.

Mesurer :

- Erreurs absolues/RMS et métrique relative pertinente entre PCM natif et ONNX, avant overlap-add, puis contrôle du flux assemblé.
- Écoute des attaques, queues et raccords ; absence de clics ou de modification systématique du niveau.
- Latences médiane, p95 et p99 du décodage complet, temps de préparation et latence de réaction aux commandes.
- Interruptions audio (`underruns`), charge CPU, mémoire GPU et comportement pendant les transitions.
- Chargement à froid, fonctionnement stabilisé pendant au moins 10 minutes et cycles répétés de changement de moteur.

Pour SAME-S, un hop de `H` latents donne un budget audio de `H × 4096 / 44100` secondes : à T2 avec un hop de 1, environ 92,9 ms. Évaluer le coût total de production du hop et la contention pendant les transitions, pas seulement le temps du noyau GPU.

**Critères de sortie :**

- Fixer et documenter les tolérances numériques à partir de la référence existante avant de conclure la campagne ; ne pas exiger une identité bit à bit.
- Aucune régression audible identifiée sur le corpus de validation.
- Aucun underrun dans les scénarios déclarés pris en charge pendant la campagne stabilisée ; marge de calcul et conditions matérielles consignées.
- Un tableau de résultats par périphérique et fenêtre explique les limites observées. Un GPU plus lent que le CPU n’est pas présenté comme une amélioration de performance.

## Phase 6 — Documenter et livrer

- [ ] Documenter les deux parcours, le changement à l’arrêt, les dépendances PyTorch SAME-S et l’accès aux poids natifs.
- [ ] Décider du conditionnement des dépendances optionnelles et vérifier qu’ONNX CPU reste utilisable sans installation native SAME-S supplémentaire.
- [ ] Mettre à jour les fichiers de dépendances et de verrouillage uniquement selon le mode d’installation retenu.
- [ ] Documenter les versions et machines validées, les fenêtres disponibles et les limites CUDA/MPS.
- [ ] Ajouter la matrice de vérification à la checklist de release et conserver les rapports de parité/performance.
- [ ] Vérifier le mode autonome historique et le démarrage d’une installation CPU existante.

**Critère de sortie :** installation reproductible, choix explicite fonctionnel, comportement ONNX existant préservé et preuves de validation jointes.

## Ordre de réalisation et décisions restantes

Ordre recommandé : référence → abstraction avec ONNX seul → moteur natif → cycle de vie → interface → campagne comparative → livraison.

Décisions à trancher pendant la phase 0 :

1. Chargement du décodeur seul ou de l’autoencodeur complet pour la première version.
2. Distribution/résolution des poids natifs et vérification de leur correspondance avec l’ONNX livré.
3. Périphériques effectivement disponibles pour valider CUDA et MPS ; ne pas annoncer ceux qui n’ont pas été vérifiés.
4. Liste commune de fenêtres validées et seuils de parité numérique.

Estimation qualitative : effort modéré pour SAME-S avec sélection à l’arrêt. Les incertitudes principales portent sur le chargement natif, la mémoire et les garanties temporelles pendant les transitions. Le chiffrage en jours doit suivre la phase 0.

La bascule sans interruption et la prise en charge d’autres VAE feront l’objet de phases ultérieures distinctes, après validation de ce socle.
