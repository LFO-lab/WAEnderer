# Roadmap — Inférence SAME-S sélectionnable : ONNX CPU / PyTorch GPU

Statut : **phases 0 à 6 implémentées ; phases 5 et 6 terminées le 2 octobre 2026 pour ONNX CPU et MPS sur le scénario matériel testé**. CUDA reste expérimental et non qualifié faute de matériel NVIDIA.

**Audit final :** 252 tests Python réussis, 1 optionnel ignoré ; trois suites Web réussies. Deux campagnes physiques de dix minutes avec politiques de production : zéro underrun par moteur, fenêtres T2 à T32 et trois modes couverts. Parités synthétique et BurntMemory : 128/128 chacune. Écoute utilisateur : très similaire, ONNX légèrement plus bruiteux. Mode autonome hors ligne : lecture et arrêt sans incident. Les échecs initiaux et les corrections restent documentés dans les [preuves de phase 5](docs/DUAL_INFERENCE_PHASE5.md) et le [guide de livraison](docs/DUAL_INFERENCE_RELEASE.md).

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

Le commit `177b0f4` a introduit le décodeur SAME-S ONNX pour le Web. Les évolutions locales présentes lors de la rédaction initiale sont maintenant intégrées au commit `d5c758a70cf14f8924dab2668e607d43da5e6a9d`, référence propre de l’audit de phase 0.

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

- [x] Relever les comportements et tests de référence : 150 tests Python réussis, 1 optionnel ignoré ; trois suites Web réussies.
- [x] Identifier les poids, la révision et les empreintes ; retenir une résolution native explicitement épinglée.
- [x] Comparer les chemins `decode` et `decode_audio(..., chunked=False)` : équivalence du chemin confirmée dans la bibliothèque installée et égalité expérimentale T2 à graine fixée.
- [x] Déterminer la stratégie mémoire : chargement CPU complet, retrait de l’encodeur de l’instance dédiée, puis transfert GPU ; environ 208 Mio de paramètres conservés contre 413 Mio initialement.
- [x] Inventorier les preuves et essayer les 16 fenêtres paires T2 à T32 sur CPU et MPS avec les poids réels.
- [x] Identifier le matériel et enregistrer les versions : M1 Max/MPS disponible ; CUDA absent, machine à provisionner avant qualification. Écart aux dépendances amont documenté ; installation propre à vérifier ultérieurement.

**Critère de sortie :** chemin natif choisi, provenance des poids identifiée, matrice de fenêtres et protocole de comparaison documentés. Aucun gain de performance présumé.

**Résultat : atteint.** Voir le [rapport](docs/DUAL_INFERENCE_PHASE0.md), le [probe reproductible](eval_scripts/audit_dual_inference_phase0.py) et les mesures [CPU](docs/dual_inference_phase0_cpu.json)/[MPS](docs/dual_inference_phase0_mps.json). La lecture intégrée et l’écoute ne sont pas encore validées.

**Découverte à préserver :** le décodage SAME-S et son graphe ONNX sont stochastiques même en évaluation. Les écarts entre moteurs doivent être comparés à la variabilité de chaque moteur ; aucune identité bit à bit n’est attendue. Conserver le bruit natif et le seuil synthétique historique (SNR > 20 dB, RMSE < 0,005) comme détecteur initial de régression, sans en faire un seuil universel de qualité audio.

## Phase 1 — Généraliser la frontière de décodage

- [x] Extraire les types et métadonnées communs des éléments spécifiques aux bundles et sessions ONNX.
- [x] Adapter le décodeur ONNX au contrat sans modifier ses validations de modèle, corpus, dtype, dimensions et valeurs finies.
- [x] Faire dépendre le transport du contrat partagé ; généraliser les noms et messages liés au moteur lorsque nécessaire.
- [x] Conserver les comportements de navigation, provenance des frames, overlap-add, fenêtres adaptatives et transitions.
- [x] Préserver les imports et points d’entrée existants si leur renommage risque de casser les scripts ou tests.

**Critère de sortie :** ONNX CPU fonctionne seul à travers la nouvelle interface, avec les tests existants pertinents toujours satisfaits et sans changement utilisateur.

**Implémentation :**

- [Contrat commun](stable_audio_wanderer/vae/decoder_contract.py) : protocoles structurels `LatentDecoder` et `DecoderInfo`, types `DecoderWindowMetadata`/`DecodedAudioWindow` et exception `DecoderRuntimeError`. Les protocoles décrivent des objets déjà préparés ; les chargeurs et la propriété des instances restent inchangés. Leur cycle de vie configurable demeure en phase 3.
- [Transport commun](stable_audio_wanderer/runtime/decoder_transport.py) : `DecoderTransportController`, utilisé par `bin/serve.py`, sans import du décodeur ONNX concret ni obligation de métadonnées de bundle. Les chemins de provenance ONNX exposés au Web restent inchangés.
- Compatibilité : `runtime.onnx_transport.OnnxTransportController` est un alias de la même classe ; les anciens imports des types et de l’exception depuis `vae.onnx_decoder` et `vae` conservent leur identité. Les signatures des métadonnées ONNX restent inchangées.
- Les fonctions de chargement, validation et décodage ONNX n’ont pas été modifiées, vérifié également par comparaison de leurs AST. Aucun changement du graphe, des poids, de l’aléatoire, des paramètres ORT, des calculs OLA ou de l’ordonnancement à deux workers.
- Les noms internes des threads et messages de transport deviennent neutres. Aucune option GPU ni modification du parcours de sélection dans l’interface n’est introduite.

**Validation :**

- 146 tests réussis et 1 optionnel ignoré dans la suite décodeur/contrat/pipeline/transport/lecteur/OLA/fenêtres/WebSocket/export ; 11 tests OSC/Erae supplémentaires réussis avec accès aux sockets locales, soit **157 réussites et 1 ignoré**.
- Trois suites JavaScript Web réussies (`test_web_onnx_ui.js`, `test_web_wander_ui.js`, `test_web_erae_ui.js`).
- [Nouveaux tests du contrat](tests/test_decoder_contract.py) : compatibilité des imports, import sans ONNX Runtime ni `stable_audio_3`, conformité des deux chargeurs ONNX, transport avec décodeur simulé indépendant d’ONNX, transition, erreur et préservation des champs de provenance.
- Essai du graphe ONNX applicatif réel : 16 fenêtres paires T2 à T32, deux décodages concurrents par fenêtre (32 sorties), formes/dtype/valeurs finies et assemblage overlap-add conformes. Pas de lecture sur le périphérique audio ni d’écoute pendant cette phase.
- Un test existant de transition avec délai court a échoué lors d’un passage avec des essais concurrents, puis réussi seul et dans la suite complète sans charge concurrente. Sa logique temporelle et celle du runtime n’ont pas été modifiées ; cette sensibilité reste à surveiller lors de la qualification temps réel.

**Résultat : atteint pour la frontière logicielle et les vérifications automatisées.** L’écoute et les sessions prolongées restent prévues dans la campagne de phase 5.

## Phase 2 — Ajouter le moteur PyTorch SAME-S

- [x] Charger les poids correspondant à la référence ONNX ; tracer leur identité dans les diagnostics.
- [x] Utiliser la résolution épinglée décidée en phase 0 et vérifier les tenseurs attendus ; valider un environnement installable malgré l’écart actuel aux versions déclarées par `stable-audio-3`.
- [x] Charger sur CPU puis retirer l’encodeur de l’instance dédiée avant transfert GPU ; conserver bottleneck, décodeur et prétransform ainsi que l’appel `decode_audio(..., chunked=False)`.
- [x] Passer explicitement le périphérique au chargeur de décodage sans modifier le `DEVICE` global des autres composants.
- [x] Utiliser le mode évaluation et le mode inférence, avec `float32` comme référence initiale ; différer les optimisations de précision.
- [x] Effectuer les conversions de disposition et transferts nécessaires, puis retourner un PCM CPU conforme au contrat.
- [x] Valider les sorties et échauffer les fenêtres annoncées avant d’autoriser la lecture.
- [x] Mesurer le temps jusqu’à disponibilité réelle du PCM sur CPU, transferts et synchronisation GPU compris.
- [x] Implémenter les chemins CUDA/MPS et les erreurs explicites de périphérique, dépendance, poids ou opération indisponibles ; aucun fallback.
- [x] Vérifier le décodage réel sur MPS et CPU : 128 comparaisons par périphérique, toutes réussies.
- [ ] Qualifier CUDA sur une machine NVIDIA (matériel absent de l’environnement actuel).
- [x] Valider l’écoute des rendus ONNX/natif générés à partir des mêmes latents (retour utilisateur consigné en phase 5).

**Critère de sortie :** décodage natif conforme au contrat sur chaque périphérique annoncé comme pris en charge, avec comparaison numérique et écoute sur les mêmes entrées que l’ONNX.

**Implémentation et preuves :** [SameSTorchDecoder](stable_audio_wanderer/vae/torch_decoder.py), [chargeur strict](stable_audio_wanderer/vae/same_s_weights.py), [tests](tests/test_torch_decoder.py), [rapport et installation](docs/DUAL_INFERENCE_PHASE2.md). Installation propre avec Torch/Torchaudio 2.7.1, 71 dépendances compatibles ; 182 tests Python réussis, 1 optionnel ignoré, trois suites Web réussies. À cette date, la qualification perceptuelle et CUDA restaient ouvertes ; l’écoute a depuis été consignée en phase 5, CUDA reste non qualifié. Le cycle de vie configurable est intégré en phase 3.

## Phase 3 — Intégrer le cycle de vie et la concurrence

- [x] Remplacer le chargement ONNX imposé par une fabrique de décodeurs basée sur la configuration validée.
- [x] Valider le corpus pour chaque démarrage, y compris lorsqu’une instance est réutilisée.
- [x] Définir la réutilisation selon moteur, périphérique et identité des poids ; ne jamais réutiliser une instance incompatible.
- [x] Auditer la libération du VAE de prétraitement et les références conservées afin d’éviter la double occupation mémoire involontaire.
- [x] Commencer avec une politique GPU sérialisée, puis n’autoriser davantage de concurrence que sur preuve de correction et de bénéfice mesuré.
- [x] Préserver la préparation des générations active et candidate pendant les transitions, sans tâches obsolètes publiées ni file de calcul non bornée.
- [x] Définir la séquence de changement : arrêt audio, fin des tâches, remise à zéro du transport, libération de l’ancien moteur, chargement et échauffement du nouveau.
- [x] En cas d’échec, rester dans un état arrêté cohérent avec une erreur visible et une possibilité de réessayer ; ne pas annoncer le nouveau moteur comme actif.
- [x] Distinguer l’arrêt du transport de la sortie de la phase `perform` : ajouter le chemin de reconfiguration nécessaire si la machine d’états actuelle ne le permet pas.

**Critère de sortie :** démarrages, arrêts, échecs et changements répétés ne laissent ni audio obsolète, ni worker actif après libération, ni accumulation de mémoire attribuable aux instances conservées.

**Résultat : implémenté et vérifié le 1er octobre 2026.** 203 tests Python réussis, 1 ignoré ; trois suites Web réussies. Cycles réels ONNX/natif CPU et MPS : une seule instance conservée, aucune après fermeture ; allocations MPS revenues à zéro après remplacement et fermeture. Voir le [rapport de phase 3](docs/DUAL_INFERENCE_PHASE3.md). Ces essais ne remplacent ni l’écoute ni la campagne temps réel de phase 5.

## Phase 4 — Exposer la sélection dans le Web

- [x] Ajouter « ONNX · CPU » et « PyTorch · GPU », avec identification CUDA/MPS lorsque pertinent.
- [x] Distinguer détection du matériel, présence des dépendances/poids et validation effective du décodeur.
- [x] Transmettre la configuration au pipeline et afficher le moteur ainsi que le périphérique réellement actifs.
- [x] Verrouiller le changement pendant la lecture ; rendre le parcours arrêt → sélection → redémarrage explicite.
- [x] Afficher les étapes de chargement/échauffement et les erreurs exploitables, sans figer durablement l’interface.
- [x] Préserver ONNX CPU pour les anciennes configurations et adapter les textes actuellement codés en dur.
- [x] Maintenir les états utilisés par WebSocket, OSC et l’intégration Erae ; aucune commande de sélection distante supplémentaire n’est requise dans cette version.

**Critère de sortie :** l’utilisateur peut identifier, sélectionner et démarrer chaque option disponible sans ambiguïté entre choix demandé et moteur actif.

**Résultat : implémenté et vérifié le 1er octobre 2026.** 206 tests Python réussis, 1 ignoré ; trois suites Web réussies. Voir le [rapport et parcours utilisateur](docs/DUAL_INFERENCE_PHASE4.md). La qualification CUDA, l’écoute et les sessions prolongées restent ouvertes.

## Phase 5 — Valider la qualité audio et le temps réel

### Vérifications automatisées

- [x] Contrat partagé : formes, dtype, valeurs finies, fenêtres et métadonnées, erreurs sur entrées invalides.
- [x] Pipeline : valeur par défaut, sélection, incompatibilités, réutilisation, libération et récupération après échec.
- [x] Transport : transitions, changements de fenêtre, politique de concurrence et absence de publication d’anciens résultats.
- [x] UI : configuration transmise, disponibilité, verrouillage et affichage du moteur effectif.
- [x] Régression : suites existantes du décodeur, pipeline, transport, lecteur, overlap-add et interface ; vérifier également les consommateurs OSC/Erae concernés.
- [x] Séparer les tests sans GPU des essais matériels ; un test simulé ne constitue pas une validation CUDA/MPS.

### Campagne comparative reproductible

Utiliser les mêmes poids, corpus, séquences de latents, graines, fenêtres et réglages de transport. Tester toutes les fenêtres proposées, les modes de navigation, les changements de fenêtre fixes/adaptatifs et les transitions.

Mesurer :

- Erreurs absolues/RMS et métrique relative pertinente entre PCM natif et ONNX, avant overlap-add, puis contrôle du flux assemblé.
- Variabilité des répétitions d’un même moteur, comparée aux écarts entre moteurs ; essais synthétiques sur plusieurs graines selon le protocole de phase 0.
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

**Résultat : atteint sur ONNX CPU et MPS, dans le périmètre mesuré.** [Protocole et résultats](docs/DUAL_INFERENCE_PHASE5.md), [contrôle des preuves](docs/dual_inference_qualification_result.json). Après augmentation de la réserve à deux hops, les deux campagnes physiques passent sans underrun. Les diagnostics initiaux en échec restent disponibles. CUDA ne fait pas partie des périphériques qualifiés.

## Phase 6 — Documenter et livrer

- [x] Documenter les deux parcours, le changement à l’arrêt, les dépendances PyTorch SAME-S et l’accès aux poids natifs.
- [x] Décider du conditionnement des dépendances optionnelles et vérifier qu’ONNX CPU reste utilisable sans installation native SAME-S supplémentaire.
- [x] Mettre à jour les fichiers de dépendances et de verrouillage uniquement selon le mode d’installation retenu.
- [x] Documenter les versions et machines validées, les fenêtres disponibles et les limites CUDA/MPS.
- [x] Ajouter la matrice de vérification à la checklist de release et conserver les rapports de parité/performance.
- [x] Vérifier le démarrage du pipeline ONNX existant sans import natif et le CLI autonome historique.
- [x] Qualifier la lecture du mode autonome historique sur le périphérique audio réel, hors ligne, et son arrêt propre.

**Critère de sortie :** installation reproductible, choix explicite fonctionnel, comportement ONNX existant préservé et preuves de validation jointes.

**Livrables :** [guide de livraison et matrice](docs/DUAL_INFERENCE_RELEASE.md), [checklist](RELEASE_CHECKLIST.md), smoke CPU sans import natif et test de démarrage CLI autonome. **Résultat : atteint pour le périmètre ONNX CPU/MPS testé.** Lecture autonome physique validée ; ordre d’arrêt corrigé pour couper la consommation PCM avant l’attente des workers. Les critères de phase 5 passent dans le périmètre documenté. Le profil natif séparé existant est conservé ; aucun changement du verrou principal.

## Ordre de réalisation et décisions restantes

Ordre recommandé : référence → abstraction avec ONNX seul → moteur natif → cycle de vie → interface → campagne comparative → livraison.

Décisions de phase 0 et suites nécessaires :

1. Charger l’autoencodeur complet sur CPU puis retirer l’encodeur avant transfert ; différer le chargement sélectif des tenseurs.
2. Résoudre config et checkpoint SAME-S à la révision `fbeb3dcf53a326e5682f38e22e7f740202d44232` ; implémenter ensuite les contrôles d’identité et le parcours d’obtention des poids.
3. Valider l’intégration d’abord sur le M1 Max disponible ; qualifier CUDA sur une machine dédiée avant annonce de support.
4. Conserver les 16 fenêtres paires T2 à T32 et le seuil synthétique historique ; compléter par plusieurs graines, corpus réels, écoute et essais temps réel.

Estimation qualitative : effort modéré pour SAME-S avec sélection à l’arrêt. Les incertitudes principales portent sur le chargement natif, la mémoire et les garanties temporelles pendant les transitions. Le chiffrage en jours doit suivre la phase 0.

La bascule sans interruption et la prise en charge d’autres VAE feront l’objet de phases ultérieures distinctes, après validation de ce socle.

## Suite — réactivation des autres VAE dans le Web

La restriction initiale à SAME-S a ensuite été levée pour les adaptateurs existants Stable Audio Open et EAR 44/48 kHz. Il s’agit d’une extension du chemin PyTorch, pas de nouveaux exports ONNX. Voir [l’implémentation, le contrôle Rack et les limites de qualification](docs/MULTI_VAE_WEB.md). La qualification des phases 5/6 ci-dessus reste celle de SAME-S.
