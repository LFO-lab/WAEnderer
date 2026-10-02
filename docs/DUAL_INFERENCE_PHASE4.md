# Phase 4 — Sélection Web du moteur

Implémentée le 1er octobre 2026. Dans **Perform → Decode with**, choisir **ONNX · CPU** (valeur par défaut) ou **PyTorch · GPU · MPS/CUDA:N**. Chaque GPU CUDA détecté possède son entrée. Le périphérique demandé est transmis explicitement ; aucun repli automatique.

## Disponibilité et validation

**Refresh availability** relit la détection du matériel, la présence des dépendances et des fichiers de poids. Cette découverte est locale : elle ne télécharge aucun fichier, ne charge aucun modèle et ne lance aucune inférence. Les options manquant de matériel, de dépendances ou de poids sont désactivées avec une explication. La configuration native et le cache épinglé sont documentés dans le [guide de phase 2](DUAL_INFERENCE_PHASE2.md).

La présence des fichiers ne prouve pas leur intégrité ni la compatibilité des bibliothèques. **Start Perform** effectue les contrôles stricts et l’échauffement natif. L’état « Active » provient du serveur après la réussite de cette préparation ; un choix dans le menu n’est jamais présenté comme un moteur actif. CUDA porte la mention de qualification matérielle encore à effectuer.

Le protocole de découverte utilise `pipeline_list_decoders` / `pipeline_decoder_list`, avec `hardware`, `dependencies`, `weights`, `selectable` et `validated: false` pour chaque option. L’état effectif reste celui des messages `pipeline_state` / `pipeline_phase_change` et du transport. Les anciennes configurations sans sélection conservent ONNX CPU.

## Changer de moteur

1. Cliquer **Stop Perform to change decoder**. Cette action coupe la lecture et attend les calculs en cours.
2. Attendre le retour à l’état arrêté ; choisir le moteur souhaité.
3. Cliquer **Start Perform**, puis démarrer le décodage audio.

**Stop Decode** reste une pause du transport ; elle ne déverrouille pas la sélection du moteur. Le menu est verrouillé pendant toute la performance, le chargement et le drainage. Les doubles clics de démarrage sont bloqués dès l’envoi de la commande.

Le panneau Perform reste visible pendant `preparing`, `stopping` et `error`. Les messages expliquent le chargement/échauffement et le drainage. En cas d’échec, le message serveur reste visible et un nouveau démarrage est possible après nettoyage ; en cas d’échec du nettoyage, le bouton d’arrêt permet de le retenter. Une déconnexion masque l’ancien état actif et verrouille les commandes ; à la reconnexion, le moteur et le périphérique effectifs sont resynchronisés depuis le serveur.

Le contrôle historique de fenêtre manuelle est désactivé pour les deux moteurs du transport commun. La sélection des fenêtres, les contrôles adaptatifs, la navigation, OSC et Erae conservent leur protocole.

## Vérification

- **206 tests Python réussis, 1 optionnel ignoré** : découverte simulée avec matériel présent/absent et plusieurs GPU, distinction présence/validation, protocole pipeline, cycle de vie, décodeurs, transport, lecteur, fenêtres, WebSocket et OSC/Erae.
- **Trois suites JavaScript réussies**. La suite de workflow teste les valeurs CPU par défaut, l’envoi explicite MPS, les options indisponibles, le verrouillage, les doubles clics, l’arrêt/reconfiguration, les erreurs et la reconnexion. Les suites Wander/Erae restent satisfaites.
- Aucun modèle ni poids modifié. Les essais numériques CPU/MPS et de libération réelle restent ceux des phases 2/3. Cette phase n’ajoute pas de qualification CUDA, d’écoute ni d’essai audio prolongé ; ils restent prévus en phase 5.
