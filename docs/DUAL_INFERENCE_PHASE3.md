# Phase 3 — Sélection et cycle de vie du décodeur

Implémentée le 1er octobre 2026. La sélection est disponible dans les messages du pipeline ; le sélecteur graphique reste en phase 4. ONNX CPU demeure la valeur par défaut. Aucun repli automatique.

## Configuration et propriété

`pipeline_start_perform` reçoit un objet `config` contenant `corpus_dir`, `decoder_window` (2 par défaut), `decoder_backend` (`onnxruntime` ou `pytorch`) et `decoder_device`. ONNX accepte uniquement `cpu`. PyTorch exige un périphérique explicite (`mps`, `cuda:N`, ou `cpu` pour diagnostic) et vérifie sa disponibilité. CUDA reste non qualifié sur matériel.

`decoder_local_files_only` vaut `true` par défaut. Pour autoriser la résolution réseau des poids natifs épinglés, fournir explicitement `false`. Le profil d’installation et les contrôles stricts de provenance restent ceux de la [phase 2](DUAL_INFERENCE_PHASE2.md).

Exemple de configuration native :

```json
{"type":"pipeline_start_perform","config":{"corpus_dir":"/absolute/path/to/corpus","decoder_backend":"pytorch","decoder_device":"mps","decoder_window":2}}
```

Le pipeline possède le décodeur et le contrôleur retourné par son callback de préparation. Le corpus est validé à chaque démarrage, même avec un décodeur réutilisé. L’identité du cache comprend le moteur, le périphérique canonique et les artefacts : empreintes réelles du graphe et des métadonnées pour ONNX ; révision/empreintes épinglées et versions de bibliothèque pour le natif. Une modification impose le remplacement.

Le VAE de prétraitement et la référence conservée dans son résultat sont supprimés avant la préparation du décodeur. Le cache de performance est également libéré avant un nouveau prétraitement ou entraînement. Une seule instance préparée peut rester en cache après sortie de performance.

## Arrêt et reconfiguration

L’arrêt audio habituel conserve la phase `perform` et permet de reprendre. Pour changer de moteur :

1. Envoyer `{"type":"pipeline_stop_perform"}`.
2. Attendre la phase `idle`.
3. Envoyer `pipeline_start_perform` avec la nouvelle configuration.

La sortie de performance détache WebSocket, visualisation et OSC/Erae, interdit tout redémarrage d’un ancien contrôleur, coupe l’audio, attend les calculs, puis vide les buffers et ferme le lecteur. Le démarrage suivant libère le moteur précédent si l’identité change, charge le nouveau et prépare le transport. Le natif échauffe toutes ses fenêtres ; ONNX conserve son comportement applicatif de préparation différée existant.

Les états `preparing` et `stopping` annoncent `decoder: null`. Seule la réussite complète annonce `perform` avec le moteur/périphérique effectifs. Un échec nettoie les ressources et revient à `idle` avec une erreur ; si le nettoyage échoue, l’état `error` conserve la propriété du modèle pour éviter une libération sous un worker. `pipeline_stop_perform` permet de retenter le nettoyage. La fermeture du serveur draine les producteurs avant la libération finale.

Les commandes du pipeline sont sérialisées et exécutées hors de la boucle WebSocket, qui continue à diffuser les messages. Le transport conserve ses générations active/candidate et sa limite de deux calculs en vol ; le verrou du décodeur natif sérialise leurs inférences. Les calculs retirés restent comptés dans cette limite. Aucune modification des calculs overlap-add.

## Vérifications

- **203 tests Python réussis, 1 optionnel ignoré**, couvrant décodeurs, contrat, pipeline, cycle de vie, transport, lecteur, fenêtres, overlap-add, export, WebSocket et OSC/Erae. La régression complète compte 202 réussites ; le test supplémentaire de coupure audio avant drainage a ensuite réussi avec les 44 autres tests de cycle de vie/transport. Les tests de sockets nécessitent l’accès aux ports locaux.
- Trois suites JavaScript réussies : `test_web_onnx_ui.js`, `test_web_wander_ui.js`, `test_web_erae_ui.js`.
- Tests de calcul bloqué pendant l’arrêt, refus des commandes Start tardives, corpus invalide avec cache, changement d’identité, erreur de chargement/préparation, échec du drainage et reprise. Douze changements simulés vérifient les références faibles : une instance conservée, aucune après fermeture.
- Le test temporel existant de longue transition a échoué lors du premier passage puis réussi dans la régression complète. Sa sensibilité aux délais reste à surveiller lors de la phase 5.
- Essais réels avec Torch 2.7.1 : séquence ONNX → natif → ONNX → natif sur CPU et MPS, chargement/échauffement natif et décodage T2 réel à chaque préparation. Une seule instance vivante entre les changements, aucune après fermeture.
- MPS : **222 789 376 octets** alloués après chacun des deux chargements natifs, **0** après retour ONNX et après fermeture. Ce compteur décrit les allocations de tenseurs MPS, pas la mémoire totale du processus.

Reproduction depuis la racine du projet, dans l’environnement natif documenté en phase 2 :

```sh
PYTHONPATH=. python eval_scripts/validate_decoder_lifecycle.py --device cpu --output docs/dual_inference_phase3_cpu.json
PYTHONPATH=. python eval_scripts/validate_decoder_lifecycle.py --device mps --output docs/dual_inference_phase3_mps.json
```

Mesures conservées : [CPU](dual_inference_phase3_cpu.json), [MPS](dual_inference_phase3_mps.json). Le probe réel utilise un contrôleur minimal sans périphérique audio ; le drainage du véritable transport est testé séparément avec un décodeur bloquant simulé. Ces résultats ne constituent pas une qualification de lecture prolongée, d’écoute, de latence temps réel ou de CUDA.
