# Phase 5 — outillage livré, qualification à réaliser ensemble

Implémentation du 4 octobre 2026. Les outils couvrent les quatre VAEs ; la qualification physique et l’écoute GPU restent à réaliser. À la demande de l'utilisateur, cette livraison privilégie le code explicable, quelques vérifications ciblées et des essais manuels partagés. Aucun essai audio physique de dix minutes n'a été lancé pour cette livraison.

## Comment fonctionne le code

Le parcours numérique est court : `corpus_decoder_spec` lit les exigences du corpus → `select_decoder` vérifie le modèle et ses sources → `create_decoder` charge le moteur → `probe` donne exactement les mêmes latents dénormalisés aux deux moteurs → `assemble` applique `StreamingFullOverlapAdd` comme la production.

- `eval_scripts/multi_vae_validation_common.py` contient les petits helpers : arguments de sélection, empreintes, contexte des rapports, contrôle des PCM et calcul des erreurs. Il ne charge aucun modèle à l'import.
- `eval_scripts/compare_corpus_decoders.py` généralise le script SAME-S existant. `probe` réalise deux appels par moteur sans réinitialiser le bruit du modèle. `run` choisit quatre positions par fichier et les fenêtres T2–T32 ; `--synthetic` ajoute les huit graines du protocole. `assemble` écrit les paires WAV sans normalisation et sans vider la queue OLA en fin d'extrait. Les échecs numériques restent dans le JSON.
- `eval_scripts/benchmark_dual_inference.py` garde le transport et le callback existants. Il accepte maintenant les sources EAR et les magasins d'artifacts. Le rapport lie la sélection au corpus, au protocole et à l'artifact, compte les samples réellement rendus et sépare le temps actif du temps de préparation/nettoyage. Il ajoute la couverture des modes/fenêtres, le buffer, les paramètres du stream et le RSS en octets.
- `eval_scripts/validate_multi_vae_lifecycle.py` passe ONNX → PyTorch CPU → GPU explicitement demandé → ONNX à travers `PipelineManager`, avec Stop entre chaque sélection. Un processus neuf redécode ensuite le même artifact avec réseau et bibliothèques natives bloqués. Ce contrôle d'ownership n'ouvre pas de périphérique audio ; il ne remplace pas le test physique ou le test des préférences dans le navigateur.
- `stable_audio_wanderer/qualification.py:evaluate_multi_vae_campaign` rassemble les preuves et recalcule les gates numériques. `bin/check_multi_vae_qualification.py` affiche la matrice. L'ancien vérificateur SAME-S reste disponible pour ses rapports historiques.

Le JSON `multi_vae_phase5_protocol.json` fige les tolérances existantes. Son hash figure dans chaque rapport. Un changement de tolérance produit une campagne différente, sans rendre valide rétroactivement un échec. Ce protocole initial conserve la tolérance GPU Stable Audio Open en attente. Après caractérisation native réussie, `multi_vae_phase5_stable_audio_open_gpu_protocol.json` fige séparément ses limites GPU : max absolu 2e-4 et RMSE 2e-5, identiques aux limites CPU. Le manifeste peut choisir un `protocol` par claim ; les preuves CPU antérieures conservent donc leur digest original.

`eval_scripts/characterize_native_gpu.py` compare CPU et GPU natifs avant tout décodage ONNX/GPU. Il utilise les limites CPU comme candidate fixée avant le run, garde deux appels par moteur et libère la référence CPU avant de charger le GPU. Pour Stable Audio Open, les 32 comparaisons (16 fenêtres × réel/synthétique) ont passé, avec max absolu 3.54e-6 et RMSE maximale 3.06e-7. Le nouveau protocole contient le hash de cette preuve native ; le checker contrôle son modèle, son corpus, son artifact et ses limites avant toute qualification GPU complète. Son chemin est relatif à la racine du projet Python.

## Un premier essai numérique

Pour déboguer rapidement, ajouter `--quick` à la commande ci-dessous et utiliser un autre nom de rapport ainsi qu'un autre dossier audio. Ce mode conserve les tolérances et les deux appels par moteur, mais réduit la couverture à T2/T8/T32, une position réelle, une graine synthétique si demandée et un seul extrait OLA de 32 frames. Avec `--synthetic`, cela représente environ 52 décodages plutôt que 1 652 pour la campagne complète EAR 48k actuelle. Le rapport indique `validation_scope: quick_diagnostic` ; il ne remplace pas la qualification. À 48 kHz EAR, le rendu OLA court dure environ 0,56 seconde et sert surtout au diagnostic numérique ; conserver les rendus complets pour une écoute utile.

Commandes depuis la racine du projet Python. Exemple EAR 48 kHz utilisant le corpus déjà préparé :

```sh
.venv-ear/bin/python -m eval_scripts.compare_corpus_decoders \
  --corpus corpus/phase3_ear_vae_48k_20261003_102119 \
  --weights /Users/dthibault/Documents/GitHub/EAR_VAE/pretrained_weight/ear_vae_v2_48k.pyt \
  --repo /Users/dthibault/Documents/GitHub/EAR_VAE \
  --device cpu --synthetic \
  --output build/phase5/ear_vae_48k_cpu_numerical.json \
  --audio-dir eval_out/multi_vae_phase5/ear_vae_48k_cpu
```

Sans `--synthetic`, le script permet un essai plus court sur le corpus et les WAV. Le vérificateur de campagne demande néanmoins les probes synthétiques avant qualification complète. Pour comparer un GPU natif, relancer avec `--device mps` et un autre nom de rapport/audio. Les devices `mps` et `mps:0` sont normalisés dans les rapports.

| Modèle | Corpus local connu | Environnement natif |
| --- | --- | --- |
| SAME-S | `corpus/BurntMemory_20260915_211501` | Environnement SAME-S natif épinglé, celui utilisé précédemment ; vérifier son chemin local |
| Stable Audio Open | `corpus/Rack_20260428_181107` | Environnement disposant de ses sources natives mises en cache |
| EAR 44k | `corpus/TC_Harpsichord_20261002_172959` | `.venv-ear`, checkpoint `ear_vae_44k.pyt` |
| EAR 48k | `corpus/phase3_ear_vae_48k_20261003_102119` | `.venv-ear`, checkpoint `ear_vae_v2_48k.pyt` |

Le corpus sélectionne le modèle. Les sources natives sont liées à l'artifact choisi ; une contradiction échoue. Les anciens corpora sans provenance exacte restent indiqués comme tels. Les extraits EAR actuels couvrent le clavecin : cela ne représente pas tous les contenus audio.

## Qualification numérique compacte

`--compact --synthetic` couvre toutes les fenêtres T2–T32, une position réelle, une graine par fenêtre pour les modèles déterministes et huit graines pour SAME-S, puis un extrait OLA de 32 frames. Cela représente 156 décodages pour les modèles déterministes et 604 pour SAME-S, toujours avec deux appels par moteur et les mêmes tolérances. Le manifeste indique `qualification_profile: "compact"`. Le rapport complet EAR 48k CPU déjà fourni satisfait ce profil et n’est pas relancé. `--quick` reste uniquement un diagnostic.

Pour EAR 44k, ajouter `--fixture build/phase3/ear_vae_44k_probes.npz` : ses latents proviennent d’un encodeur dont la source est vérifiée contre l’artifact. Le rapport conserve le hash du corpus de navigation et enregistre séparément le hash/source de la fixture. Cela ne certifie pas la provenance historique du corpus clavecin.

Si les nouveaux rendus diffèrent des WAV déjà écoutés, le claim utilise `listening_report` pour désigner le rapport d’origine. Les avis existants restent attachés aux fichiers exacts ; ils ne sont pas transférés à de nouveaux rendus.

## Mesurer le transport à ton rythme

Commencer par l'interface : sélectionner le corpus, choisir ONNX CPU, démarrer, changer Wander/Manual/Reorganized et les fenêtres, arrêter, sélectionner PyTorch CPU puis MPS et redémarrer. Vérifier aussi le retour au choix explicite après un refresh/reconnect. Si un comportement étonne, conserver le corpus, le moteur, la fenêtre et la commande qui le déclenche : ce sont les entrées utiles pour déboguer ensemble.

Le banc peut ensuite enregistrer les mêmes familles de transitions. Diagnostic court, sans ouvrir le périphérique :

```sh
.venv/bin/python -m eval_scripts.benchmark_dual_inference \
  --backend onnxruntime --device cpu \
  --corpus corpus/phase3_ear_vae_48k_20261003_102119 \
  --seconds 20 --output build/phase5/ear_48k_diagnostic.json
```

Pour une mesure physique de qualification, utiliser `--navigation production --audio-device "Haut-parleurs MacBook Pro" --seconds 620 --require-zero-underruns`. Le nom doit correspondre au périphérique local. La sortie physique est muette : elle mesure le callback, pas l'écoute. Les 620 secondes laissent une marge pour les arrêts/repréparations ; seul le total de samples attribués à une génération PCM stable doit dépasser 600 secondes. Si ce total reste insuffisant, le rapport échoue sans effacer les mesures. Faire les runs un par un. Une session contient les trois modes, les 16 fenêtres et l'adaptatif ; elle ne prouve pas dix minutes indépendantes sur chaque fenêtre.

Pour un moteur EAR natif, utiliser `.venv-ear`, `--backend pytorch --device cpu` ou `mps`, et les mêmes `--weights`/`--repo` que ci-dessus. Les CPU restent utilisables si le rapport relève des underruns ; on distingue fonctionnalité et temps réel.

Les réactions de commande mesurent l'activation de génération PCM, pas la latence au haut-parleur. Les samples par scénario sont attribués à la génération PCM effectivement rendue. Les transitions mélangées, les générations non attribuées et les callbacks avec underrun sont exclus. `callback_audio_seconds` conserve le temps total du callback ; `active_audio_seconds` mesure uniquement le temps attribué. Le coût de préparation inclut le warm-up du constructeur natif ; le RSS couvre le processus entier, navigation comprise. Les p99 ont leur nombre d'observations ; quelques appels ne constituent pas une estimation robuste.

Pour vérifier les changements de moteur et le reload offline sans audio :

```sh
.venv-ear/bin/python -m eval_scripts.validate_multi_vae_lifecycle \
  --corpus corpus/phase3_ear_vae_48k_20261003_102119 \
  --weights /Users/dthibault/Documents/GitHub/EAR_VAE/pretrained_weight/ear_vae_v2_48k.pyt \
  --repo /Users/dthibault/Documents/GitHub/EAR_VAE --gpu mps \
  --output build/phase5/ear_vae_48k_lifecycle.json
```

Pour restreindre une session, ajouter par exemple `--windows 8 16 --modes manual --no-adaptive`. Le claim correspondant déclare `realtime_profile: {"windows": [8, 16], "modes": ["manual"], "adaptive": false}`. Sans ces options, le profil inclut toutes les fenêtres, les trois modes et l’adaptatif. L’adaptatif utilise toutes les fenêtres supportées entre les bornes min/max du profil et commence par Stop/Start. Les 600 secondes portent sur le profil combiné, avec au moins une seconde stable par membre, et non sur chaque réglage individuellement.

## Écoute et matrice finale

Copier `docs/multi_vae_phase5_campaign.example.json` dans le dossier des rapports. Copier le protocole à côté, ajuster les chemins et les devices mesurés. L'exemple commence sans claim temps réel : mettre `claim_realtime: true` seulement pour les scénarios que l'on souhaite réellement qualifier. Les chemins des rapports sont relatifs au manifeste.

Chaque comparaison écrit `listening_pairs[].files` avec les noms et hashes des deux WAV. Après écoute, ajouter un enregistrement à `claims[].listening` :

```json
{
  "file_index": 0,
  "files": {"onnx": {"path": "copier la valeur du rapport", "sha256": "copier le hash"}, "native": {"path": "copier la valeur du rapport", "sha256": "copier le hash"}},
  "reviewer": "ton nom",
  "date": "date réelle d'écoute",
  "observation": "ce que tu entends",
  "new_engine_regression": false
}
```

Laisser `listening: []` tant que l'écoute n'a pas eu lieu. Une observation SAME-S ne peut pas valider EAR ou Stable Audio Open. Un verdict positif concerne les paires effectivement écoutées, pas toute la musique possible.

```sh
.venv/bin/python -m bin.check_multi_vae_qualification \
  build/phase5/campaign.json --output build/phase5/qualification.json
```

Le checker conserve les résultats indépendants : fonctionnel/numérique, temps réel, lifecycle et écoute. Un rapport absent est pending/not tested. Un échec physique conserve le résultat numérique ; un CPU sans claim temps réel peut terminer sa qualification fonctionnelle avec une limite de performance explicite. La campagne complète exige les deux CPU pour chaque VAE et les preuves des GPU inclus dans le manifeste. CUDA reste non testé sur cette machine.

La matrice réelle en cours est dans `build/phase5/qualification.json`. Les diagnostics courts réussis apparaissent comme `diagnostic_passed`, avec fonctionnement `observed` ; ils ne sont ni des échecs numériques ni une qualification complète. Le lifecycle reste lié aux sources et artifacts sans dépendre d'un changement des seules tolérances numériques. Une écoute enregistrée reste également indépendante de la couverture numérique complète, tout en désignant exactement ses fichiers.

## Vérifications effectuées pour cette livraison

Les 29 tests ciblés passent et couvrent les gates historiques, les profils compacts/restreints, la provenance EAR, les rapports d’écoute séparés et l’attribution PCM au passage d’une génération à l’autre. La suite complète n’est pas relancée, car les interfaces de production restent inchangées.

Les quatre modèles disposent de diagnostics CPU/MPS réussis et de changements arrêtés ONNX CPU → PyTorch CPU → MPS → ONNX CPU, suivis d’un reload dans un processus neuf hors réseau. La qualification physique SAME-S historique reste séparée et passe son ancien checker.

Le retour utilisateur « Pas de dégradation audible » est enregistré pour une paire ONNX CPU/PyTorch CPU par VAE dans [la fiche d’écoute](multi_vae_phase5_listening.json), avec les hashes exacts. Durées : EAR 48k 2,5 s, Stable Audio Open 1,3 s, EAR 44k 0,65 s, SAME-S 2,6 s. Le verdict concerne uniquement ces extraits CPU. Les rendus GPU restent à écouter.

Le rapport complet EAR 48k CPU fourni par l’utilisateur est conservé : 192 probes réelles, 128 synthétiques et trois paires OLA réussies. Les trois qualifications compactes CPU passent : SAME-S 16 probes réelles + 128 synthétiques + une OLA ; Stable Audio Open et EAR 44k chacun 16 réelles + 16 synthétiques + une OLA. Les huit claims CPU passent avec leurs lifecycle et écoutes. La livraison ajoute ces qualifications compactes pour les autres CPU ; les rapports et leurs hashes sont recensés dans [le relevé des contrôles](multi_vae_phase5_checks.json). La fixture EAR 44k fournit les entrées numériques à provenance vérifiée ; l’écoute existante porte toujours sur l’ancien extrait du corpus.

La matrice locale est `build/phase5/qualification.json` ; `build/phase5/qualification_cpu.json` isole les huit claims CPU réussis sans claim temps réel. Les diagnostics GPU courts restent `diagnostic_passed`, avec fonctionnement `observed`, jusqu’à leur qualification numérique complète. Les lifecycle restent indépendants du protocole numérique. Aucune nouvelle session physique longue n’est lancée ; les résultats physiques manquants restent `not_tested`.

Un ancien smoke logiciel EAR 48k de deux secondes avait compté 55 underruns buffer. Cette limite est conservée comme diagnostic historique ; son schéma ne permet pas une qualification actuelle. Le nouvel essai logiciel de quatre secondes, Manual T8/T16, rend 3,22 s de callback mais seulement 0,47 s de PCM attribué, avec 129 underruns buffer et la transition T16 non finalisée. Il vérifie le parcours du banc et conserve cet échec de performance, sans qualification physique.

## Corrections après les essais manuels

Les observations utilisateur ont conduit à activer le parallélisme CPU et les optimisations ORT, puis à préparer une fenêtre latente suivante pendant l’inférence. Le court run logiciel Rack T8 avec navigation de production passe maintenant sans underrun ; il ne remplace pas l’écoute ni la qualification physique. Le décodage natif utilise un processus persistant séparé et un registre générique de sources et d’overrides facultatifs. Le runtime utilise l’interpréteur courant par défaut ; les dépendances natives viennent des profils d’installation déclarés et isole le runtime Metal du décodeur de celui de la navigation. Le check EAR MPS T8→T30→T8 passe. Voir [les chemins, l’ownership et le protocole de retest](NATIVE_DECODER_RUNTIMES.md).
