# Phase 5 — Campagne de validation

Statut : **phase 5 terminée sur ONNX CPU et MPS dans le scénario matériel documenté**. Le corpus demandé est `BurntMemory_20260915_211501` (43 982 frames, 256 dimensions, deux fichiers, SAME-S). Les rapports portent son SHA-256. Aucun audio du corpus n’est ajouté aux fichiers suivis par Git.

## Protocole et seuils

Les tolérances ont été fixées avant les nouveaux essais :

- Entrées synthétiques historiques : SNR > 20 dB et RMSE < 0,005, huit graines par fenêtre paire T2 à T32. Le script consigne désormais la variabilité des répétitions ONNX **et** natives, et les percentiles de latence. Il conserve la stochasticité des modèles.
- Corpus réel : quatre positions par fichier, 16 fenêtres, deux répétitions par moteur. Le seuil de régression est `RMSE inter-moteurs ≤ 3 × max(RMSE répétition ONNX, RMSE répétition native, 1e-7)`. Ce seuil relatif au bruit est un contrôle expérimental, pas un seuil perceptuel universel. Un échec doit être analysé, pas masqué par un changement de seuil a posteriori.
- Rendus d’écoute : mêmes frames dénormalisées, T8/hop 4, overlap-add de production, pas de normalisation ou clipping ajouté. Les différences après OLA sont consignées séparément.
- Transport : dix minutes par moteur, exécutées successivement. Callback de 1 024 samples à 44,1 kHz piloté par une horloge logicielle silencieuse. Changements toutes les dix secondes : T2 à T32, puis bornes adaptatives, en random/manual/reorganized. Le transport, le décodeur, le planificateur, l’OLA et les buffers sont réels ; les sources de navigation sont des adaptateurs de replay déterministes, pas les politiques apprises.

Le corpus et le programme de commandes sont identiques entre moteurs ; le nombre de calculs préchargés peut différer. Les comparaisons directes du script de parité, elles, portent sur exactement les mêmes latents. Les timings de préparation concernent le constructeur du décodeur, pas un redémarrage à froid du système d’exploitation.

## Diagnostic initial — avant correction

Le rapport ONNX a terminé 600 secondes avec 434 callbacks sous-alimentés. Les pointes sont concentrées autour de T18/T20 ; la cause n’est pas établie par ce seul essai. Le scénario ne passe donc **pas** le critère zéro underrun. Les échecs sont conservés dans le rapport, sans restreindre silencieusement les fenêtres. La correction et les nouveaux essais sont décrits plus bas.

Les tableaux détaillés ci-dessous comparent les deux sessions ; MPS a terminé 600 secondes sans sous-alimentation. Les percentiles par fenêtre incluent le nombre d’observations : un p99 calculé sur quelques dizaines d’appels reste une estimation fragile. `process_cpu_percent` compte les secondes CPU du processus rapportées au temps réel (100 % = un cœur) ; le RSS maximal utilise les unités de `getrusage` de la plateforme. Le compteur mémoire MPS représente les tenseurs alloués, pas toute la mémoire du pilote.

Un callback sous-alimenté n’est pas nécessairement un décrochage du périphérique audio : le banc logiciel n’en ouvre aucun. Inversement, zéro underrun dans ce banc ne prouve pas qu’un périphérique réel fonctionnera sans interruption. Les commandes peuvent prendre du temps à devenir audibles parce que le transport prépare puis traverse une transition.

## Reproduire

Dans l’environnement natif épinglé, depuis la racine du dépôt :

```sh
PYTHONPATH=. .venv-native/bin/python eval_scripts/validate_torch_decoder.py --device mps --output docs/dual_inference_phase5_synthetic_mps.json
PYTHONPATH=. .venv-native/bin/python eval_scripts/compare_corpus_decoders.py --device mps --corpus corpus/BurntMemory_20260915_211501 --output docs/dual_inference_phase5_corpus_mps.json --audio-dir eval_out/dual_inference_phase5
PYTHONPATH=. .venv-native/bin/python eval_scripts/benchmark_dual_inference.py --backend onnxruntime --device cpu --seconds 600 --corpus corpus/BurntMemory_20260915_211501 --output docs/dual_inference_phase5_onnx.json
PYTHONPATH=. .venv-native/bin/python eval_scripts/benchmark_dual_inference.py --backend pytorch --device mps --seconds 600 --corpus corpus/BurntMemory_20260915_211501 --output docs/dual_inference_phase5_mps.json
```

Pour un périphérique physique, ajouter `--audio-device "nom du périphérique"`. Ce mode est explicitement demandé et sa sortie est muette ; il exerce le callback physique, mais ne remplace pas l’écoute. Ajouter `--require-zero-underruns` pour que la commande échoue si ce critère n’est pas satisfait, tout en conservant le rapport. Ne pas exécuter les essais de performance simultanément ni avec d’autres calculs d’inférence lourds.

Le smoke CPU peut s’exécuter dans l’environnement ONNX existant :

```sh
PYTHONPATH=. .venv/bin/python eval_scripts/smoke_cpu_installation.py --corpus corpus/BurntMemory_20260915_211501 --output docs/dual_inference_phase6_cpu.json
```

## Périmètre de qualification

La livraison vise les scénarios ONNX CPU et PyTorch MPS mesurés sur le M1 Max et Core Audio. CUDA reste expérimental et non qualifié faute de machine NVIDIA ; aucune mesure MPS ne lui est attribuée. La validation perceptuelle repose sur le retour comparatif de l’utilisateur et conserve sa différence de bruit observée.

Les échecs initiaux sont conservés ci-dessous pour expliquer la correction. Les rapports physiques après correction et le manifeste de qualification font foi pour la conclusion finale.

## Mesures historiques sur horloge logicielle (avant correction)

| Mesure | ONNX CPU | PyTorch MPS |
| --- | ---: | ---: |
| Durée (s) | 600.02 | 600.03 |
| Préparation du décodeur (ms) | 167.99 | 4145.04 |
| Callbacks sous-alimentés | 434.00 | 0.00 |
| CPU du processus (%) | 59.90 | 10.13 |
| Réaction commandes p50 (ms, hors activation adaptive) | 378.3 | 50.2 |
| Réaction commandes p95 (ms, hors activation adaptive) | 730.7 | 72.5 |
| Réaction commandes p99 (ms, hors activation adaptive) | 4561.3 | 75.8 |

| Fenêtre | Budget hop (ms) | ONNX médiane / p99 (ms) | MPS médiane / p99 (ms) | Appels ONNX / MPS |
| --- | ---: | ---: | ---: | ---: |
| T2 | 92.9 | 65.6 / 67.7 | 24.2 / 31.4 | 535 / 546 |
| T4 | 185.8 | 110.8 / 113.8 | 26.8 / 36.1 | 224 / 220 |
| T6 | 278.6 | 156.3 / 164.6 | 25.5 / 39.3 | 151 / 148 |
| T8 | 371.5 | 200.1 / 209.5 | 26.8 / 39.1 | 116 / 112 |
| T10 | 464.4 | 242.5 / 249.7 | 28.4 / 40.1 | 92 / 92 |
| T12 | 557.3 | 283.2 / 296.8 | 29.3 / 38.4 | 78 / 76 |
| T14 | 650.2 | 326.7 / 355.3 | 30.2 / 38.5 | 69 / 68 |
| T16 | 743.0 | 371.6 / 445.4 | 31.6 / 46.0 | 60 / 60 |
| T18 | 835.9 | 414.0 / 3254.5 | 33.0 / 41.4 | 53 / 52 |
| T20 | 928.8 | 453.6 / 7118.5 | 33.4 / 59.2 | 27 / 36 |
| T22 | 1021.7 | 496.5 / 728.9 | 36.3 / 47.3 | 33 / 33 |
| T24 | 1114.6 | 536.0 / 579.2 | 38.8 / 80.1 | 33 / 30 |
| T26 | 1207.4 | 597.5 / 661.9 | 51.4 / 71.8 | 30 / 30 |
| T28 | 1300.3 | 628.5 / 660.0 | 56.1 / 71.2 | 27 / 27 |
| T30 | 1393.2 | 668.1 / 700.6 | 59.7 / 73.4 | 27 / 27 |
| T32 | 1486.1 | 712.0 / 758.2 | 62.6 / 71.4 | 36 / 38 |

Rapports bruts : [ONNX](dual_inference_phase5_onnx.json), [MPS](dual_inference_phase5_mps.json). La mémoire de tenseurs MPS échantillonnée varie entre 222 789 376 et 246 089 728 octets, selon les fenêtres et les calculs en cours.

## Comparaison sur BurntMemory et écoute

Les **128 comparaisons** (8 positions × 16 fenêtres) passent le seuil relatif à la variabilité intra-moteur fixé ci-dessus. RMSE inter-moteurs : 0,000118 à 0,001390 ; SNR : 29,37 à 36,45 dB. Voir [le rapport complet](dual_inference_phase5_corpus_mps.json), qui conserve aussi les répétitions ONNX et natives.

Les deux paires d’écoute ont chacune 507 904 samples (~11,52 s). Après OLA, la paire du premier fichier mesure RMSE 0,000379 / SNR 38,05 dB ; celle du second RMSE 0,000164 / SNR 35,26 dB. Les WAV restent locaux :

- Premier fichier : [ONNX](../eval_out/dual_inference_phase5/file0_onnx.wav), [MPS](../eval_out/dual_inference_phase5/file0_native.wav).
- Second fichier : [ONNX](../eval_out/dual_inference_phase5/file1_onnx.wav), [MPS](../eval_out/dual_inference_phase5/file1_native.wav).

Ces liens sont utilisables dans le checkout qui a produit les fichiers ; les WAV ne sont pas distribués dans le dépôt ni l’archive source. Le retour de l’utilisateur est consigné dans [la fiche d’écoute](DUAL_INFERENCE_LISTENING.md), avec les détails non fournis laissés inconnus.

## Correction et qualification physique — suite du 2 octobre

Le contrôle ciblé Reorganized a distingué **un manque de PCM dans le buffer** d’un problème de pilote (1 événement buffer, 0 pilote). La réserve du transport passe de un à **deux hops** : le producteur peut préparer les prochaines frames de navigation et le décodage pendant qu’un hop supplémentaire reste disponible. La borne de deux calculs en vol et le rejet des générations obsolètes sont conservés. Un test rend deux hops sous verrou sans permettre au producteur de recharger le buffer ; il détecte une réserve insuffisante.

Les compteurs `buffer_underruns` et `device_underruns` sont désormais séparés, en conservant `underruns` comme total compatible. Le lecteur bloquant prend également en compte l’underflow retourné par `OutputStream.write`, auparavant ignoré. Aucun échauffement ONNX systématique n’a été ajouté : le profil ciblé T2..T32, y compris deux appels concurrents, n’a pas reproduit le coût anormal supposé de premier appel.

Les nouveaux essais utilisent `--navigation production --audio-device "Haut-parleurs MacBook Pro"` : politiques et artefacts de BurntMemory chargés par le vrai setup Web, Core Audio à 44,1 kHz / 1 024 samples, sortie muette. Le programme couvre les 16 fenêtres, les trois modes et l’activation adaptive. Le coût de préchargement supplémentaire est volontaire ; les nouveaux rapports mesurent sa conséquence sur les latences de commande.

L’utilisateur a écouté les extraits le 2 octobre : rendus très similaires, ONNX légèrement plus bruiteux que MPS. L’observation est consignée dans la fiche, sans prétendre à une identité perceptuelle.

## Reproduire la qualification physique

Exécuter successivement dans l’environnement natif épinglé :

```sh
PYTHONPATH=. .venv-native/bin/python eval_scripts/benchmark_dual_inference.py --backend onnxruntime --device cpu --navigation production --audio-device "Haut-parleurs MacBook Pro" --seconds 600 --corpus corpus/BurntMemory_20260915_211501 --output docs/dual_inference_phase5_physical_onnx.json --require-zero-underruns
PYTHONPATH=. .venv-native/bin/python eval_scripts/benchmark_dual_inference.py --backend pytorch --device mps --navigation production --audio-device "Haut-parleurs MacBook Pro" --seconds 600 --corpus corpus/BurntMemory_20260915_211501 --output docs/dual_inference_phase5_physical_mps.json --require-zero-underruns
PYTHONPATH=. .venv/bin/python bin/check_dual_inference_qualification.py docs/dual_inference_qualification.json
```

Chaque rapport brut laisse `audio_device_qualified: false` et `listening_review: pending` : le banc seul ne prononce pas un verdict perceptuel. Le contrôle séparé combine les rapports, les comparaisons numériques et la fiche d’écoute du manifeste. Une sortie muette vérifie le callback physique ; les WAV comparatifs servent à l’écoute.

## Conclusion — campagne physique après correction, 2 octobre 2026

**Phase 5 terminée pour le périmètre ONNX CPU / MPS testé.** Les deux sessions de dix minutes passent sans underrun, avec navigation de production, les 16 fenêtres fixes et les réglages adaptatifs dans les trois modes. Le [contrôle des preuves](dual_inference_qualification_result.json) passe. CUDA reste expérimental et non qualifié.

| Mesure | ONNX CPU | PyTorch MPS |
| --- | ---: | ---: |
| Durée (s) | 600.38 | 600.19 |
| Préparation (ms) | 176.50 | 4960.33 |
| Callbacks rendus | 25785.00 | 25799.00 |
| Sous-alimentations PCM | 0.00 | 0.00 |
| Incidents périphérique | 0.00 | 0.00 |
| CPU processus (%) | 70.01 | 19.36 |
| RSS maximal (Mio) | 1278.4 | 1606.5 |
| Réaction commandes p50 (ms, hors activation adaptive) | 835.6 | 142.5 |
| Réaction commandes p95 (ms, hors activation adaptive) | 1611.8 | 553.1 |
| Réaction commandes p99 (ms, hors activation adaptive) | 1826.8 | 681.5 |

| Fenêtre | Budget hop (ms) | ONNX médiane / p95 / p99 (ms) | MPS médiane / p95 / p99 (ms) | Appels ONNX / MPS |
| --- | ---: | ---: | ---: | ---: |
| T2 | 92.9 | 66.4 / 68.2 / 68.6 | 26.1 / 32.2 / 44.0 | 511 / 530 |
| T4 | 185.8 | 109.9 / 114.0 / 114.7 | 30.4 / 37.3 / 63.3 | 228 / 225 |
| T6 | 278.6 | 154.1 / 160.3 / 161.0 | 32.4 / 41.2 / 60.2 | 156 / 153 |
| T8 | 371.5 | 198.5 / 207.9 / 209.5 | 32.2 / 41.5 / 77.1 | 119 / 120 |
| T10 | 464.4 | 243.4 / 254.8 / 256.0 | 32.6 / 42.5 / 57.4 | 100 / 96 |
| T12 | 557.3 | 284.1 / 301.5 / 307.1 | 35.2 / 44.0 / 95.6 | 87 / 80 |
| T14 | 650.2 | 328.6 / 349.8 / 362.2 | 36.1 / 54.7 / 78.9 | 72 / 72 |
| T16 | 743.0 | 372.3 / 393.6 / 395.3 | 36.7 / 47.1 / 73.4 | 64 / 64 |
| T18 | 835.9 | 414.8 / 437.0 / 451.9 | 37.4 / 48.2 / 55.2 | 61 / 55 |
| T20 | 928.8 | 459.2 / 482.9 / 486.0 | 41.5 / 50.7 / 56.9 | 39 / 39 |
| T22 | 1021.7 | 500.5 / 526.4 / 546.9 | 44.9 / 51.4 / 64.7 | 36 / 36 |
| T24 | 1114.6 | 542.1 / 569.8 / 673.4 | 44.1 / 72.4 / 86.7 | 35 / 34 |
| T26 | 1207.4 | 588.1 / 650.6 / 883.6 | 50.4 / 76.8 / 81.5 | 33 / 33 |
| T28 | 1300.3 | 630.9 / 659.7 / 666.5 | 58.6 / 82.5 / 83.9 | 30 / 30 |
| T30 | 1393.2 | 665.2 / 702.3 / 713.2 | 71.3 / 85.4 / 88.2 | 30 / 30 |
| T32 | 1486.1 | 715.9 / 753.8 / 765.8 | 76.3 / 86.7 / 90.6 | 39 / 42 |

Rapports complets : [ONNX physique](dual_inference_phase5_physical_onnx.json), [MPS physique](dual_inference_phase5_physical_mps.json). Mémoire de tenseurs MPS échantillonnée : 228.7 à 249.5 Mio. Le RSS inclut navigation et modèles ; il ne mesure pas seulement le décodeur.

Le premier essai physique ONNX avait relevé 12 incidents ([rapport conservé](dual_inference_phase5_physical_onnx_baseline.json)). L’essai après correction en relève zéro. Ce résultat valide le scénario mesuré ; il ne démontre pas que toutes les pointes de latence du premier banc logiciel avaient une cause unique. La réserve accrue peut allonger le préchargement et les transitions ; les percentiles ci-dessus incluent cet effet.

Parité synthétique : 128/128 ; parité BurntMemory : 128/128. Écoute utilisateur : très similaire, ONNX légèrement plus bruiteux. Aucun changement du bruit natif ou des seuils n’a été utilisé pour faire passer ces critères. Ce retour ne remplace pas une écoute détaillée de toutes les interactions sur une autre machine.

Régression finale : **252 tests Python réussis, 1 optionnel ignoré**, trois suites Web réussies. Les nouveaux tests couvrent la réserve de deux hops, la séparation des compteurs et le rejet de preuves de qualification incomplètes.
