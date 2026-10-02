# Fiche de validation perceptuelle

Statut : **écoute comparative effectuée par l’utilisateur le 2 octobre 2026**. Retour : « très similaire », avec la version ONNX légèrement plus bruiteuse que MPS. Cette observation est conservée ; l’identité perceptuelle n’est pas déclarée.

Corpus : `BurntMemory_20260915_211501`. Les paires produites par
`eval_scripts/compare_corpus_decoders.py` sont conservées localement dans
`eval_out/dual_inference_phase5/`. Ne pas les redistribuer sans autorisation.
Le script dénormalise les mêmes frames, décode T8 avec hop 4 et assemble avec
l’overlap-add de production, sans normalisation ni clipping des sorties.

Pour chaque paire `fileN_onnx.wav` / `fileN_native.wav` : comparer à volume
identique, puis écouter les attaques, les queues et les raccords. Comparer
également une session interactive avec changements de fenêtre/mode.

| Champ | Observation |
| --- | --- |
| Auditeur et date | Utilisateur du projet, 2 octobre 2026 |
| Machine, périphérique audio, fréquence/bloc | Non précisé dans le retour utilisateur |
| Casque/enceintes et niveau | Non précisé dans le retour utilisateur |
| Empreinte du corpus et rapport associé | `dual_inference_phase5_corpus_mps.json` |
| Attaques et transitoires | Non précisé dans le retour utilisateur |
| Queues et timbres | Non précisé dans le retour utilisateur |
| Clics/raccords OLA et transitions | Non précisé dans le retour utilisateur |
| Niveau, saturation, asymétrie stéréo | Non précisé dans le retour utilisateur |
| Verdict et anomalies horodatées | Rendus très similaires ; ONNX légèrement plus bruiteux que MPS. Aucun horodatage fourni. |

Un écart numérique compatible avec le bruit stochastique ne remplace pas ce
verdict. L’écoute comparative est effectuée ; elle ne constitue pas un protocole aveugle
ni une validation détaillée de toutes les transitions interactives. Les champs
non précisés restent explicitement inconnus. Le critère retenu est l’absence
de dégradation du nouveau moteur signalée dans ce retour, sans prétendre que
les différences de bruit ont disparu.
