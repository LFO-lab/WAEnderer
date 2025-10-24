# Feuille de route Stable Audio Wanderer

## Objectif
Développement progressif du moteur d’exploration et de génération audio basé sur Stable Audio Open, organisé en six étapes évolutives.

---

## 1. Refactorisation et métriques
- Structurer le code en modules (`preprocess`, `runtime`, `audio`, `models`, `utils`)
- Centraliser les paramètres (YAML/JSON)
- Ajouter logs et métriques :
  - Trustworthiness / Continuity
  - Corrélation distance(embedding) vs distance(MFCC)
  - Visualisation : centroïde spectral, brillance, RMS

---

## 2. Réduction de dimension
- Intégrer plusieurs méthodes :
  - **PCA** (par défaut)
  - **UMAP** (`umap-learn`) – bon voisinage local
  - **t-SNE** – analyse visuelle
- Paramètres exposés : `n_neighbors`, `min_dist`, `metric`
- Normalisation min–max par dimension
- Script d’évaluation : `eval_projection.py`

---

## 3. Augmentation de données
- **Techniques :**
  - Transposition ±1–3 demi-tons
  - Variation de volume ±3 dB
  - Time-stretch 0.9×–1.1×
  - EQ tilt doux (±1–2 dB/oct)
  - Légère reverb convolution
  - Ajout de bruit rose/blanc subtil
- Appliquer avant MFCC et encodage VAE
- Option `--augment` dans le preprocess

---

## 4. Moteur audio
- **Améliorations :**
  - Fenêtres sqrt-Hann
  - Granulation multi-voix (mélange de plusieurs kNN)
  - Hops irréguliers (Poisson jitter)
  - Freeze spectral (drone statique)
  - Étirement latent (`hop_lat` variable)
  - Pitch-shift (phase vocoder)
- Paramètres : `--voices`, `--mix_knn`, `--freeze`, `--pitch_shift`, `--latent_stretch`

---

## 5. MLP régressif (navigation lisse)
- **But :** remplacer ou compléter kNN par un MLP D→64
- Données : `(coords_DR, z_latent)`
- Réseau : MLP 2×128 ReLU, régularisation L2 + lissage
- Mode hybride : blend kNN/MLP (`--nav_mode hybrid`)
- Scripts : `train_mlp.py`, `mlp.pt`

---

## 6. LSTM pour trajectoires génératives
- **But :** générer des mouvements continus dans l’espace latent
- Données : séquences temporelles de `z` ou `coords_DR`
- Entraînement : prédiction next-step (teacher forcing)
- Mode “assisté” : OSC + drift LSTM (poids α)
- Scripts : `train_lstm.py`, `--lstm_assist α`

---

## Ordre de développement
1. Refactor + métriques  
2. Réduction de dimension  
3. Moteur audio  
4. Augmentation de données  
5. MLP  
6. LSTM

---

## Idées additionnelles
- Presets et sessions sauvegardables (`.yaml`)
- API OSC étendue : `/cursor`, `/freeze`, `/voices`, `/preset`
- Visualisation interactive (p5.js, WebGL)
