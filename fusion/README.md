# Real-Time Emotion Detection System

Ce projet implémente un système de détection d'émotions en temps réel en utilisant plusieurs modalités : vidéo (avec YOLO), audio (avec un modèle de détection d'émotions audio) et texte (avec la reconnaissance vocale et une analyse du texte). Les résultats de chaque modalité sont fusionnés pour déterminer l'émotion la plus probable. Le système utilise des technologies telles que OpenCV, PyAudio et SpeechRecognition pour l'acquisition en temps réel des données.

## Table des Matières

- [Pré-requis](#pré-requis)
- [Installation](#installation)
- [Structure du Projet](#structure-du-projet)
- [Détails du Modèle](#détails-du-modèle)
- [Utilisation](#utilisation)
- [Améliorations Potentielles](#améliorations-potentielles)

---

## Pré-requis

Avant de commencer, assurez-vous que les outils et librairies suivants sont installés :

- Python >= 3.7
- OpenCV
- PyAudio
- SpeechRecognition
- TensorFlow >= 2.0

Installez les dépendances nécessaires avec :

```bash
pip install opencv-python pyaudio SpeechRecognition tensorflow
```

---

## Installation

1. Clonez ce dépôt :
```bash
git clone https://github.com/votre-utilisateur/real-time-emotion-detection.git
cd real-time-emotion-detection
```

2. Installez les dépendances via `pip` :
```bash
pip install -r requirements.txt
```

3. Assurez-vous d'avoir les modèles pré-entraînés nécessaires pour l'audio et la vidéo dans les chemins spécifiés dans le code. Vous pouvez également entraîner vos propres modèles si nécessaire.

---

## Structure du Projet

- **fusion/** : Contient le code pour la fusion des résultats des différentes modalités.
- **RealTimevideo/** : Scripts pour la gestion de la vidéo en temps réel avec YOLO.
- **realtimeaudio/** : Scripts pour la détection d'émotions audio.
- **realtimetext/** : Scripts pour la détection des émotions à partir du texte (reconnaissance vocale et analyse du texte).
- **fenetre.py** : Le script principal qui exécute le système en temps réel.

---

## Détails du Modèle

### 1. Capture Vidéo en Temps Réel
- Utilisation de YOLO pour la détection des objets et la classification des émotions à partir des visages dans les vidéos en temps réel.
- Le modèle YOLO est chargé depuis le répertoire `RealTimevideo` et utilisé pour annoter les frames de la vidéo.

### 2. Capture Audio en Temps Réel
- Utilisation de PyAudio pour capturer l'audio en temps réel.
- Un modèle d'émotions audio est utilisé pour analyser les émotions dans l'audio capturé.
- Le modèle de détection audio est chargé depuis `realtimeaudio`.

### 3. Reconnaissance Vocale et Prédiction Textuelle
- Utilisation de la bibliothèque SpeechRecognition pour convertir l'audio en texte.
- Le texte est ensuite analysé pour en extraire l'émotion associée.

### 4. Fusion des Prédictions
- Les prédictions des trois modalités (vidéo, audio, texte) sont fusionnées à l'aide de la fonction `unified_prediction` dans `fusion.py` pour donner la prédiction d'émotion finale.
  
### 5. Annotation des Frames
- Les frames vidéo sont annotées avec les résultats des prédictions d'émotions issues de chaque modalité et de la fusion finale.

---

## Utilisation

Pour exécuter le système en temps réel, exécutez simplement le script principal :

```bash
python main.py
```

Le système commencera à capturer la vidéo et l'audio, puis effectuera des prédictions sur l'émotion présente dans l'environnement. Vous pouvez arrêter le système en appuyant sur `Q` dans la fenêtre de la vidéo.

---

## Améliorations Potentielles

1. **Amélioration de la Précision** :
   - Entraîner des modèles spécifiques pour chaque modalité sur un jeu de données plus large.
   - Optimiser le processus de fusion des résultats pour une détection plus précise.

2. **Surveillance et Débogage** :
   - Ajouter une interface utilisateur pour afficher les prédictions en temps réel et l'historique des émotions détectées.

3. **Améliorations de l'Interface** :
   - Ajouter une interface graphique pour visualiser les résultats d'émotions sur la vidéo, audio et texte en temps réel.

4. **Optimisation** :
   - Réduire le temps de latence pour le traitement en temps réel avec des techniques de réduction de la taille des modèles.

---

**Note** : Assurez-vous que vos chemins de modèles sont corrects dans les scripts pour éviter toute erreur lors du chargement des modèles.
