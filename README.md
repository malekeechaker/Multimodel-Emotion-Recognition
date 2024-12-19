# Real-Time Emotion Detection System

## Introduction

Le **Système de Détection des Émotions en Temps Réel** est un projet innovant qui utilise plusieurs modalités de détection pour analyser les émotions humaines en temps réel. Ce système combine la vidéo, l'audio et le texte pour fournir une détection multi-modalité, offrant ainsi une approche robuste pour identifier les émotions dans des scénarios dynamiques et interactifs. En utilisant des modèles avancés de reconnaissance d'objets, de traitement audio et de reconnaissance vocale, ce système est capable de détecter les émotions humaines avec un haut degré de précision.

## Objectif

Le principal objectif de ce projet est de créer un système d'analyse émotionnelle en temps réel qui peut être utilisé dans diverses applications, telles que :

- **Amélioration de l'interaction homme-machine** : Le système peut être utilisé dans des environnements interactifs comme les jeux vidéo, les assistants virtuels, ou les plateformes de formation, permettant une compréhension plus profonde des émotions des utilisateurs.
- **Surveillance de la santé mentale** : En analysant les émotions humaines en temps réel, ce système peut être un outil précieux dans la détection précoce des troubles émotionnels ou des signes de stress.
- **Sécurité et marketing** : Le système peut également être utilisé dans des contextes de sécurité, comme la surveillance des émotions dans des environnements sensibles, ou dans le marketing pour analyser les réactions des clients.

## Fonctionnalités

- **Capture vidéo en temps réel** : Utilisation de YOLO pour détecter les visages et analyser les émotions exprimées par les expressions faciales.
- **Capture audio en temps réel** : Utilisation de PyAudio pour capter l'audio et un modèle d'émotion audio pour détecter les émotions dans la voix.
- **Reconnaissance vocale** : Utilisation de la bibliothèque SpeechRecognition pour convertir l'audio en texte et en analyser les émotions.
- **Fusion multi-modalité** : Fusion des résultats des trois modalités (vidéo, audio, texte) pour une détection plus précise de l'émotion dominante.
- **Affichage en temps réel** : Visualisation des prédictions émotionnelles sur les frames vidéo et présentation des résultats pour chaque modalité.

## Impact

### 1. **Amélioration de l'Interaction Sociale**
Ce système peut améliorer les interactions avec les machines en permettant aux systèmes intelligents de répondre en fonction de l'état émotionnel de l'utilisateur. Par exemple, dans un jeu vidéo, un personnage peut ajuster ses réponses en fonction des émotions de l'utilisateur, rendant l'expérience plus immersive.

### 2. **Support dans la Surveillance de la Santé Mentale**
Le projet peut jouer un rôle crucial dans le suivi de la santé mentale. En identifiant les signes de stress, de dépression ou d'autres troubles émotionnels, ce système pourrait potentiellement alerter les professionnels de santé ou fournir des interventions adaptées.

### 3. **Optimisation des Services Client**
Dans des domaines comme le marketing ou la relation client, comprendre les émotions d'un client pendant une interaction peut conduire à des réponses plus adaptées, améliorant ainsi l'expérience client et augmentant l'efficacité des campagnes marketing.

### 4. **Sécurité Renforcée**
Dans des environnements de sécurité, détecter des émotions telles que la peur, la colère ou l'anxiété peut aider à identifier des comportements suspects ou des menaces potentielles avant qu'elles ne deviennent des problèmes majeurs.

## Technologies Utilisées

- **YOLO (You Only Look Once)** : Pour la détection d'objets et l'analyse des émotions basées sur les expressions faciales dans la vidéo.
- **PyAudio** : Pour la capture audio en temps réel.
- **SpeechRecognition** : Pour la conversion de la parole en texte et l'analyse des émotions à partir du texte.
- **TensorFlow** : Pour le traitement et la classification des émotions dans les données audio et textuelles.
- **OpenCV** : Pour l'affichage vidéo en temps réel et l'annotation des résultats sur les frames vidéo.

## Structure du Projet

- **fusion/** : Contient le code pour la fusion des résultats des prédictions des trois modalités.
- **Video/** : Code de traitement et de détection vidéo en temps réel avec YOLO.
- **Audio/** : Code de traitement de l'audio et de la détection d'émotions audio.
- **Text/** : Code pour la reconnaissance vocale et l'analyse des émotions textuelles.
- **interface.py** : code d'une proposition d'interface en utilisant streamlit

## Installation

1. Clonez ce dépôt :
```bash
git clone https://github.com/malekeechaker/Multimodel-Emotion-Recognition.git
cd fusion
```

2. Installez les dépendances via `pip` :
```bash
pip install -r requirements.txt
```

3. Téléchargez les modèles pré-entraînés nécessaires pour l'audio, la vidéo et le texte (assurez-vous que les chemins sont correctement configurés dans le code).

## Utilisation

Pour exécuter le système en temps réel, exécutez le script principal :

```bash
python fenetre.py
```

Le système commencera à capturer la vidéo et l'audio, puis effectuera des prédictions sur l'émotion présente dans l'environnement. Vous pouvez arrêter le système en appuyant sur `Q` dans la fenêtre de la vidéo.

---

## Améliorations et Développements Futurs

- **Précision améliorée** : Optimiser les modèles en utilisant des jeux de données plus diversifiés pour chaque modalité.
- **Interface Utilisateur** : Développer une interface graphique pour afficher les émotions détectées en temps réel et permettre une interaction avec l'utilisateur.
- **Adaptation en fonction de l'environnement** : Améliorer la capacité du système à s'adapter aux différents environnements (par exemple, variations d'éclairage, bruits de fond) pour des prédictions plus robustes.
- **Support pour d'autres modalités** : Intégrer d'autres sources de données telles que des capteurs biométriques (pouls, température) pour une analyse plus complète des émotions.

---

## Conclusion

Ce projet est un pas vers des systèmes intelligents plus humains et interactifs. En combinant des informations provenant de différentes sources (vidéo, audio et texte), nous créons une approche plus complète et fiable pour détecter les émotions, avec des applications potentielles dans de nombreux domaines, allant des soins de santé à la sécurité en passant par le marketing.
