# Détection des Émotions avec YOLO

Ce projet utilise le modèle **YOLO** (You Only Look Once) pour détecter et classifier les émotions humaines en temps réel. Grâce à sa rapidité et son efficacité, YOLO est particulièrement adapté aux applications en temps réel telles que la sécurité, le service client et la santé mentale.

## Table des Matières
- [Pré-requis](#pré-requis)
- [Installation](#installation)
- [Structure du Projet](#structure-du-projet)
- [Détails du Modèle](#détails-du-modèle)
- [Résultats](#résultats)
- [Améliorations Potentielles](#améliorations-potentielles)

---

## Pré-requis

Avant de commencer, assurez-vous que les outils et librairies suivants sont installés :

- Python >= 3.8
- PyTorch (compatible avec votre GPU si disponible)
- Optuna
- NumPy

Installez les dépendances nécessaires avec :
```bash
pip install torch optuna numpy
```

---

## Installation

1. Clonez ce dépôt :
```bash
git clone https://github.com/votre-repo/unified-emotion-prediction.git
cd unified-emotion-prediction
```

2. Installez les dépendances via `pip` :
```bash
pip install -r requirements.txt
```

3. Assurez-vous que les données d'entraînement et de test sont formatées correctement avant d'exécuter les scripts.

---

## Structure du Projet

- **data/** : Contient les fichiers de données (entraînement, validation, test).
- **scripts/** : Scripts Python pour l’analyse des données, l’entraînement et l’évaluation du modèle.
- **models/** : Modèles sauvegardés et résultats d’entraînement.
- **notebooks/** : Notebooks Jupyter pour l’exploration des données et l’expérimentation.
- **README.md** : Documentation du projet.

---

## Détails du Modèle

### 1. Préparation des Données
- Création du dossier `yolo-data` pour organiser les données.
- Copie des données de validation et de test dans les répertoires appropriés.
- Équilibrage des données d’entraînement pour éviter le biais.

### 2. Entraînement du Modèle
- Utilisation du modèle YOLO pour la détection des émotions sur les images.
- Optimisation des hyperparamètres avec **Optuna** pour maximiser le mAP sur le jeu de validation.
- Configuration :
  - Optimiseur : Adam
  - Fonction de perte : `Binary Crossentropy`
  - Taux d'apprentissage ajustable.
  - Batch size et epochs personnalisables.

### 3. Évaluation
- Calcul des métriques (mAP, précision, rappel).
- Visualisation des courbes d'apprentissage (perte et mAP).
- Création d’une matrice de confusion pour analyser les performances par classe.

### 4. Sauvegarde et Chargement du Modèle
- Modèle sauvegardé dans le répertoire `models/`.
- Validation que le modèle chargé reproduit les performances initiales.

---

## Résultats

- **mAP obtenu** : 0.78
- **Visualisation** : Les courbes montrent une convergence stable pendant l’entraînement.
- **Performances** : Le modèle atteint un niveau satisfaisant de détection des émotions en temps réel.

---

## Améliorations Potentielles

1. **Optimisation des Hyperparamètres** :
   Vous pouvez explorer des configurations plus fines des hyperparamètres en modifiant les plages de recherche dans Optuna.

3. **Amélioration de la Précision** :
   - Ajouter des techniques de régularisation comme le **Dropout** ou l’**augmentation de données** pour améliorer la généralisation du modèle.
   - Utiliser des modèles plus complexes pour une détection plus précise des émotions.

4. **Détection Plus Rapide** :
   - Explorez l’utilisation de modèles plus légers comme **MobileNet** ou **Tiny-YOLO** pour des applications en temps réel avec des ressources limitées.

5. **Évaluation Avancée** :
   - Ajoutez des visualisations détaillées, comme des cartes de chaleur (heatmaps), pour observer l’attention du modèle et ses zones de détection.
```
