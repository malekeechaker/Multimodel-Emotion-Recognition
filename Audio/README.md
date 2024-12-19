# Model Training and Callbacks for Classification

Ce projet montre comment entraîner un modèle de classification de data de type audio avec des callbacks personnalisés pour suivre la performance du modèle et ajuster l'entraînement en conséquence. L'objectif est de calculer des métriques telles que le score F1, de sauvegarder le meilleur modèle et de suivre l'évolution des performances pendant l'entraînement.

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

- Python >= 3.7
- TensorFlow >= 2.0
- scikit-learn
- Numpy

Installez les dépendances nécessaires avec :
```bash
pip install tensorflow scikit-learn numpy
```

---

## Installation

1. Clonez ce dépôt :
```bash
git clone https://github.com/username/Model-Training-Callbacks.git
cd Model-Training-Callbacks
```

2. Installez les dépendances via `pip` :
```bash
pip install -r requirements.txt
```

3. Assurez-vous que les données d'entraînement et de test sont formatées correctement avant d'exécuter le script.

---

## Structure du Projet

- **data/** : Contient les fichiers de données (entraînement et test).
- **scripts/** : Scripts Python pour le traitement des données et l'entraînement du modèle.
- **models/** : Modèles sauvegardés au format HDF5.
- **notebooks/** : Notebooks Jupyter pour visualisation et expérimentation.
- **README.md** : Documentation du projet.

---

## Détails du Modèle

### 1. Préparation des Données
- Chargement des données d'entraînement et de test.
- Prétraitement des données, y compris la normalisation des caractéristiques et l'encodage des labels.
  
### 2. Entraînement du Modèle
- Utilisation d'un modèle de réseau de neurones ou d'un modèle pré-entraîné (par exemple, BERT pour la classification de texte).
- Configuration :
  - Optimiseur : Adam
  - Fonction de perte : `CategoricalCrossentropy`
  - Taux d'apprentissage ajustable.
  - Batch size et epochs personnalisables.
  
### 3. Callbacks Personnalisés
- **F1 Score Callback** : Calcul et affichage du score F1 à chaque époque.
- **ModelCheckpoint** : Sauvegarde du modèle à chaque amélioration du score F1.
- **EarlyStopping** : Arrêt anticipé si la performance ne s'améliore pas pendant plusieurs époques.

### 4. Évaluation
- Calcul des métriques (Accuracy, F1-score, Precision, Recall).
- Visualisation des courbes de perte et d'exactitude.
- Sauvegarde du meilleur modèle basé sur les performances du score F1.

### 5. Sauvegarde et Chargement du Modèle
- Sauvegarde du modèle au format HDF5 si le score F1 est amélioré.
- Validation que le modèle rechargé conserve les performances initiales.

---

## Résultats

- **F1-Score** : ~0.75 sur l'ensemble de validation.
- **Accuracies** : Précision globale de ~80% pour les données de test.
- **Visualisation** : Les courbes de perte et d'exactitude montrent une convergence stable après plusieurs époques d'entraînement.

---

## Améliorations Potentielles

1. **Gestion des avertissements de parallélisme** :
   Ajoutez ce code pour supprimer les avertissements liés au tokenizer dans des environnements parallèles :
   ```python
   import os
   os.environ["TOKENIZERS_PARALLELISM"] = "false"
   ```

2. **Prévention du surapprentissage** :
   - Introduire une régularisation via Dropout.
   - Utiliser une stratégie d'arrêt anticipé pour prévenir l'overfitting.

3. **Modèles Alternatifs** :
   - Explorer des modèles plus légers comme **DistilBERT** ou **MobileBERT** pour réduire la complexité et améliorer la vitesse de prédiction.

4. **Évaluation Avancée** :
   - Ajouter des métriques comme la **matrice de confusion** pour une évaluation détaillée des classes :
     ```python
     from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
     cm = confusion_matrix(y_test, y_pred)
     ConfusionMatrixDisplay(cm).plot()
     ```

---

## Contribuer

Si vous souhaitez contribuer à ce projet, veuillez forker le dépôt, créer une branche et soumettre une Pull Request. N'oubliez pas de mettre à jour la documentation si vous ajoutez de nouvelles fonctionnalités.

---

## Licence

Distribué sous la licence MIT. 

---

### Explication du README

1. **Pré-requis** : Liste les versions minimales de Python et des bibliothèques nécessaires.
2. **Installation** : Indique les étapes pour cloner le projet et installer les dépendances.
3. **Structure du projet** : Explique l'organisation du répertoire du projet.
4. **Détails du modèle** : Présente la préparation des données, l'entraînement, les callbacks, l'évaluation et la sauvegarde du modèle.
5. **Résultats** : Fournit un aperçu des performances du modèle.
6. **Améliorations potentielles** : Propose des idées pour améliorer le modèle, comme la gestion des avertissements de parallélisme, la prévention du surapprentissage et l'utilisation de modèles plus légers.
