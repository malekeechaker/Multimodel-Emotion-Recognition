# BERT-Based Emotion Classification

Ce projet utilise un modèle BERT pour effectuer une classification multi-classes sur des données textuelles afin de détecter différentes émotions. Le workflow inclut la préparation des données, l'entraînement du modèle, l'évaluation et la sauvegarde.

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
- Transformers >= 4.0
- scikit-learn
- Matplotlib

Installez les dépendances nécessaires avec :
```bash
pip install tensorflow transformers scikit-learn matplotlib
```

---

## Installation

1. Clonez ce dépôt :
```bash
git clone https://github.com/malekeechaker/Multimodel-Emotion-Recognition.git
cd Text
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
- Tokenisation des textes avec le tokenizer BERT.
- Conversion des labels en format catégoriel pour une classification multi-classes.

### 2. Entraînement du Modèle
- Utilisation d'un modèle pré-entraîné BERT (de la bibliothèque Transformers).
- Configuration :
  - Optimiseur : AdamW
  - Fonction de perte : `CategoricalCrossentropy`
  - Taux d'apprentissage ajustable.
  - Batch size et epochs personnalisables.

### 3. Évaluation
- Calcul des métriques (Accuracy, F1-score, Precision, Recall).
- Visualisation des courbes de perte et d'exactitude.
- Création d'une matrice de confusion pour analyser les performances par classe.

### 4. Sauvegarde et Chargement du Modèle
- Modèle sauvegardé au format HDF5.
- Validation que le modèle rechargé reproduit les performances initiales.

---

## Résultats

- **Balanced Accuracy** : 85%+
- **F1-Score** : ~80% pour toutes les classes.
- **Visualisation** : Les courbes montrent une convergence stable avec une légère divergence validation/entraînement, suggérant un surapprentissage potentiel.

---

## Améliorations Potentielles

1. **Avertissement de parallélisme** :
   Ajoutez ce code pour supprimer les avertissements liés au tokenizer :
   ```python
   import os
   os.environ["TOKENIZERS_PARALLELISM"] = "false"
   ```

2. **Surapprentissage** :
   - Introduire une régularisation via Dropout.
   - Utiliser une stratégie d'arrêt anticipé.

3. **Modèles Plus Légers** :
   - Explorer des variantes comme DistilBERT ou MobileBERT pour réduire la complexité.

4. **Évaluation Avancée** :
   - Ajoutez une matrice de confusion pour un aperçu détaillé des prédictions :
     ```python
     from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
     cm = confusion_matrix(test_data.Label, test_predictions)
     ConfusionMatrixDisplay(cm, display_labels=["Class 0", "Class 1", "Class 2", ...]).plot()
     ```
