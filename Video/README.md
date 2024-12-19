# Emotion Detection Using YOLO  

## Description  
Ce projet vise à détecter et classifier les émotions humaines en temps réel en utilisant le modèle YOLO (You Only Look Once). YOLO est reconnu pour sa rapidité et son efficacité, ce qui le rend idéal pour les besoins en temps réel comme la sécurité, le service client et la santé mentale.  

## Pourquoi YOLO ?  
- Détection rapide et efficace.  
- Convient aux applications en temps réel.  
- Permet une reconnaissance émotionnelle instantanée sur des flux vidéo en direct.  

---

## Objectifs  
- Détecter et classifier les émotions des individus en temps réel.  
- Atteindre un niveau précis de performance avec un mAP (mean Average Precision) de 0.78.  

---

## Structure des Données  
- **Fichiers d'entraînement :** 17 101  
- **Fichiers de validation :** 5 406  
- **Fichiers de test :** 2 755  

### Préparation des Données  
1. Création d'un dossier `yolo-data`.  
2. Copie des dossiers `val` et `test`.  
3. Équilibrage du dossier `train`.  
4. Mise à jour du chemin des données dans `data.yaml`.  

---

## Hyperparamètres  
### Optimisation avec Optuna :  
- **Nombre de combinaisons testées :** 20  
- **Objectif :** Maximiser le mAP sur le jeu de validation.  
- **Hyperparamètres explorés :**  
  - Taux d’apprentissage : entre `0.0001` et `0.1` (log-uniform).  
  - Momentum : entre `0.8` et `0.98`.  
  - Décroissance de poids : entre `0.000001` et `0.001` (log-uniform).  

### Meilleure configuration trouvée :  
- `lr0` : 0.000104  
- `momentum` : 0.881  
- `weight_decay` : 7.27e-6  

---

## Entraînement du Modèle  
- **Nombre d'époques :** 100 avec un arrêt anticipé après 7 époques sans amélioration.  
- **Framework utilisé :** YOLO.  
- **Suivi automatique :** métriques de perte et mAP enregistrées pendant l'entraînement.  

---

## Évaluation du Modèle  
- **Résultat obtenu :** mAP = 0.78  

---

## Tests  
- Démonstration locale avec une application en temps réel.  
- Fichiers de test disponibles dans le répertoire correspondant.  

---

## Arborescence des Fichiers  
```plaintext
.
├── Result/                        # Résultats d'entraînement (modèles et métriques)
├── explore-balance-yolo-data.ipynb  # Analyse et équilibrage des données
├── fine-tuning-yolo-model.ipynb     # Optimisation des hyperparamètres
├── train-yolo-model.ipynb           # Script d'entraînement du modèle
├── test-yolo-model.ipynb            # Tests et validation
├── RealTimeTest.py                  # Détection en temps réel via caméra
└── best_params.json                 # Meilleurs hyperparamètres trouvés
```
---

## Installation et Utilisation  

### 1. Installation  
Assurez-vous que les dépendances nécessaires sont installées avant d'exécuter le projet.  

#### Prérequis :  
- Python 3.8 ou plus récent  
- PyTorch installé (compatible avec votre GPU si disponible)  
- Autres dépendances :  
  - `torch`  
  - `numpy`  

#### Étapes d'installation :  
1. **Cloner le projet :**  
   ```bash
   git clone https://github.com/votre-repo/unified-emotion-prediction.git
   cd unified-emotion-prediction
