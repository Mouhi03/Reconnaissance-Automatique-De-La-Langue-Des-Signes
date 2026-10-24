# Reconnaissance de la langue des signes (ASL) en temps réel

Une application de vision par ordinateur qui lit l'alphabet dactylologique de la langue des signes américaine (ASL) via la webcam et le convertit en texte en temps réel. Les lettres reconnues forment des mots, qui sont vérifiés par un dictionnaire, corrigés si nécessaire, et peuvent être lus à voix haute, depuis une fenêtre de bureau ou une interface web.

---

## Table des matières
1. [Présentation](#présentation)
2. [Fonctionnalités](#fonctionnalités)
3. [Fonctionnement](#fonctionnement)
4. [Structure du projet](#structure-du-projet)
5. [Technologies](#technologies)
6. [Installation](#installation)
7. [Utilisation](#utilisation)
8. [Jeu de données](#jeu-de-données)
9. [Entraînement et évaluation du modèle](#entraînement-et-évaluation-du-modèle)
10. [Commandes](#commandes)
11. [Configuration](#configuration)
12. [Limites connues](#limites-connues)
13. [Pistes d'amélioration](#pistes-damélioration)
14. [Auteur](#auteur)

---

## Présentation

La dactylologie est la partie de l'ASL qui sert à épeler les noms, les lieux et les mots n'ayant pas de signe propre. Ce projet détecte la main dans chaque image de la webcam, en extrait une description numérique compacte de la posture, puis la classe en une lettre. Le classifieur travaille sur les *points de repère* de la main plutôt que sur les pixels : il est donc léger, rapide et assez robuste face à l'éclairage, à l'arrière-plan et à la couleur de peau.

Le projet couvre toute la chaîne de traitement : **collecte des données → extraction de caractéristiques → entraînement du modèle → prédiction en direct → interface utilisateur**.

## Fonctionnalités

- **Suivi de la main avec MediaPipe** : 21 points de repère 3D détectés à chaque image.
- **Classification par KNN** : modèle des K plus proches voisins entraîné sur un jeu de données personnalisé ; un modèle LSTM expérimental est également fourni.
- **Outil de collecte de données** : `collect_data.py` permet d'enregistrer ses propres échantillons afin d'adapter le modèle à sa main, à sa caméra et à son environnement.
- **Recherche automatique du meilleur K** : `find_best_k.py` teste les valeurs de K de 1 à 15, affiche la précision de chacune et sauvegarde le meilleur modèle.
- **Deux modes d'exécution** :
  - une fenêtre de bureau réalisée avec OpenCV ;
  - une interface web réalisée avec Flask et WebSocket (Flask-SocketIO).
- **Construction de mots avec filtre de stabilité** : une lettre n'est validée qu'après 20 images consécutives avec la même prédiction, ce qui évite les lettres accidentelles.
- **Correction orthographique et suggestions** : chaque mot est vérifié et une correction est proposée s'il n'est pas reconnu.
- **Définitions de mots** (interface web) : les mots anglais valides sont recherchés via une API de dictionnaire publique.
- **Suggestion de la lettre suivante** : une table de bigrammes propose une lettre probable.
- **Synthèse vocale** : les lettres et les mots sont prononcés à voix haute avec `pyttsx3`.
- **Historique des détections** : les dernières lettres reconnues sont affichées avec leur confiance, leur heure et leur équivalent en alphabet arabe.

## Fonctionnement

```
Image de la webcam
     │
     ▼
MediaPipe Hands ──► 21 points de repère (x, y, z)
     │
     ▼
Extraction de caractéristiques
  • 63 coordonnées normalisées (normalisation par boîte englobante)
  • 15 angles d'articulation des doigts
  = 78 caractéristiques
     │
     ▼
Classifieur KNN ──► lettre + confiance
     │
     ▼
Vote majoritaire sur les 20 dernières images (stabilité)
     │
     ▼
Construction du mot ──► correction ──► suggestions ──► lecture vocale
```

**Extraction de caractéristiques.** Les positions brutes des points dépendent de l'endroit où se trouve la main dans l'image et de sa distance à la caméra. Pour supprimer cette dépendance :

1. **Normalisation par boîte englobante** : chaque point est recalculé par rapport à la boîte englobante de la main, ce qui rend les valeurs indépendantes de la position et de la taille (63 valeurs : 21 points × 3 axes).
2. **Angles d'articulation** : pour chaque doigt, trois angles sont calculés entre les articulations successives (5 doigts × 3 = 15 valeurs), ramenés entre 0 et 1. Les angles sont naturellement insensibles à la translation et à l'échelle et décrivent très bien la flexion des doigts.

Les mêmes fonctions d'extraction sont utilisées pendant la collecte et pendant la prédiction. C'est indispensable : toute différence entre les deux dégraderait la précision.

**Lissage des prédictions.** Une prédiction image par image peut osciller. Les 20 dernières prédictions sont conservées dans une fenêtre glissante, et une lettre n'est ajoutée au mot que si elle est majoritaire et reste stable sur toute la fenêtre. Retirer la main du champ de la caméra réinitialise l'état, ce qui permet de saisir deux fois la même lettre de suite.

## Structure du projet

```
asl-sign-language-recognition/
├── collect_data.py        # Enregistrement des échantillons de chaque lettre dans un CSV
├── find_best_k.py         # Évaluation de K = 1..15, entraînement et sauvegarde du meilleur KNN
├── main_camera.py         # Reconnaissance en direct sur le bureau (fenêtre OpenCV)
├── server.py              # Serveur Flask + SocketIO pour l'interface web
├── templates/
│   └── index.html         # Interface web (vidéo, mot, historique, suggestions)
├── test_camera.py         # Test rapide du bon fonctionnement de la webcam
├── dataset_camera.csv     # Jeu de données collecté (78 caractéristiques + étiquette par ligne)
├── model_knn_asl.pkl      # Modèle KNN entraîné
├── model_lstm_asl.h5      # Modèle LSTM expérimental
├── label_encoder.pkl      # Encodeur des classes
├── requirements.txt       # Dépendances Python
└── .gitignore
```

## Technologies

| Domaine | Outils |
|---------|--------|
| Langage | Python |
| Vision par ordinateur | OpenCV, MediaPipe Hands |
| Apprentissage automatique | scikit-learn (KNN), TensorFlow / Keras (LSTM, expérimental), NumPy, joblib |
| Web | Flask, Flask-SocketIO (WebSocket), HTML / CSS / JavaScript |
| Outils linguistiques | pyspellchecker, pyttsx3 (synthèse vocale), Requests (API de dictionnaire) |

## Installation

**Prérequis**
- Python 3.9 ou supérieur (une version compatible avec vos versions de MediaPipe et TensorFlow)
- Une webcam fonctionnelle
- Windows est la plateforme principale testée (la caméra est ouverte avec le backend DirectShow, voir [Limites connues](#limites-connues))

**Étapes**

```bash
# 1. Récupérer le projet
git clone <url-de-votre-depot>
cd asl-sign-language-recognition

# 2. Créer et activer un environnement virtuel
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS / Linux

# 3. Installer les dépendances
pip install -r requirements.txt
```

## Utilisation

**1. Vérifier la caméra (facultatif)**
```bash
python test_camera.py          # appuyer sur Q pour fermer la fenêtre
```

**2. Lancer l'application de bureau**
```bash
python main_camera.py
```

**3. Lancer l'application web**
```bash
python server.py
```
Ouvrez ensuite **http://localhost:5000** dans votre navigateur et cliquez sur *Start*.

**4. Enregistrer vos propres données (facultatif)**
```bash
python collect_data.py
```
Appuyez sur une touche-lettre et maintenez le signe correspondant devant la caméra : 300 échantillons sont capturés automatiquement. Appuyez sur `Q` pour quitter et sauvegarder. Les nouveaux échantillons sont ajoutés à `dataset_camera.csv`.

**5. Réentraîner le modèle**
```bash
python find_best_k.py
```
Le script teste plusieurs valeurs de K, affiche la précision de chacune et remplace `model_knn_asl.pkl` par la meilleure configuration entraînée sur l'ensemble des données.

## Jeu de données

- **Fichier :** `dataset_camera.csv`
- **Format d'une ligne :** 78 caractéristiques numériques suivies de l'étiquette de la lettre (79 colonnes)
- **Taille :** 7 200 échantillons, soit 300 par classe
- **Classes incluses :** A, B, C, E, F, G, H, I, J, K, L, M, N, O, P, R, S, T, U, V, W, X, Y, Z (24 lettres)
- **Absentes du jeu actuel :** D et Q (`collect_data.py` permet de les enregistrer, ainsi que les gestes `del` et `space`)

Les données ont été enregistrées avec une vraie webcam : le modèle fonctionne donc mieux dans des conditions proches de celles de la collecte. Pour de meilleurs résultats dans un nouveau contexte (autre caméra, autre main, autre éclairage), enregistrez des échantillons supplémentaires puis réentraînez le modèle.

## Entraînement et évaluation du modèle

`find_best_k.py` suit un flux d'apprentissage supervisé classique :

1. Chargement du jeu de données et séparation **80 % entraînement / 20 % test**, stratifiée par classe, avec une graine aléatoire fixe pour la reproductibilité.
2. Entraînement d'un classifieur KNN pour chaque valeur de K de 1 à 15, avec la distance euclidienne et un **vote pondéré par la distance** (les voisins les plus proches comptent davantage).
3. Affichage de la précision sur le jeu de test pour chaque K et sélection du meilleur.
4. Réentraînement sur l'ensemble des données avec ce K et sauvegarde du modèle dans `model_knn_asl.pkl`.

Le KNN a été choisi parce qu'il est simple, ne demande pas d'entraînement long, s'adapte instantanément à l'ajout de nouveaux échantillons et donne de bons résultats sur des caractéristiques de points de repère compactes et bien normalisées.

## Commandes

**Fenêtre de bureau (`main_camera.py`)**

| Touche | Action |
|--------|--------|
| `Espace` | Insérer un espace |
| `Retour arrière` | Effacer la dernière lettre |
| `C` | Effacer tout le mot |
| `Q` | Quitter |

**Collecte de données (`collect_data.py`)**

| Touche | Action |
|--------|--------|
| `A`–`Z` | Démarrer l'enregistrement des échantillons de cette lettre |
| `Espace` | Enregistrer le geste `space` |
| `Q` | Quitter et sauvegarder |

**Interface web (`server.py`)** : utilisez les boutons à l'écran pour démarrer ou arrêter la caméra, supprimer la dernière lettre ou effacer le mot.

## Configuration

Les principaux paramètres sont définis en haut des scripts :

| Paramètre | Valeur par défaut | Signification |
|-----------|-------------------|---------------|
| `CONFIRM_FRAMES` | 20 | Nombre d'images stables consécutives nécessaires pour valider une lettre |
| `SAMPLES_PER_LABEL` | 300 | Échantillons enregistrés par lettre dans `collect_data.py` |
| `min_detection_confidence` | 0,5 | Seuil de détection de la main de MediaPipe |
| `min_tracking_confidence` | 0,5 | Seuil de suivi de la main de MediaPipe |
| Résolution de la caméra | 640 × 480 à 30 images/s | Paramètres de capture |

Diminuez `CONFIRM_FRAMES` pour une saisie plus rapide, augmentez-le pour réduire les lettres accidentelles.

## Limites connues

- **Lettres statiques uniquement.** Les lettres qui impliquent un mouvement (comme J et Z) sont classées à partir d'une seule posture, ce qui limite leur précision. Le modèle LSTM expérimental vise à corriger ce point.
- **Une seule main.** Une seule main est suivie à la fois.
- **Caméra orientée Windows.** Le code ouvre la caméra avec `cv2.CAP_DSHOW` ; sur macOS ou Linux, supprimez cet argument (`cv2.VideoCapture(0)`).
- **Classes manquantes.** Le jeu de données fourni ne contient pas encore D et Q.
- **Fonctions en ligne.** Les définitions de mots nécessitent une connexion Internet ; tout le reste fonctionne hors ligne.
- **Langue du correcteur.** L'interface web utilise un dictionnaire anglais, tandis que le script de bureau est configuré pour le français ; adaptez `SpellChecker(language=...)` selon votre usage.

## Pistes d'amélioration

- Ajouter les lettres manquantes (D, Q) et les gestes `space` / `del` au jeu de données
- Entraîner et intégrer le modèle LSTM pour les lettres avec mouvement
- Prendre en charge les signes à deux mains et les signes de mots entiers
- Gérer la caméra de façon multiplateforme
- Proposer une interface plus soignée avec un panneau de calibrage et de réglages
- Générer en option une matrice de confusion et un rapport de précision par lettre
