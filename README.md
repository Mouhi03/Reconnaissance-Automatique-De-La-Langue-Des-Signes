# ASL Real-Time Sign Language Recognition

Real-time recognition of American Sign Language (ASL) letters from a webcam, with word building, spell-checking and text-to-speech.

## Features
- Hand landmark detection with MediaPipe
- Letter classification with a KNN model (and an experimental LSTM model)
- Tools to collect your own dataset and to search the best `k`
- Live recognition window (OpenCV) and a Flask + WebSocket web interface
- Word suggestions (spell-checker) and speech output

## Architecture
`collect_data.py` records landmarks to `dataset_camera.csv`, `find_best_k.py` evaluates KNN, the trained model is saved in `model_knn_asl.pkl` / `label_encoder.pkl`, then `main_camera.py` (desktop) or `server.py` + `templates/index.html` (web) run the live prediction.

## Tech stack
Python · OpenCV · MediaPipe · scikit-learn · TensorFlow / Keras · Flask-SocketIO · pyttsx3

## Getting started
```bash
python -m venv venv && venv\Scripts\activate   # Windows
pip install -r requirements.txt
python main_camera.py      # desktop recognition
python server.py           # web UI on http://localhost:5000
python collect_data.py     # record your own samples
```

## Author
**Ahlam Assamsadi**: [Portfolio](https://ahlam-assamsadi.vercel.app) · [GitHub](https://github.com/assamsadi)

---

# Reconnaissance de la langue des signes (ASL) en temps réel

Reconnaissance en temps réel des lettres de la langue des signes américaine (ASL) via la webcam, avec construction de mots, correction orthographique et synthèse vocale.

## Fonctionnalités
- Détection des points de la main avec MediaPipe
- Classification des lettres par KNN (et un modèle LSTM expérimental)
- Outils de collecte de données et de recherche du meilleur `k`
- Fenêtre de reconnaissance (OpenCV) et interface web Flask + WebSocket
- Suggestions de mots et lecture à voix haute

## Architecture
`collect_data.py` enregistre les points dans `dataset_camera.csv`, `find_best_k.py` évalue le KNN, le modèle est sauvegardé dans `model_knn_asl.pkl`, puis `main_camera.py` (bureau) ou `server.py` (web) exécutent la prédiction en direct.

## Technologies
Python · OpenCV · MediaPipe · scikit-learn · TensorFlow / Keras · Flask-SocketIO · pyttsx3

## Installation et lancement
Voir *Getting started* ci-dessus.

## Auteure
**Ahlam Assamsadi** : [Portfolio](https://ahlam-assamsadi.vercel.app) · [GitHub](https://github.com/assamsadi)
