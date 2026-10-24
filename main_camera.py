import os
import warnings
import joblib
import numpy as np
import cv2
import pyttsx3
import mediapipe.python.solutions.hands as mp_hands
import mediapipe.python.solutions.drawing_utils as mp_drawing
from collections import deque
from spellchecker import SpellChecker

"initialisation "
spell = SpellChecker(language='fr')

# ── Configuration système ─────────────────────────────────────────────────────
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['ABSL_LOGGING_LEVEL']   = '0'
warnings.filterwarnings("ignore", category=UserWarning)

# ── Initialisation voix ───────────────────────────────────────────────────────
engine = pyttsx3.init()
voices = engine.getProperty('voices')
for voice in voices:
    if "english" in voice.name.lower():
        engine.setProperty('voice', voice.id)
        break
engine.setProperty('rate', 160)

# ── Dictionnaire arabe ────────────────────────────────────────────────────────
alphabet_ar = {
    'A': 'أ', 'B': 'ب', 'C': 'ج', 'D': 'د', 'E': 'إ', 'F': 'ف',
    'G': 'ج', 'H': 'هـ', 'I': 'ي', 'J': 'ج', 'K': 'ك', 'L': 'ل',
    'M': 'م', 'N': 'ن', 'O': 'و', 'P': 'ب', 'Q': 'ق', 'R': 'ر',
    'S': 'س', 'T': 'ت', 'U': 'و', 'V': 'ف', 'W': 'و', 'X': 'كس',
    'Y': 'ي', 'Z': 'ز', 'space': 'مسافة', 'del': 'حذف'
}

# ── Chargement du modèle ──────────────────────────────────────────────────────
knn = joblib.load('model_knn_asl.pkl')

# ── Triplets d'articulations pour les angles ──────────────────────────────────
# IDENTIQUE à collect_data.py — très important !
ANGLE_TRIPLETS = [
    (0, 1, 2), (1, 2, 3), (2, 3, 4),       # Pouce
    (0, 5, 6), (5, 6, 7), (6, 7, 8),        # Index
    (0, 9, 10), (9, 10, 11), (10, 11, 12),  # Majeur
    (0, 13, 14), (13, 14, 15), (14, 15, 16),# Annulaire
    (0, 17, 18), (17, 18, 19), (18, 19, 20),# Auriculaire
]





# ── Fonctions de feature extraction ──────────────────────────────────────────
# IDENTIQUES à collect_data.py — très important !

def normalize_landmarks(row):
    """Normalisation Bounding Box — indépendante de la position et taille."""
    row_np   = np.array(row).reshape(21, 3)
    x_coords = row_np[:, 0]
    y_coords = row_np[:, 1]
    min_x, max_x = np.min(x_coords), np.max(x_coords)
    min_y, max_y = np.min(y_coords), np.max(y_coords)
    width  = max_x - min_x
    height = max_y - min_y
    normalized = []
    for x, y, z in row_np:
        norm_x = (x - min_x) / width  if width  > 0 else 0
        norm_y = (y - min_y) / height if height > 0 else 0
        norm_z = z / width             if width  > 0 else 0
        normalized.extend([norm_x, norm_y, norm_z])
    return normalized

def compute_angles(row):
    """Calcule 15 angles d'articulation normalisés entre 0 et 1."""
    pts    = np.array(row).reshape(21, 3)
    angles = []
    for p1_idx, vertex_idx, p2_idx in ANGLE_TRIPLETS:
        p1     = pts[p1_idx]
        vertex = pts[vertex_idx]
        p2     = pts[p2_idx]
        v1     = p1 - vertex
        v2     = p2 - vertex
        norm1  = np.linalg.norm(v1)
        norm2  = np.linalg.norm(v2)
        if norm1 > 0 and norm2 > 0:
            cos_a = np.dot(v1, v2) / (norm1 * norm2)
            angle = np.degrees(np.arccos(np.clip(cos_a, -1.0, 1.0)))
        else:
            angle = 0.0
        angles.append(angle / 180.0)
    return angles

def extract_features(raw_row):
    landmarks = normalize_landmarks(raw_row)  # 63 features
    angles    = compute_angles(raw_row)        # 15 features
    return landmarks + angles                  # 78 features au total
# ── Caméra ────────────────────────────────────────────────────────────────────
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M','J','P','G'))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)

for _ in range(30):
    cap.read()

print("✅ Caméra ouverte")
print("   ESPACE=espace | RETOUR=effacer lettre | C=effacer tout | Q=quitter")

# ── Paramètres ────────────────────────────────────────────────────────────────
CONFIRM_FRAMES = 20
MAX_WORD_LEN   = 30

# ── État ──────────────────────────────────────────────────────────────────────
word            = ""
prediction      = ""
confidence      = 0.0
text_ar         = ""
last_prediction = ""
last_added      = ""
stable_count    = 0
prediction_hist = deque(maxlen=CONFIRM_FRAMES)

# ── Boucle principale ─────────────────────────────────────────────────────────
with mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as hands:

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame     = cv2.flip(frame, 1)
        h, w      = frame.shape[:2]
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False
        results   = hands.process(image_rgb)
        image_rgb.flags.writeable = True

        hand_detected = False

        if results.multi_hand_landmarks:
            hand_detected = True
            for hand_landmarks in results.multi_hand_landmarks:

                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
                )

                # Landmarks bruts
                raw_row = []
                for lm in hand_landmarks.landmark:
                    raw_row.extend([lm.x, lm.y, lm.z])

                # 78 features (identique à collect_data.py)
                features = extract_features(raw_row)
                X        = np.array(features).reshape(1, -1)

                # Prédiction + confiance
                proba      = knn.predict_proba(X)[0]
                pred_idx   = np.argmax(proba)
                confidence = float(proba[pred_idx])
                prediction = str(knn.classes_[pred_idx]).upper()
                text_ar    = alphabet_ar.get(prediction, prediction)

                # Boîte autour de la main
                x_coords = [lm.x for lm in hand_landmarks.landmark]
                y_coords = [lm.y for lm in hand_landmarks.landmark]
                x1 = int(min(x_coords) * w) - 20
                y1 = int(min(y_coords) * h) - 20
                x2 = int(max(x_coords) * w) + 20
                y2 = int(max(y_coords) * h) + 20
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # Voix quand la lettre change
                if prediction != last_prediction:
                    print(f"✅ Predicted: {prediction} | AR: {text_ar}")
                    engine.say(prediction)
                    engine.runAndWait()
                    last_prediction = prediction

                # Lissage + confirmation
                prediction_hist.append(prediction)
                smoothed = max(set(prediction_hist),
                               key=list(prediction_hist).count)

                if smoothed == prediction:
                    stable_count += 1
                else:
                    stable_count = 0

                if (stable_count >= CONFIRM_FRAMES and
                        smoothed != last_added and
                        smoothed not in ['DEL', 'SPACE']):
                    word        += smoothed
                    last_added   = smoothed
                    stable_count = 0
                    if len(word) > MAX_WORD_LEN:
                        word = word[-MAX_WORD_LEN:]

        else:
            stable_count = 0
            last_added   = ""
            prediction_hist.clear()
            prediction   = ""
            confidence   = 0.0
            text_ar      = ""

        # ── Interface graphique ───────────────────────────────────────────────
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, h - 150), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        # Lettre prédite
        if prediction:
            cv2.putText(frame, prediction,
                        (20, h - 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 2.5, (0, 255, 120), 4)

        # Lettre en arabe
        if text_ar:
            cv2.putText(frame, f"AR: {text_ar}",
                        (130, h - 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)

        # Barre de confiance
        bar_x, bar_y = 20, h - 75
        bar_w_max    = 280
        bar_fill     = int(bar_w_max * confidence)
        bar_color    = (0, 220, 100) if confidence >= 0.75 else \
                       (0, 180, 255) if confidence >= 0.50 else (0, 80, 220)

        cv2.rectangle(frame, (bar_x, bar_y),
                      (bar_x + bar_w_max, bar_y + 18), (60, 60, 60), -1)
        cv2.rectangle(frame, (bar_x, bar_y),
                      (bar_x + bar_fill, bar_y + 18), bar_color, -1)
        cv2.putText(frame, f"Confiance : {confidence:.0%}",
                    (bar_x, bar_y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

        # Barre de stabilité
        stab_y    = bar_y + 26
        stab_fill = int(bar_w_max * (stable_count / CONFIRM_FRAMES))
        cv2.rectangle(frame, (bar_x, stab_y),
                      (bar_x + bar_w_max, stab_y + 10), (40, 40, 40), -1)
        cv2.rectangle(frame, (bar_x, stab_y),
                      (bar_x + stab_fill, stab_y + 10), (200, 200, 0), -1)
        cv2.putText(frame, "Stabilite",
                    (bar_x, stab_y - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1)

        # Mot formé
        cv2.putText(frame, f"Mot : {word}",
                    (20, h - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2)
        # Indicateur main
        color = (0, 220, 80) if hand_detected else (0, 0, 200)
        cv2.circle(frame, (w - 25, 25), 10, color, -1)

        cv2.putText(frame, "1=Q  2=ESPACE  3=DEL  Q=quitter",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (180, 180, 180), 1)

        cv2.imshow('ASL Detection - Ahlam', frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break

        # 👉 touche 1 → ajouter Q
        elif key == ord('1'):
            word += 'Q'
            last_added = ''
            print("👉 Ajout manuel : Q")

        # 👉 touche 2 → espace
        elif key == ord('2'):
            word += ' '
            last_added = ''
            print("👉 Ajout manuel : SPACE")

        # 👉 touche 3 → supprimer
        elif key == ord('3'):
            word = word[:-1]
            last_added = ''
            print("👉 Suppression dernière lettre")

cap.release()
cv2.destroyAllWindows()

print(f"\n Caméra fermée. Mot détecté : {word}")

#  Correction automatique avec pyspellchecker
if word.strip() != "":
    mot_corrige = spell.correction(word.lower())
    print(f" Mot corrigé : {mot_corrige.upper()}")
else:
    print("⚠ Aucun mot à corriger")