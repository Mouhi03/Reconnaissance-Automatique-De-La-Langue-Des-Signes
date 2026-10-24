"""
collect_data.py — Collecte de données ASL avec ta caméra
=========================================================
Instructions :
  - Appuie sur la lettre du clavier (A, B, C...) pour démarrer la collecte
  - Maintiens le signe pendant 3 secondes → 300 exemples collectés automatiquement
  - Appuie sur Q pour quitter et sauvegarder

Features extraites : 63 landmarks normalisés + 15 angles de doigts = 78 features
"""

import cv2
import numpy as np
import mediapipe.python.solutions.hands as mp_hands
import mediapipe.python.solutions.drawing_utils as mp_drawing
import csv
import os
import time

# ── Configuration ─────────────────────────────────────────────────────────────
LABELS            = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ") + ["del", "space"]
SAMPLES_PER_LABEL = 300
OUTPUT_FILE       = "dataset_camera.csv"
# ──────────────────────────────────────────────────────────────────────────────

# Charger les données existantes
existing_counts = {label: 0 for label in LABELS}
if os.path.exists(OUTPUT_FILE):
    with open(OUTPUT_FILE, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if row and row[-1] in existing_counts:
                existing_counts[row[-1]] += 1
    print(f"✅ Fichier existant chargé — {sum(existing_counts.values())} exemples déjà collectés")

# ── Caméra ────────────────────────────────────────────────────────────────────
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M','J','P','G'))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)

for _ in range(30):
    cap.read()

print("✅ Caméra ouverte")
print("👉 Appuie sur une lettre (A-Z) ou D=del, ESPACE=space pour collecter")
print("👉 Q pour quitter et sauvegarder\n")

# ── MediaPipe ─────────────────────────────────────────────────────────────────
mp_h    = mp_hands.Hands(static_image_mode=False, max_num_hands=1,
                         min_detection_confidence=0.7, min_tracking_confidence=0.5)
mp_draw = mp_drawing

# ── Triplets d'articulations pour calculer les angles ─────────────────────────
# Chaque tuple = (point1, sommet, point2) → angle calculé au sommet
# 3 angles par doigt × 5 doigts = 15 angles
ANGLE_TRIPLETS = [
    # Pouce
    (0, 1, 2), (1, 2, 3), (2, 3, 4),
    # Index
    (0, 5, 6), (5, 6, 7), (6, 7, 8),
    # Majeur
    (0, 9, 10), (9, 10, 11), (10, 11, 12),
    # Annulaire
    (0, 13, 14), (13, 14, 15), (14, 15, 16),
    # Auriculaire
    (0, 17, 18), (17, 18, 19), (18, 19, 20),
]

# ── Fonctions de feature extraction ──────────────────────────────────────────

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
    """
    Calcule 15 angles d'articulation des doigts (en degrés, normalisés entre 0 et 1).
    Un angle = 0° → doigt plié, 180° → doigt tendu.
    """
    pts = np.array(row).reshape(21, 3)
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
        angles.append(angle / 180.0)   # Normaliser entre 0 et 1
    return angles

def extract_features(raw_row):
    """
    Extrait les features complètes :
    63 landmarks normalisés + 15 angles = 78 features au total
    """
    landmarks = normalize_landmarks(raw_row)   # 63 valeurs
    angles    = compute_angles(raw_row)         # 15 valeurs
    return landmarks + angles                   # 78 valeurs

# ── État ──────────────────────────────────────────────────────────────────────
current_label   = None
collecting      = False
collect_count   = 0
collected_data  = []
countdown_start = None
COUNTDOWN_SEC   = 3

# ── Boucle principale ─────────────────────────────────────────────────────────
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame     = cv2.flip(frame, 1)
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image_rgb.flags.writeable = False
    results   = mp_h.process(image_rgb)
    image_rgb.flags.writeable = True

    hand_detected = False

    if results.multi_hand_landmarks:
        hand_detected = True
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Extraire les landmarks bruts
            raw_row = []
            for lm in hand_landmarks.landmark:
                raw_row.extend([lm.x, lm.y, lm.z])

            # Extraire les 78 features (landmarks + angles)
            features = extract_features(raw_row)

            # Collecte active
            if collecting and current_label is not None:
                if collect_count < SAMPLES_PER_LABEL:
                    collected_data.append(features + [current_label])
                    collect_count += 1
                else:
                    with open(OUTPUT_FILE, 'a', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerows(collected_data)
                    existing_counts[current_label] += collect_count
                    print(f"✅ {current_label} : {collect_count} exemples sauvegardés "
                          f"(total {existing_counts[current_label]})")
                    collected_data  = []
                    collect_count   = 0
                    collecting      = False
                    current_label   = None

    # ── HUD ───────────────────────────────────────────────────────────────────
    h, w = frame.shape[:2]

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h - 80), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

    if collecting and current_label:
        progress = collect_count / SAMPLES_PER_LABEL
        bar_w    = int((w - 40) * progress)
        cv2.rectangle(frame, (20, h - 30), (w - 20, h - 12), (60, 60, 60), -1)
        cv2.rectangle(frame, (20, h - 30), (20 + bar_w, h - 12), (0, 200, 100), -1)
        cv2.putText(frame, f"Collecte [{current_label}] : {collect_count}/{SAMPLES_PER_LABEL}",
                    (20, h - 38), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 120), 2)

    elif countdown_start and current_label:
        elapsed   = time.time() - countdown_start
        remaining = COUNTDOWN_SEC - elapsed
        if remaining > 0:
            cv2.putText(frame, f"Prepare le signe [{current_label}]... {remaining:.1f}s",
                        (20, h - 38), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
        else:
            collecting      = True
            countdown_start = None
    else:
        done    = sum(1 for v in existing_counts.values() if v >= SAMPLES_PER_LABEL)
        pending = [l for l, v in existing_counts.items() if v < SAMPLES_PER_LABEL]
        cv2.putText(frame,
                    f"Lettres completes : {done}/{len(LABELS)}  | Suivante : {pending[0] if pending else 'TOUT FAIT!'}",
                    (10, h - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
        cv2.putText(frame, "Appuie sur une lettre pour collecter | Q = quitter",
                    (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1)

    color = (0, 255, 0) if hand_detected else (0, 0, 255)
    cv2.circle(frame, (w - 30, 30), 12, color, -1)

    cv2.imshow('Collecte ASL - Ahlam', frame)
    
    key = cv2.waitKey(1) & 0xFF

    # 1. Quitter
    if key == ord('q'):
        if collected_data:
            with open(OUTPUT_FILE, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerows(collected_data)
            print(f"✅ Données partielles sauvegardées")
        break

    # 2. Détecter la touche pressée (si on ne collecte pas déjà)
    elif key != 255 and not collecting:
        char = None
        
        if key == ord('1'):
            char = 'space'
        elif key == ord('2'):
            char = 'del'
        elif ord('a') <= key <= ord('z'):
            char = chr(key).upper()

        # 3. Lancer le compte à rebours si la touche est valide
        if char in LABELS:
            current_label   = char
            countdown_start = time.time()
            collect_count   = 0
            collected_data  = []
            print(f"⏳ Prépare le signe [{char}]...")

cap.release()
cv2.destroyAllWindows()
mp_h.close()

total = sum(existing_counts.values())
print(f"\n📊 Total collecté : {total} exemples")
print(f"📁 Fichier : {OUTPUT_FILE}")
print(f"📐 Features par exemple : 78 (63 landmarks + 15 angles)")
print("\nLettres complètes :")
for label, count in existing_counts.items():
    status = "✅" if count >= SAMPLES_PER_LABEL else f"⚠️  {count}/{SAMPLES_PER_LABEL}"
    print(f"  {label} : {status}")