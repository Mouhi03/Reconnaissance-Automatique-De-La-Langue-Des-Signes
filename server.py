import os
import warnings
import numpy as np
import cv2
import joblib
import mediapipe.python.solutions.hands as mp_hands
import mediapipe.python.solutions.drawing_utils as mp_drawing
from collections import deque
from flask import Flask, render_template
from flask_socketio import SocketIO
from spellchecker import SpellChecker
import pyttsx3
import threading
import base64
import datetime
import requests

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings("ignore")

app      = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')
spell    = SpellChecker()

# ── Bigrammes
BIGRAMS = {
    'A':'N','B':'E','C':'H','D':'E','E':'R',
    'F':'O','G':'H','H':'E','I':'N','J':'U',
    'K':'E','L':'L','M':'E','N':'D','O':'N',
    'P':'R','Q':'U','R':'E','S':'T','T':'H',
    'U':'S','V':'E','W':'H','X':'P','Y':'O','Z':'E'
}

def suggest_next_letter(word):
    if not word:
        return ''
    return BIGRAMS.get(word[-1].upper(), '')

# ── Définition
def get_word_definition(word):
    try:
        r = requests.get(
            f"https://api.dictionaryapi.dev/api/v2/entries/en/{word.lower()}",
            timeout=3
        )
        if r.status_code == 200:
            return r.json()[0]['meanings'][0]['definitions'][0]['definition']
    except:
        pass
    return None

# ── Vocal
def speak(text):
    def _speak():
        engine = pyttsx3.init()
        engine.setProperty('rate', 160)
        for voice in engine.getProperty('voices'):
            if "english" in voice.name.lower():
                engine.setProperty('voice', voice.id)
                break
        engine.say(text)
        engine.runAndWait()
    threading.Thread(target=_speak, daemon=True).start()

# ── Vérification mot
def check_word(word):
    word_clean = word.lower().strip()
    if len(word_clean) < 2:
        return '', '', ''
    if spell.known([word_clean]):
        definition = get_word_definition(word_clean)
        if definition:
            speak(f"{word}. {definition}")
            return f"✅ '{word}' — {definition}", 'green', definition
        else:
            speak(word)
            return f"✅ '{word}' est un mot anglais valide !", 'green', ''
    else:
        correction = spell.correction(word_clean)
        if correction and correction != word_clean:
            return f"❌ '{word}' ne signifie rien. Vouliez-vous dire '{correction.upper()}' ?", 'orange', ''
        return f"❌ '{word}' ne signifie rien.", 'red', ''

# ── Modèle
knn = joblib.load('model_knn_asl.pkl')
le  = joblib.load('label_encoder.pkl')

# ── Alphabet arabe
alphabet_ar = {
    'A':'أ','B':'ب','C':'ج','D':'د','E':'إ','F':'ف',
    'G':'ج','H':'هـ','I':'ي','J':'ج','K':'ك','L':'ل',
    'M':'م','N':'ن','O':'و','P':'ب','Q':'ق','R':'ر',
    'S':'س','T':'ت','U':'و','V':'ف','W':'و','X':'كس',
    'Y':'ي','Z':'ز'
}

# ── Features
ANGLE_TRIPLETS = [
    (0,1,2),(1,2,3),(2,3,4),
    (0,5,6),(5,6,7),(6,7,8),
    (0,9,10),(9,10,11),(10,11,12),
    (0,13,14),(13,14,15),(14,15,16),
    (0,17,18),(17,18,19),(18,19,20),
]

def normalize_landmarks(row):
    row_np = np.array(row).reshape(21, 3)
    min_x, max_x = np.min(row_np[:,0]), np.max(row_np[:,0])
    min_y, max_y = np.min(row_np[:,1]), np.max(row_np[:,1])
    width  = max_x - min_x
    height = max_y - min_y
    normalized = []
    for x, y, z in row_np:
        normalized.extend([
            (x - min_x) / width  if width  > 0 else 0,
            (y - min_y) / height if height > 0 else 0,
            z / width             if width  > 0 else 0
        ])
    return normalized

def compute_angles(row):
    pts = np.array(row).reshape(21, 3)
    angles = []
    for p1_idx, vertex_idx, p2_idx in ANGLE_TRIPLETS:
        v1 = pts[p1_idx] - pts[vertex_idx]
        v2 = pts[p2_idx] - pts[vertex_idx]
        n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if n1 > 0 and n2 > 0:
            angle = np.degrees(np.arccos(np.clip(np.dot(v1,v2)/(n1*n2),-1,1)))
        else:
            angle = 0.0
        angles.append(angle / 180.0)
    return angles

def extract_features(raw_row):
    return normalize_landmarks(raw_row) + compute_angles(raw_row)

# ── État global
state = {
    'word':              '',
    'prediction':        '',
    'confidence':        0.0,
    'stable_count':      0,
    'last_added':        '',
    'det_count':         0,
    'history':           [],
    'running':           False,
    'word_status':       '',
    'word_status_color': '',
    'next_letter':       '',
    'definition':        ''
}

CONFIRM_FRAMES  = 20
prediction_hist = deque(maxlen=CONFIRM_FRAMES)
cap             = None
camera_thread   = None

def camera_loop():
    global cap
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as hands:
        while state['running']:
            ret, frame = cap.read()
            if not ret:
                continue

            frame     = cv2.flip(frame, 1)
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image_rgb.flags.writeable = False
            results   = hands.process(image_rgb)
            image_rgb.flags.writeable = True

            prediction = ''
            confidence = 0.0

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
                    )
                    raw_row = []
                    for lm in hand_landmarks.landmark:
                        raw_row.extend([lm.x, lm.y, lm.z])

                    features   = extract_features(raw_row)
                    X          = np.array(features).reshape(1, -1)
                    proba      = knn.predict_proba(X)[0]
                    pred_idx   = np.argmax(proba)
                    confidence = float(proba[pred_idx])
                    prediction = str(knn.classes_[pred_idx]).upper()

                    prediction_hist.append(prediction)
                    smoothed = max(set(prediction_hist), key=list(prediction_hist).count)

                    if smoothed == prediction:
                        state['stable_count'] += 1
                    else:
                        state['stable_count'] = 0

                    if (state['stable_count'] >= CONFIRM_FRAMES and
                            smoothed != state['last_added']):
                        state['word']       += smoothed
                        state['last_added']  = smoothed
                        state['stable_count'] = 0
                        state['det_count']  += 1

                        # 🔊 Vocal lettre
                        speak(smoothed)

                        # ✅ Vérification mot
                        status, color, definition = check_word(state['word'])
                        state['word_status']       = status
                        state['word_status_color'] = color
                        state['definition']        = definition
                        state['next_letter']       = suggest_next_letter(state['word'])

                        # Historique
                        state['history'].insert(0, {
                            'letter': str(smoothed),
                            'arabic': str(alphabet_ar.get(smoothed, smoothed)),
                            'conf':   f"{confidence:.0%}",
                            'time':   datetime.datetime.now().strftime('%H:%M:%S')
                        })
                        state['history'] = state['history'][:10]

                state['prediction'] = prediction
                state['confidence'] = confidence

            else:
                state['stable_count'] = 0
                state['last_added']   = ''
                prediction_hist.clear()
                state['prediction'] = ''
                state['confidence'] = 0.0

            # Encoder frame
            _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
            frame_b64 = base64.b64encode(buffer).decode('utf-8')

            socketio.emit('frame', {
                'image':             frame_b64,
                'prediction':        state['prediction'],
                'arabic':            alphabet_ar.get(state['prediction'], ''),
                'confidence':        round(state['confidence'] * 100),
                'stability':         round((state['stable_count'] / CONFIRM_FRAMES) * 100),
                'word':              state['word'],
                'det_count':         state['det_count'],
                'history':           state['history'],
                'word_status':       state['word_status'],
                'word_status_color': state['word_status_color'],
                'next_letter':       state['next_letter'],
                'definition':        state['definition']
            })

    cap.release()

@app.route('/')
def index():
    return render_template('index.html')

@socketio.on('start')
def handle_start():
    global camera_thread
    if not state['running']:
        state['running'] = True
        camera_thread = threading.Thread(target=camera_loop, daemon=True)
        camera_thread.start()

@socketio.on('stop')
def handle_stop():
    state['running'] = False

@socketio.on('clear_word')
def handle_clear():
    state['word']              = ''
    state['last_added']        = ''
    state['word_status']       = ''
    state['word_status_color'] = ''
    state['next_letter']       = ''
    state['definition']        = ''

@socketio.on('delete_letter')
def handle_delete():
    state['word']       = state['word'][:-1]
    state['last_added'] = ''
    status, color, definition     = check_word(state['word'])
    state['word_status']          = status
    state['word_status_color']    = color
    state['definition']           = definition
    state['next_letter']          = suggest_next_letter(state['word'])

if __name__ == '__main__':
    socketio.run(app, debug=False, host='0.0.0.0', port=5000)