import os
import librosa
import numpy as np
import kagglehub
import soundfile as sf
import tkinter as tk
import pickle
from tkinter import filedialog
from pydub import AudioSegment

from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# ==========================================
# 1. FILE SELECTION
# ==========================================
def select_file():
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)

    print("📂 Select an audio file...")
    file_selected = filedialog.askopenfilename(
        title="Select Audio for Emotion Analysis",
        filetypes=[("Audio Files", "*.wav *.mp3 *.m4a *.aac *.flac")]
    )
    root.destroy()
    return file_selected


# ==========================================
# 2. FEATURE EXTRACTION
# ==========================================
def extract_feature(file_path):
    temp_wav = "temp_processing.wav"

    try:
        if not file_path.lower().endswith(".wav"):
            audio = AudioSegment.from_file(file_path)
            audio = audio.set_channels(1).set_frame_rate(22050)
            audio.export(temp_wav, format="wav")
            file_path = temp_wav

        X, sr = sf.read(file_path, dtype="float32")
        if X.ndim > 1:
            X = np.mean(X, axis=1)

        mfcc = librosa.feature.mfcc(y=X, sr=sr, n_mfcc=40)
        delta = librosa.feature.delta(mfcc)
        delta2 = librosa.feature.delta(mfcc, order=2)

        features = np.hstack([
            np.mean(mfcc, axis=1),
            np.mean(delta, axis=1),
            np.mean(delta2, axis=1)
        ])

        return features

    except Exception as e:
        print(f"❌ Feature extraction failed: {e}")
        return None

    finally:
        if os.path.exists(temp_wav):
            os.remove(temp_wav)


# ==========================================
# 3. MODEL LOADING / TRAINING
# ==========================================
def get_ready_model():
    model_file = "emotion_model.pkl"

    if os.path.exists(model_file):
        print("🧠 Loading trained model...")
        with open(model_file, "rb") as f:
            return pickle.load(f)

    print("📥 Downloading RAVDESS dataset...")
    path = kagglehub.dataset_download("uwrfkaggler/ravdess-emotional-speech-audio")

    emotion_map = {
        "01": "neutral",
        "02": "calm",
        "03": "happy",
        "04": "sad",
        "05": "angry",
        "06": "fearful",
        "07": "disgust",
        "08": "surprised"
    }

    X, y = [], []

    print("📊 Extracting features...")
    for root, _, files in os.walk(path):
        for file in files:
            if file.endswith(".wav"):
                parts = file.split("-")
                if len(parts) > 2 and parts[2] in emotion_map:
                    feat = extract_feature(os.path.join(root, file))
                    if feat is not None:
                        X.append(feat)
                        y.append(emotion_map[parts[2]])

    X = np.array(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("mlp", MLPClassifier(
            hidden_layer_sizes=(256, 128),
            activation="relu",
            solver="adam",
            max_iter=800,
            alpha=0.0005,
            random_state=42
        ))
    ])

    print("🧠 Training model...")
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"✅ Validation Accuracy: {acc*100:.2f}%")

    with open(model_file, "wb") as f:
        pickle.dump(model, f)

    print("💾 Model saved.")
    return model


# ==========================================
# 4. MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    user_file = select_file()

    if not user_file:
        print("⚠️ No file selected.")
        exit()

    print(f"🎯 File Selected: {os.path.basename(user_file)}")

    model = get_ready_model()

    print("🔎 Analyzing emotion...")
    features = extract_feature(user_file)

    if features is None:
        print("❌ Could not process file.")
        exit()

    probs = model.predict_proba([features])[0]
    emotions = model.classes_

    top_idx = np.argmax(probs)
    emotion = emotions[top_idx]
    confidence = probs[top_idx] * 100

    print("\n" + "=" * 35)
    print("📊 EMOTION ANALYSIS RESULT")
    print(f"EMOTION     : {emotion.upper()}")
    print(f"CONFIDENCE  : {confidence:.2f}%")
    print("=" * 35)
