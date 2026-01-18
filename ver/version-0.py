import os, librosa, numpy as np, kagglehub, soundfile as sf, tkinter as tk, pickle, time
from tkinter import filedialog
from pydub import AudioSegment
from sklearn.neural_network import MLPClassifier

# ==========================================
# 1. ADVANCED FEATURE EXTRACTION
# ==========================================
def extract_feature(file_path):
    temp_wav = "temp_processing.wav"
    try:
        if not file_path.lower().endswith('.wav'):
            audio = AudioSegment.from_file(file_path)
            audio.export(temp_wav, format="wav")
            target = temp_wav
        else:
            target = file_path

        with sf.SoundFile(target) as sound_file:
            X = sound_file.read(dtype="float32")
            sr = sound_file.samplerate
            if len(X.shape) > 1: X = X.mean(axis=1)
            
            # Combine Speech (MFCC), Pitch (Chroma), and Texture (Mel)
            mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sr, n_mfcc=40).T, axis=0)
            chroma = np.mean(librosa.feature.chroma_stft(S=np.abs(librosa.stft(X)), sr=sr).T, axis=0)
            mel = np.mean(librosa.feature.melspectrogram(y=X, sr=sr).T, axis=0)
            return np.hstack((mfccs, chroma, mel[:40]))
    except Exception:
        return None
    finally:
        if os.path.exists(temp_wav): os.remove(temp_wav)

# ==========================================
# 2. UNIVERSAL INTELLIGENCE SYSTEM
# ==========================================
def get_ready_model():
    model_file = "universal_emotion_model.pkl"
    if os.path.exists(model_file):
        print("\n🧠 [SYSTEM] Loading Universal Intelligence...")
        with open(model_file, 'rb') as f: return pickle.load(f)

    print("\n📥 [SYSTEM] Downloading Universal Dataset (First-time setup)...")
    path = kagglehub.dataset_download("uwrfkaggler/ravdess-emotional-speech-audio")
    
    # 01=Neutral, 02=Calm, 03=Happy, 04=Sad, 05=Angry, 06=Fear, 07=Disgust, 08=Surprise
    emotion_map = {
        '01': 'neutral', '02': 'calm', '03': 'happy/laughing', 
        '04': 'sad/crying', '05': 'angry/screaming', '06': 'fearful',
        '07': 'disgust', '08': 'surprised'
    }
    
    x, y = [], []
    files = []
    for root, _, filenames in os.walk(path):
        for filename in filenames:
            if filename.endswith(".wav"): files.append(os.path.join(root, filename))
    
    print(f"📊 [SYSTEM] Training on {len(files)} patterns. Please wait...")
    for file in files:
        parts = os.path.basename(file).split('-')
        if len(parts) >= 3 and parts[2] in emotion_map:
            feat = extract_feature(file)
            if feat is not None:
                x.append(feat); y.append(emotion_map[parts[2]])

    model = MLPClassifier(hidden_layer_sizes=(512, 256), max_iter=700, alpha=0.001)
    model.fit(np.array(x), y)
    with open(model_file, 'wb') as f: pickle.dump(model, f)
    return model

# ==========================================
# 3. INPUT-FIRST MAIN LOOP
# ==========================================
if __name__ == "__main__":
    # Setup hidden window for file selection
    root = tk.Tk(); root.withdraw(); root.attributes("-topmost", True)
    
    print("🚀 UNIVERSAL EMOTION AI LOADED")

    while True:
        print("\n" + "="*50)
        user_input = input("👉 Press [ENTER] to pick a file | Type 'exit' to quit: ").lower().strip()
        
        if user_input == 'exit':
            print("👋 System shutting down.")
            break
        
        # STEP 1: Ask for input first!
        user_file = filedialog.askopenfilename(title="Select Audio for Emotion Analysis")
        
        if not user_file:
            print("⚠️ No file selected.")
            continue

        print(f"\n🎯 Target Selected: {os.path.basename(user_file)}")
        
        # STEP 2: Start the process (Load/Train) only after file is selected
        model = get_ready_model()
        
        # STEP 3: Make Prediction
        print(f"🔎 Analyzing voice patterns in {os.path.basename(user_file)}...")
        features = extract_feature(user_file)
        
        if features is not None:
            probs = model.predict_proba([features])[0]
            classes = model.classes_
            best_idx = np.argmax(probs)
            
            print("\n" + "📊 EMOTION SCOREBOARD")
            print("-" * 40)
            for i, emotion in enumerate(classes):
                pct = probs[i] * 100
                tag = "🌟 [WINNER]" if i == best_idx else ""
                print(f"{emotion.upper():<18} | {pct:>6.2f}% {tag}")
            print("-" * 40)
            print(f"🎯 FINAL PREDICTION: {classes[best_idx].upper()}\n")
        else:
            print("❌ ERROR: Could not process audio. Ensure FFmpeg is installed.")