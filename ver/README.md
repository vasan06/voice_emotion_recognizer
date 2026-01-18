🚀 Quick Start Guide: Automated Setup

This project is designed to be "Plug-and-Play." You do not need to manually download datasets. The script handles the environment setup, dataset acquisition, and model training automatically upon the first run.
1. One-Time Setup

Ensure you have Python 3.8+ and FFmpeg installed on your system.
Bash

# Install all required libraries
pip install -r requirements.txt

2. Run the Engine

Launch the script to begin the automated process:
Bash

python main.py

3. What Happens Automatically:

    Step 1: Dataset Acquisition: The script uses kagglehub to download the RAVDESS Emotional Speech Audio dataset (~1.08 GB) to your local cache folder.

    Step 2: Feature Extraction: It scans the downloaded files and extracts 40 MFCCs, 40 Deltas, and 40 Delta-2s from each audio clip.

    Step 3: Training: A Multi-Layer Perceptron (MLP) neural network is trained and standardized using a StandardScaler pipeline for maximum precision.

    Step 4: Persistence: The trained model is saved as emotion_model.pkl so you never have to train it again.

4. Test with Your Own Files

Once the training is complete (or if the model is already loaded):

    A File Explorer window will pop up.

    Select any audio file (.mp3, .wav, .m4a, etc.).

    The AI will immediately output the Emotion Label and Confidence Score in the console.

🛠️ Requirements (Copy-Paste)

Save this as requirements.txt:
Plaintext

librosa
numpy
kagglehub
soundfile
pydub
scikit-learn