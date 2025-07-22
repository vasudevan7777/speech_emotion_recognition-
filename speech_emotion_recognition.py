import os
import librosa
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import sounddevice as sd
import soundfile as sf
from colorama import Fore, Style, init

# Initialize colorama
init(autoreset=True)

# Path to dataset
DATA_PATH = "data"

# Emotion labels based on RAVDESS filenames
EMOTIONS = {
    '01': 'neutral', '02': 'calm', '03': 'happy', '04': 'sad',
    '05': 'angry', '06': 'fearful', '07': 'disgust', '08': 'surprised'
}

# Feature extractor: MFCC + Chroma + Spectral Contrast
def extract_features(file_path):
    y, sr = librosa.load(file_path, duration=3, offset=0.5)
    mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)
    chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr).T, axis=0)
    spec_contrast = np.mean(librosa.feature.spectral_contrast(y=y, sr=sr).T, axis=0)
    return np.hstack([mfccs, chroma, spec_contrast])

# Load dataset
def load_data():
    x, y = [], []
    for file in os.listdir(DATA_PATH):
        if file.endswith(".wav"):
            emotion_code = file.split("-")[2]
            emotion = EMOTIONS.get(emotion_code)
            if emotion:
                features = extract_features(os.path.join(DATA_PATH, file))
                x.append(features)
                y.append(emotion)
    return np.array(x), np.array(y)

# Train MLP model
def train_model(x_train, y_train):
    model = make_pipeline(
        StandardScaler(),
        MLPClassifier(hidden_layer_sizes=(256, 128), max_iter=1000, learning_rate_init=0.001)
    )
    model.fit(x_train, y_train)
    return model

# Record live audio
def record_audio(filename="live.wav", duration=3, samplerate=44100):
    print(Fore.CYAN + "🎙 Recording...")
    recording = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1)
    sd.wait()
    sf.write(filename, recording, samplerate)
    print(Fore.GREEN + f"✅ Recording saved as {filename}")

# Predict emotion from file
def predict_emotion(model, filename):
    features = extract_features(filename).reshape(1, -1)
    proba = model.predict_proba(features)
    prediction = model.classes_[np.argmax(proba)]
    confidence = np.max(proba)
    return prediction, confidence

# Adaptive training: update model with new confident sample
def adaptive_train(model, x_train, y_train, new_feature, new_label):
    x_train = np.vstack([x_train, new_feature])
    y_train = np.append(y_train, new_label)
    model.fit(x_train, y_train)
    return model, x_train, y_train

# ✅ Entry point
if __name__ == "__main__":
    print(Fore.YELLOW + "\n===================================")
    print("🔄 Loading dataset...")
    x, y = load_data()

    print("🧠 Encoding labels and training model...")
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    x_train, x_test, y_train, y_test = train_test_split(x, y_encoded, test_size=0.2, random_state=42)

    model = train_model(x_train, y_train)
    print(Fore.GREEN + "✅ Model training completed!")
    
    y_pred = model.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred) * 100
    print(Fore.YELLOW + f"🎯 Accuracy: {accuracy:.2f}%")
    print("===================================\n")

    # Record and Predict
    record_audio()
    pred, conf = predict_emotion(model, "live.wav")
    
    print(Fore.MAGENTA + "\n🎤 Emotion Prediction Result")
    print("-----------------------------------")
    print(Fore.BLUE + f"🗣 Detected Emotion : {label_encoder.inverse_transform([pred])[0]}")
    print(Fore.BLUE + f"📊 Confidence Level  : {conf * 100:.2f}%")
    print("-----------------------------------\n")

    # Optionally adapt model
    if conf > 0.9:
        print(Fore.CYAN + "➕ High confidence detected. Adding to training data...")
        new_feat = extract_features("live.wav").reshape(1, -1)
        model, x_train, y_train = adaptive_train(model, x_train, y_train, new_feat, pred)