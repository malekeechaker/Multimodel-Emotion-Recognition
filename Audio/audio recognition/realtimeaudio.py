import pyaudio
import wave
import numpy as np
import librosa
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
import joblib
import os

# Paramètres d'enregistrement
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
CHUNK = 1024
RECORD_SECONDS = 5
WAVE_OUTPUT_FILENAME = "output.wav"

# Initialiser PyAudio
p = pyaudio.PyAudio()

# Démarrer l'enregistrement
stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK)

print("Enregistrement...")
frames = []

# Enregistrer l'audio pendant le nombre de secondes spécifié
for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
    data = stream.read(CHUNK)
    frames.append(data)

# Arrêter l'enregistrement
print("Enregistrement terminé.")
stream.stop_stream()
stream.close()
p.terminate()

# Sauvegarder l'audio dans un fichier WAV
with wave.open(WAVE_OUTPUT_FILENAME, 'wb') as wf:
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))

# Charger le fichier audio avec librosa
audio, sample_rate = librosa.load(WAVE_OUTPUT_FILENAME, sr=RATE)

# Extraire les caractéristiques du signal audio
mel_feature = np.mean(librosa.feature.melspectrogram(y=audio, sr=sample_rate).T, axis=0)
mfcc_feature = np.mean(librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=13).T, axis=0)

# Combiner les deux types de caractéristiques
features = np.hstack((mel_feature, mfcc_feature))

# Charger le scaler pour mettre à l'échelle les caractéristiques
# Note: Assurez-vous que le scaler a été déjà ajusté avec des données d'entraînement
scaler = joblib.load("C:.\\scaler.pkl")

# Mettre à l'échelle les caractéristiques
features_scaled = scaler.transform([features])

# Charger votre modèle pré-entraîné
model = load_model("C:.\\Audio_model.h5")

# Effectuer une prédiction
prediction = model.predict(features_scaled)

# Map des étiquettes d'émotions
label_map_ravdess = {
    '01': 'neutral', '02': 'calm', '03': 'happy', '04': 'sad', '05': 'angry',
    '06': 'fearful', '07': 'disgust', '08': 'surprised'
}

# Décoder la classe prédite
predicted_class_index = np.argmax(prediction)
predicted_class = label_map_ravdess[str(predicted_class_index + 1).zfill(2)] 

# Afficher l'émotion prédite
print(f"L'émotion prédite est : {predicted_class}")
