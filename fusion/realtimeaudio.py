import pyaudio
import wave
import numpy as np
import librosa
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
import joblib

# Paramètres d'enregistrement
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
CHUNK = 1024
RECORD_SECONDS = 2  # Durée d'enregistrement pour chaque segment
WAVE_OUTPUT_FILENAME = "output.wav"

class AudioEmotionDetector:
    def __init__(self, scaler_path, model_path):
        """
        Initialise le détecteur d'émotions audio avec le modèle et le scaler fournis.
        """
        self.scaler = joblib.load(scaler_path)
        self.model = load_model(model_path)

    def record_audio_segment(self):
        """
        Enregistre un segment audio de durée fixe et retourne les frames audio.
        """
        p = pyaudio.PyAudio()
        stream = p.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=RATE,
                        input=True,
                        frames_per_buffer=CHUNK)

        frames = []

        for _ in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
            data = stream.read(CHUNK)
            frames.append(data)

        stream.stop_stream()
        stream.close()
        p.terminate()

        return frames

    def save_audio_to_file(self, frames, output_filename=WAVE_OUTPUT_FILENAME):
        """
        Sauvegarde les frames audio dans un fichier WAV.
        """
        with wave.open(output_filename, 'wb') as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(pyaudio.PyAudio().get_sample_size(FORMAT))
            wf.setframerate(RATE)
            wf.writeframes(b''.join(frames))

    def predict_emotion(self, audio_filename=WAVE_OUTPUT_FILENAME):
        """
        Prédit l'émotion à partir d'un fichier audio WAV et retourne la classe prédite et la confiance.
        """
        audio, sample_rate = librosa.load(audio_filename, sr=RATE)

        mel_feature = np.mean(librosa.feature.melspectrogram(y=audio, sr=sample_rate).T, axis=0)
        mfcc_feature = np.mean(librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=13).T, axis=0)

        features = np.hstack((mel_feature, mfcc_feature))
        features_scaled = self.scaler.transform([features])

        prediction = self.model.predict(features_scaled)

        label_map_ravdess = {
            '01': 'Neutral', '02': 'Contempt', '03': 'Happy', '04': 'Sad', '05': 'Angry',
            '06': 'Fear', '07': 'Disgust', '08': 'Surprise'
        }

        # Trouver l'index de la classe avec la plus grande probabilité
        predicted_class_index = np.argmax(prediction)
        predicted_class = label_map_ravdess[str(predicted_class_index + 1).zfill(2)]
        
        # Trouver la confiance (probabilité) de la classe prédite
        confidence = np.max(prediction)

        return predicted_class, confidence


    def run_prediction(self, audio_data):

        if isinstance(audio_data, list):
            audio_filename = "output.wav"
            frames = b''.join(audio_data)
            with wave.open(audio_filename, 'wb') as wf:
                wf.setnchannels(CHANNELS)
                wf.setsampwidth(pyaudio.PyAudio().get_sample_size(FORMAT))
                wf.setframerate(RATE)
                wf.writeframes(frames)
            emotion, confidence = self.predict_emotion()
        else:
            # Handle non-list inputs (e.g., numpy array)
            audio_filename = "output.wav"
            self.save_audio_to_file(audio_data, audio_filename)
            emotion, confidence = self.predict_emotion()

        return emotion, confidence


    def run(self):
        """
        Lance l'enregistrement, la sauvegarde, et la détection en boucle.
        """
        print("Appuyez sur Ctrl+C pour arrêter le programme.")
        try:
            while True:
                frames = self.record_audio_segment()
                self.save_audio_to_file(frames)
                emotion,confidence = self.predict_emotion()
                print(f"L'émotion prédite est : {emotion}")
        except KeyboardInterrupt:
            print("Programme arrêté.")


# Exemple d'utilisation : pour être importé et utilisé dans un autre fichier
if __name__ == "__main__":
    scaler_path = "C:\\Users\\Home\\Desktop\\ids5\\YOLO\\final_results\\Audio\\scaler.pkl"
    model_path = "C:\\Users\\Home\\Desktop\\ids5\\YOLO\\final_results\\Audio\\Audio_model.h5"
    detector = AudioEmotionDetector(scaler_path, model_path)
    detector.run()
