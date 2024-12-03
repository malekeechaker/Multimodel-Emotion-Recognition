import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import av
import librosa
from queue import Queue
import numpy as np
import speech_recognition as sr  # Pour la conversion audio -> texte

# Modèles fictifs (remplacez par vos modèles réels)
def predict_audio_model(audio_data, sample_rate):
    return f"Prédiction Audio "

def predict_text_model(transcribed_text):
    return f"Prédiction Texte "

def predict_video_model(frame):
    return "Prédiction Vidéo : Visage détecté"

# Conversion Audio -> Texte
def audio_to_text(audio_data, sample_rate):
    try:
        # Sauvegarde temporaire du fichier audio en WAV
        temp_wav = "temp_audio.wav"
        librosa.output.write_wav(temp_wav, audio_data, sample_rate)

        # Utilisation de SpeechRecognition pour la transcription
        recognizer = sr.Recognizer()
        with sr.AudioFile(temp_wav) as source:
            audio = recognizer.record(source)
            text = recognizer.recognize_google(audio, language="fr-FR")
        return text
    except Exception as e:
        return f"Erreur STT : {e}"

# Callback pour le flux vidéo
def video_callback(frame, video_queue):
    img = frame.to_ndarray(format="bgr24")
    prediction = predict_video_model(img)
    video_queue.put(prediction)
    return av.VideoFrame.from_ndarray(img, format="bgr24")

# Callback pour le flux audio
def audio_callback(frame: av.AudioFrame, audio_queue, text_queue):
    audio_data = frame.to_ndarray()
    audio_data = librosa.util.buf_to_float(audio_data.flatten())

    # Prédiction audio
    audio_prediction = predict_audio_model(audio_data, frame.sample_rate)
    audio_queue.put(audio_prediction)

    # Conversion audio -> texte et prédiction texte
    transcribed_text = audio_to_text(audio_data, frame.sample_rate)
    text_prediction = predict_text_model(transcribed_text)
    text_queue.put(text_prediction)

# Application principale
st.title("Prédictions en temps réel : Audio, Texte et Vidéo")

# Configuration WebRTC
rtc_config = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})
video_queue = Queue()
audio_queue = Queue()
text_queue = Queue()

# Ajouter un bouton unique pour démarrer les flux
if "stream_active" not in st.session_state:
    st.session_state["stream_active"] = False

if st.button("Démarrer la collecte audio et vidéo"):
    st.session_state["stream_active"] = True

if st.session_state["stream_active"]:
    st.subheader("Flux Vidéo et Audio en cours...")

    # Flux vidéo et audio combinés
    webrtc_ctx = webrtc_streamer(
        key="stream",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=rtc_config,
        media_stream_constraints={"video": True, "audio": True},
        video_frame_callback=lambda frame: video_callback(frame, video_queue),
        audio_frame_callback=lambda frame: audio_callback(frame, audio_queue, text_queue),
    )

    # Affichage des résultats
    st.subheader("Résultats des Prédictions")
    video_placeholder = st.empty()
    audio_placeholder = st.empty()
    text_placeholder = st.empty()

    # Boucle principale pour les mises à jour
    while webrtc_ctx and webrtc_ctx.state.playing:
        # Affichage des prédictions vidéo
        if not video_queue.empty():
            video_placeholder.text(f"Prédiction Vidéo : {video_queue.get()}")

        # Affichage des prédictions audio
        if not audio_queue.empty():
            audio_placeholder.text(f"Prédiction Audio : {audio_queue.get()}")

        # Affichage des prédictions texte
        if not text_queue.empty():
            text_placeholder.text(f"Prédiction Texte : {text_queue.get()}")
