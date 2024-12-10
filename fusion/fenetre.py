import cv2

import queue
import threading
import sys
import time
import pyaudio
import speech_recognition as sr


from fusion import unified_prediction
from RealTimevideo import YOLOTester
from realtimeaudio import AudioEmotionDetector
#from realtimetext import predict

# Global variables for audio processing
audio_queue = queue.Queue()
audio_stream_active = True


def audio_prediction(audio):
    scaler_path = "C:\\Users\\Home\\Desktop\\ids5\\YOLO\\final_results\\Audio\\scaler.pkl"
    model_path = "C:\\Users\\Home\\Desktop\\ids5\\YOLO\\final_results\\Audio\\Audio_model.h5"
    detector = AudioEmotionDetector(scaler_path, model_path)
    return detector.run_prediction(audio)

def text_prediction(audio):
    '''
    nltk.download('stopwords')
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    # Charger le modèle d'émotion
    loaded_model = tf.keras.models.load_model(
        "C:\\Users\\Home\\Desktop\\ids5\\YOLO\\final_results\\Text\\model.h5", 
        custom_objects={"TFBertModel": TFBertModel}
    )
    
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    '''
    recognizer = sr.Recognizer()
    text = recognizer.recognize_google(audio, language="en-EN")  # Changez la langue si nécessaire
    if text:
        print(f"Texte reconnu : {text}")
        return predict( text)
 

def annotate_frame(frame, audio):
    model_path = ".\\final_results\\runs\\detect\\final_run\\weights\\best.pt"

    label_colors = {
        'Happy': (0, 255, 0),      # Green
        'Sad': (0, 0, 255),        # Blue
        'Anger': (255, 0, 0),      # Red
        'Neutral': (255, 255, 0),  # Yellow
        'Contempt': (128, 0, 128), # Purple
        'Fear': (255, 165, 0),     # Orange
        'Surprise': (0, 255, 255), # Cyan
        'Disgust': (0, 128, 0)     # Dark Green
    }
    yolo_tester = YOLOTester(model_path, label_colors)
    video_results = yolo_tester.make_live_prediction(frame)
    video_label , video_confidence = yolo_tester.result_prediction(video_results)

    audio_label , audio_confidence = audio_prediction(audio)
    #text_label , text_confidence = text_prediction(audio)
    text_label , text_confidence ="Happy",0.01
    label , confidence = unified_prediction(text_label , text_confidence,audio_label , audio_confidence, video_label , video_confidence)
    yolo_tester.annotate_frame(frame, label , confidence)
    yolo_tester.annotate_frame(frame, video_label , video_confidence,(50,70))
    yolo_tester.annotate_frame(frame, audio_label , audio_confidence,(50,90))
    yolo_tester.annotate_frame(frame, text_label , text_confidence,(50,110))






# Function to start audio capture
def start_audio_capture():
    """
    Capture l'audio en continu et place les segments audio dans la file.
    """
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 44100
    CHUNK = 1024
    RECORD_SECONDS = 2  # Durée d'enregistrement pour chaque segment
    WAVE_OUTPUT_FILENAME = "output.wav"

    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

    while audio_stream_active:
        frames = []
        for _ in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
            data = stream.read(CHUNK)
            frames.append(data)


        audio_queue.put(frames)

    stream.stop_stream()
    stream.close()
    p.terminate()

# Function to display the camera feed with audio capture
def display_camera_and_audio_feed():
    cap = cv2.VideoCapture(0)  # Capture video from the default camera
    audio_thread = threading.Thread(target=start_audio_capture)
    audio_thread.start()

    # Set a custom frame size for the window
    frame_width = 1280
    frame_height = 720
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

    last_audio_time = time.time()
    audio_data = None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Check if it's time to collect new audio data (every 2 seconds)
        current_time = time.time()
        if current_time - last_audio_time >= 2:
            # Retrieve audio data from the queue if available
            if not audio_queue.empty():
                audio_data = audio_queue.get()
            last_audio_time = current_time

        if audio_queue.empty():
            audio_data = []
        # Annotate the frame with audio data if available
        annotate_frame(frame, audio_data)

        # Show the video feed
        cv2.imshow("Real-Time Camera Feed", frame)

        # Stop when 'Q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Stop audio capture thread
    global audio_stream_active
    audio_stream_active = False
    audio_thread.join()

    cap.release()
    cv2.destroyAllWindows()

    return frame, audio_data



# Main function to run the real-time camera and audio capture
if __name__ == "__main__":
    frame, audio = display_camera_and_audio_feed()
    print("Audio and frame capture stopped.")
    # Optionally, you can save the audio or process it further here
