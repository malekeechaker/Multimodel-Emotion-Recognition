import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image

# Define a class for YOLO predictions
class YOLOTester:
    def __init__(self, model_path, label_colors):
        self.model = YOLO(model_path)
        self.label_colors = label_colors

    def make_prediction(self, img_input):
        # Handle both image file paths and numpy arrays (e.g., frames from a video)
        if isinstance(img_input, str):  # If it's a file path
            img = Image.open(img_input)
        elif isinstance(img_input, np.ndarray):  # If it's a numpy array (from cv2)
            img = Image.fromarray(cv2.cvtColor(img_input, cv2.COLOR_BGR2RGB))  # Convert to PIL format
        else:
            raise ValueError("Unsupported image input format")

        # Run model prediction
        results = self.model.predict(source=img, save=False)
        return results
    
    def make_live_prediction(self, frame):
        # Convert OpenCV frame (BGR) to PIL image (RGB)
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        # Run model prediction
        results = self.model.predict(source=img, save=False)
        return results

    def result_prediction(self, results):
        max_prob = 0
        detected_emotion = None
        
        # Parcourir chaque résultat de détection
        for result in results:
            for box in result.boxes:
                confidence = box.conf[0].item()  # Confiance associée à la détection
                class_id = int(box.cls[0].item())
                label = result.names[class_id]

                # Vérifier si cette détection a la plus grande probabilité
                if confidence > max_prob:
                    max_prob = confidence
                    detected_emotion = label

        # Retourner l'émotion avec la plus grande probabilité et la confiance
        return detected_emotion, max_prob




    def annotate_frame(self, frame, detected_emotion, max_prob, position=(50, 50), color=(255, 255, 255)):
        """
        Affiche le label de l'émotion et la confiance sur l'image.

        :param frame: Image à annoter.
        :param detected_emotion: Émotion détectée avec la plus grande probabilité.
        :param max_prob: Confiance de la détection de l'émotion.
        :param position: Position où afficher le texte sur l'image.
        :param color: Couleur du texte.
        """
        if detected_emotion is not None and max_prob is not None:
            # Créer le texte à afficher
            color = self.label_colors.get(detected_emotion, (255, 255, 255))

            text = f"{detected_emotion} ({max_prob:.2f})"
            
            # Afficher le texte sur l'image
            cv2.putText(frame, text, position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)



# Define label colors
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

# Function to create a new YOLOTester instance
def create_yolo_tester(model_path):
    return YOLOTester(model_path, label_colors)

# Function to display the live camera feed and make predictions
def display_camera_feed(tester):
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or frame is None or frame.size == 0:
            print("Received an empty frame. Skipping.")
            continue

        # Make prediction and extract detected emotion and confidence
        results = tester.make_live_prediction(frame)
        detected_emotion, max_prob = tester.result_prediction(results)

        # Annotate the frame with detected emotion and confidence
        annotated_frame = tester.annotate_frame(frame, detected_emotion, max_prob)

        if annotated_frame is None or annotated_frame.size == 0:
            print("Annotated frame is empty. Skipping.")
            continue

        # Display the frame with annotations
        cv2.imshow("Real-Time YOLO Detection", annotated_frame)

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


# Main function to execute the live detection
if __name__ == "__main__":
    # Specify the path to your YOLO model weights
    model_path = "C:\\Users\\Home\\Desktop\\ids5\\YOLO\\final_results\\runs\\detect\\final_run\\weights\\best.pt"
    yolo_tester = create_yolo_tester(model_path)
    display_camera_feed(yolo_tester)
