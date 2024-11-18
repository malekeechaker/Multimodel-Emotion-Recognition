import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image

# Define a class for YOLO predictions
class YOLOTester:
    def __init__(self, model_path, label_colors):
        self.model = YOLO(model_path)
        self.label_colors = label_colors

    def make_prediction(self, frame):
        # Convert OpenCV frame (BGR) to PIL image (RGB)
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        # Run model prediction
        results = self.model.predict(source=img, save=False)
        return results

    def annotate_frame(self, frame, results):
        # Annotate frame with bounding boxes and labels
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                class_id = int(box.cls[0].item())
                label = result.names[class_id]
                color = self.label_colors.get(label, (255, 255, 255))

                # Draw bounding box and label
                # Draw the rectangle around the detected area
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                # Add the label above the rectangle
                cv2.putText(frame, str(label), (x1+10, y1 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        return frame

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

# Initialize YOLO model tester
model_path = 'results/runs/detect/emotion-detection/weights/best.pt'
tester = YOLOTester(model_path, label_colors)

# Start webcam and make predictions in real-time
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Make prediction and annotate frame
    results = tester.make_prediction(frame)
    annotated_frame = tester.annotate_frame(frame, results)

    # Display the frame with annotations
    cv2.imshow("Real-Time YOLO Detection", annotated_frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()
