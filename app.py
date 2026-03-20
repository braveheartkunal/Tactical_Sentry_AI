import cv2
from ultralytics import YOLO
import numpy as np

# Load the model
model = YOLO('yolov8n.pt') 

def process_video_upload(video_path):
    cap = cv2.VideoCapture(video_path)
    prev_centroids = {} # To track movement
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        # Detect only class 0 (Person)
        results = model.predict(frame, classes=[0], conf=0.5)
        
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # Get center of the human detection
                x_center = (box.xyxy[0][0] + box.xyxy[0][2]) / 2
                y_center = (box.xyxy[0][1] + box.xyxy[0][3]) / 2
                current_centroid = (int(x_center), int(y_center))

                # Logic for "Movement": Compare current pos to previous frame
                # If movement is > 5 pixels, trigger an alert
                # [Detailed movement tracking logic goes here]
                print(f"Human detected at {current_centroid}")

    cap.release()

# For a web deployment, you would wrap this in a Flask/FastAPI route