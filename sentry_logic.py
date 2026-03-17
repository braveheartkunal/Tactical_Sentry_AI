import cv2
from ultralytics import YOLO

class SentryAI:
    def __init__(self):
        # Load the YOLOv8 model (Nano version for speed)
        self.model = YOLO('yolov8n.pt')
        self.target_class = 0  # 0 is the COCO class for 'person'

    def process_frame(self, frame):
        # Run inference: detect only humans (class 0)
        results = self.model.predict(frame, classes=[self.target_class], conf=0.5, verbose=False)
        
        # Get annotated frame
        annotated_frame = results[0].plot()
        
        # Logic for Alert Trigger
        detection_count = len(results[0].boxes)
        is_alert = detection_count > 0
        
        return annotated_frame, is_alert, detection_count