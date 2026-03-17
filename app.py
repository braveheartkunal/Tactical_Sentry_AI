from flask import Flask, render_template, Response
import cv2
from sentry_logic import SentryAI

app = Flask(__name__)
sentry = SentryAI()

def generate_frames():
    # REPLACE '0' with your CCTV RTSP link
    # Example: camera = cv2.VideoCapture("rtsp://admin:12345@192.168.1.100:554/live")
    camera = cv2.VideoCapture(0) 
    
    # Set buffer size to low for real-time performance
    camera.set(cv2.CAP_PROP_BUFFERSIZE, 2)
    
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            processed_frame, alert, count = sentry.process_frame(frame)
            ret, buffer = cv2.imencode('.jpg', processed_frame)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)