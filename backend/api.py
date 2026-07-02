from flask import Flask, Response, jsonify
from flask_cors import CORS
import cv2

from detect import process_frame

# ==========================================
# FLASK APP
# ==========================================

app = Flask(__name__)

CORS(app)

# ==========================================
# CAMERA
# ==========================================

camera = cv2.VideoCapture(0)

latest_status = "Awake"

# ==========================================
# GENERATE VIDEO FRAMES
# ==========================================

def generate_frames():

    global latest_status

    while True:

        success, frame = camera.read()

        if not success:
            break

        # Process AI Detection
        frame, status = process_frame(frame)

        latest_status = status

        # Convert frame to jpg
        ret, buffer = cv2.imencode(
            '.jpg',
            frame
        )

        frame_bytes = buffer.tobytes()

        yield (
            b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' +
            frame_bytes +
            b'\r\n'
        )

# ==========================================
# VIDEO FEED ROUTE
# ==========================================

@app.route('/video_feed')

def video_feed():

    return Response(
        generate_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )

# ==========================================
# STATUS ROUTE
# ==========================================

@app.route('/status')

def status():

    return jsonify({
        "status": latest_status
    })

# ==========================================
# HOME ROUTE
# ==========================================

@app.route('/')

def home():

    return jsonify({
        "message": "AI Driver Drowsiness Detection Backend Running"
    })

# ==========================================
# MAIN
# ==========================================

if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )