from flask import Flask, jsonify
from flask_cors import CORS
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import pygame
from twilio.rest import Client

app = Flask(__name__)
CORS(app)

# =========================
# LOAD MODEL
# =========================
model = load_model("ml_model/drowsiness_model.h5")

# =========================
# CLASS LABELS
# =========================
labels = ["Closed", "Open", "no_yawn", "yawn"]

# =========================
# LOAD FACE DETECTOR
# =========================
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# =========================
# ALARM SOUND
# =========================
pygame.mixer.init()
pygame.mixer.music.load("alarm.mp3")

# =========================
# TWILIO CONFIG
# =========================
account_sid = "YOUR_ACCOUNT_SID"
auth_token = "YOUR_AUTH_TOKEN"

client = Client(account_sid, auth_token)

twilio_number = "YOUR_TWILIO_NUMBER"
receiver_number = "YOUR_PHONE_NUMBER"

sms_sent = False

# =========================
# DETECT API
# =========================
@app.route("/detect")
def detect():

    global sms_sent

    cap = cv2.VideoCapture(0)

    status = "Awake"
    confidence = 0
    alert = "Low"

    ret, frame = cap.read()

    if not ret:
        return jsonify({"error": "Camera not working"})

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:

        face = frame[y:y+h, x:x+w]

        # IMPORTANT FIX
        img = cv2.resize(face, (64, 64))

        img = img.astype("float32") / 255.0

        img = np.expand_dims(img, axis=0)

        prediction = model.predict(img, verbose=0)

        predicted_class = np.argmax(prediction)

        confidence = float(np.max(prediction))

        label = labels[predicted_class]

        # =========================
        # DROWSINESS DETECTION
        # =========================
        if label == "Closed" or label == "yawn":

            status = "Drowsy"
            alert = "High"

            # PLAY ALARM
            if not pygame.mixer.music.get_busy():
                pygame.mixer.music.play()

            # SEND SMS ONLY ONCE
            if not sms_sent:

                try:
                    message = client.messages.create(
                        body="⚠ ALERT! Driver drowsiness detected!",
                        from_=twilio_number,
                        to=receiver_number
                    )

                    print("SMS Sent Successfully")
                    sms_sent = True

                except Exception as e:
                    print("SMS Error:", e)

        else:

            status = "Awake"
            alert = "Low"

            pygame.mixer.music.stop()

            sms_sent = False

    cap.release()

    return jsonify({
        "prediction": status,
        "driver_status": status,
        "alert_level": alert,
        "confidence": round(confidence * 100, 2)
    })

# =========================
# RUN SERVER
# =========================
if __name__ == "__main__":
    app.run(debug=True)