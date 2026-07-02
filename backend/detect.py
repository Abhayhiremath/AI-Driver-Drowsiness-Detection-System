import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import pygame
import threading
import time

from twilio.rest import Client

# ==========================================
# LOAD AI MODEL
# ==========================================

model = tf.keras.models.load_model(
    "ml_model/mobilenetv2_model.h5"
)

print("✅ AI Model Loaded Successfully")

# ==========================================
# LOAD ALARM SOUND
# ==========================================

pygame.mixer.init()

alarm_sound = pygame.mixer.Sound("alarm.mp3")

print("✅ Alarm Sound Loaded Successfully")

# ==========================================
# TWILIO CONFIGURATION
# ==========================================

ACCOUNT_SID = "Twilio SID"

AUTH_TOKEN = "Twilio Token"

TWILIO_PHONE = "Twilio Number"

TO_PHONE = "Your Number"

# CREATE CLIENT
client = Client(
    ACCOUNT_SID,
    AUTH_TOKEN
)

print("✅ Twilio Initialized")

# ==========================================
# FACE DETECTOR
# ==========================================

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)

# ==========================================
# GLOBAL VARIABLES
# ==========================================

alarm_playing = False

drowsy_counter = 0
awake_counter = 0

# Consecutive frames for stable detection
CONSECUTIVE_FRAMES = 5

# Prevent SMS spam
last_alert_time = 0

# Alert cooldown in seconds
ALERT_COOLDOWN = 60

# ==========================================
# PLAY ALARM
# ==========================================

def play_alarm():

    global alarm_playing

    try:

        if not alarm_playing:

            alarm_sound.play()

            alarm_playing = True

            print("🚨 Alarm Started")

    except Exception as e:

        print("❌ Alarm Error:", e)

# ==========================================
# STOP ALARM
# ==========================================

def stop_alarm():

    global alarm_playing

    try:

        alarm_sound.stop()

        alarm_playing = False

        print("🔇 Alarm Stopped")

    except Exception as e:

        print("❌ Stop Alarm Error:", e)

# ==========================================
# SEND SMS + CALL ALERT
# ==========================================

def send_alert():

    global last_alert_time

    current_time = time.time()

    # Prevent repeated alerts
    if current_time - last_alert_time < ALERT_COOLDOWN:
        return

    last_alert_time = current_time

    try:

        # ==========================================
        # SEND SMS
        # ==========================================

        sms = client.messages.create(

            body="""
⚠️ DRIVER DROWSINESS ALERT

Driver is feeling drowsy.

Please stop the vehicle immediately.
            """,

            from_=TWILIO_PHONE,

            to=TO_PHONE
        )

        print("✅ SMS Sent")
        print("SMS SID:", sms.sid)

        # ==========================================
        # MAKE VOICE CALL
        # ==========================================

        call = client.calls.create(

            twiml="""
<Response>
    <Say>
        Warning.
        Driver is feeling drowsy.
        Please stop the vehicle immediately.
    </Say>
</Response>
            """,

            from_=TWILIO_PHONE,

            to=TO_PHONE
        )

        print("✅ Call Alert Sent")
        print("Call SID:", call.sid)

    except Exception as e:

        print("❌ Twilio Error:", e)

# ==========================================
# MAIN DETECTION FUNCTION
# ==========================================

def process_frame(frame):

    global drowsy_counter
    global awake_counter

    status = "Awake"

    # ==========================================
    # CONVERT FRAME TO GRAYSCALE
    # ==========================================

    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )

    # ==========================================
    # DETECT FACE
    # ==========================================

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(100, 100)
    )

    # ==========================================
    # LOOP THROUGH DETECTED FACES
    # ==========================================

    for (x, y, w, h) in faces:

        try:

            # Extract face ROI
            face = frame[y:y+h, x:x+w]

            # ==========================================
            # PREPROCESS IMAGE
            # ==========================================

            # Convert BGR → RGB
            rgb_face = cv2.cvtColor(
                face,
                cv2.COLOR_BGR2RGB
            )

            # Resize to model size
            resized = cv2.resize(
                rgb_face,
                (224, 224)
            )

            # Normalize image
            normalized = preprocess_input(
            resized.astype(np.float32)
            )

            # Reshape image
            reshaped = np.expand_dims(
                normalized,
                axis=0
            )

            # ==========================================
            # MODEL PREDICTION
            # ==========================================

            prediction = float(
                model.predict(
                    reshaped,
                    verbose=0
                )[0][0]
            )

            print("Prediction Value:", prediction)

            # ==========================================
            # DROWSINESS LOGIC
            # ==========================================

            # HIGH VALUE = DROWSY
            # LOW VALUE = AWAKE

            if prediction > 0.5:

                drowsy_counter += 1

                awake_counter = 0

                print(
                    "⚠️ Drowsy Frame:",
                    drowsy_counter
                )

                # Confirm Drowsiness
                if drowsy_counter >= CONSECUTIVE_FRAMES:

                    status = "Drowsy"

                    print("🚨 DROWSINESS DETECTED")

                    # START ALARM
                    play_alarm()

                    # SEND ALERT
                    threading.Thread(
                        target=send_alert,
                        daemon=True
                    ).start()

            else:

                awake_counter += 1

                print(
                    "✅ Awake Frame:",
                    awake_counter
                )

                # Confirm Awake
                if awake_counter >= 3:

                    drowsy_counter = 0

                    status = "Awake"

                    print("✅ DRIVER AWAKE")

                    # STOP ALARM
                    stop_alarm()

            # ==========================================
            # DRAW RECTANGLE
            # ==========================================

            if status == "Drowsy":

                color = (0, 0, 255)

            else:

                color = (0, 255, 0)

            cv2.rectangle(
                frame,
                (x, y),
                (x + w, y + h),
                color,
                2
            )

            cv2.putText(
                frame,
                status,
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                color,
                2
            )

        except Exception as e:

            print("❌ Prediction Error:", e)

    return frame, status