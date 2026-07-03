# 🚗 AI Driver Drowsiness Detection System

An AI-powered real-time Driver Drowsiness Detection System developed using **Computer Vision** and **Deep Learning**. The system monitors the driver's face through a webcam, detects signs of drowsiness, and generates instant alerts to help prevent road accidents.

---

## 📌 Features

- 👁️ Real-time driver monitoring using a webcam
- 😊 Face detection with OpenCV
- 🧠 Drowsiness prediction using MobileNetV2
- 🔊 Audio alarm using Pygame
- 📩 SMS alerts using Twilio
- 📞 Voice call notifications using Twilio
- 💻 User-friendly React.js interface
- ⚡ Real-time prediction and status display

---

## 🛠️ Tech Stack

### Frontend
- React.js
- HTML5
- CSS3
- JavaScript

### Backend
- Flask
- Python

### AI & Computer Vision
- TensorFlow
- Keras
- MobileNetV2
- OpenCV
- NumPy

### Notification Services
- Twilio API
- Pygame

---

## 🏗️ System Workflow

1. Webcam captures the driver's live video.
2. OpenCV detects the driver's face.
3. Image preprocessing is performed.
4. MobileNetV2 predicts whether the driver is **Awake** or **Drowsy**.
5. Consecutive drowsy frames are verified.
6. If drowsiness is confirmed:
   - Audio alarm is triggered.
   - SMS notification is sent.
   - Voice call is initiated.
7. The driver's status is displayed on the dashboard.

---

## 📊 Model Comparison

| Model | Validation Accuracy |
|--------|--------------------:|
| CNN | 97.31% |
| ResNet50 | 98.64% |
| **MobileNetV2** | **99.08%** ✅ |

**MobileNetV2** was selected because it offers high accuracy, fast inference, and is lightweight enough for real-time applications.

---

## 📂 Project Structure

```
AI-Driver-Drowsiness-Detection-System/
│
├── frontend/
├── backend/
├── ml_model/
├── screenshots/
├── README.md
├── package.json
└── requirements.txt
```

---

## 🚀 Installation

### Clone the Repository

```bash
git clone https://github.com/Abhayhiremath/AI-Driver-Drowsiness-Detection-System.git
```

### Navigate to the Project

```bash
cd AI-Driver-Drowsiness-Detection-System
```

### Install Backend Dependencies

```bash
pip install -r requirements.txt
```

### Install Frontend Dependencies

```bash
cd frontend
npm install
```

### Start the Backend

```bash
python app.py
```

### Start the Frontend

```bash
npm run dev
```

---

## 📸 Project Screenshots

> Add screenshots of:
- Home Page
- Live Detection Screen
- Drowsiness Detection
- SMS Alert
- Voice Call Alert

---

## 🎯 Project Outcome

- Successfully detects driver drowsiness in real time.
- Achieved **99.08% validation accuracy** using MobileNetV2.
- Generates instant audio, SMS, and voice call alerts.
- Helps improve road safety by reducing fatigue-related accidents.

---

## 🔮 Future Enhancements

- GPS location sharing during emergencies.
- Mobile application support.
- Cloud-based monitoring dashboard.
- Night vision support.
- Multi-driver detection.
- Driver fatigue analytics.

---

## 👨‍💻 Author

**Abhay S. Hiremath**

- GitHub: https://github.com/Abhayhiremath
- LinkedIn: https://www.linkedin.com/in/abhay-hiremath/

---

## 📜 License

This project is developed for educational and academic purposes.
