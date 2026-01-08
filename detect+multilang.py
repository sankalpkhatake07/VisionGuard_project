from ultralytics import YOLO
import cv2
import pyttsx3
import time

# Initialize speech engine
engine = pyttsx3.init()
engine.setProperty("rate", 175)
engine.setProperty("volume", 1.0)

# 💬 Choose language (options: 'english', 'hindi', 'marathi')
LANG = "hindi"

# Load YOLO model
model = YOLO("runs/detect/custom_yolo_model/weights/best.pt")

# Try to get voice by language
voices = engine.getProperty('voices')
for voice in voices:
    if LANG == "hindi" and ("hi" in voice.languages or "Hindi" in voice.name):
        engine.setProperty('voice', voice.id)
        break
    elif LANG == "marathi" and ("mr" in voice.languages or "Marathi" in voice.name):
        engine.setProperty('voice', voice.id)
        break
    elif LANG == "english" and ("en" in voice.languages or "English" in voice.name):
        engine.setProperty('voice', voice.id)
        break

def speak(text):
    engine.say(text)
    engine.runAndWait()

# Start webcam
cap = cv2.VideoCapture(0)

last_spoken = ""
last_time = time.time()

print("🎙️ Starting YOLO Voice Detection... Press ESC to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)
    annotated_frame = results[0].plot()

    boxes = results[0].boxes
    if boxes is not None and len(boxes) > 0:
        names = model.names
        detected_labels = [names[int(cls)] for cls in boxes.cls]

        # Speak current detections every 3 seconds
        current_label = ", ".join(set(detected_labels))
        if (time.time() - last_time) > 3:
            if LANG == "hindi":
                speak(f"मुझे दिख रहा है {current_label}")
            elif LANG == "marathi":
                speak(f"मला दिसत आहे {current_label}")
            else:
                speak(f"I see {current_label}")
            print(f"🗣 {current_label}")
            last_time = time.time()

    cv2.imshow("YOLOv8 Voice Detection", annotated_frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
