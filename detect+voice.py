from ultralytics import YOLO
import cv2
import pyttsx3
import time

# Initialize text-to-speech engine
engine = pyttsx3.init()
engine.setProperty("rate", 180)  # Speech speed
engine.setProperty("volume", 1.0)

def speak(text):
    engine.say(text)
    engine.runAndWait()

# Load YOLOv8 model
model = YOLO("runs/detect/custom_yolo_model/weights/best.pt")

# Start webcam
cap = cv2.VideoCapture(0)

last_spoken = ""
last_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run detection
    results = model(frame)
    annotated_frame = results[0].plot()

    # Extract class names from detections
    boxes = results[0].boxes
    if boxes is not None and len(boxes) > 0:
        names = model.names
        detected_labels = [names[int(cls)] for cls in boxes.cls]

        # Announce only if object changed or after 2 seconds
        current_label = ", ".join(set(detected_labels))
        if current_label != last_spoken and (time.time() - last_time) > 2:
            print(f"🗣 Detected: {current_label}")
            speak(f"I see {current_label}")
            last_spoken = current_label
            last_time = time.time()

    # Display result
    cv2.imshow("YOLOv8 Voice Detection", annotated_frame)

    # Exit on ESC
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
