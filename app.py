import streamlit as st
from ultralytics import YOLO
import cv2
import tempfile
import pyttsx3

# Initialize model and TTS
model = YOLO("runs/detect/custom_yolo_model/weights/best.pt")
engine = pyttsx3.init()
engine.setProperty('rate', 160)

st.set_page_config(page_title="YOLOv8 Live Object Detection", layout="wide")

st.title("🔍 YOLOv8 Real-Time Object Detection Dashboard")
st.write("Upload an image or use your webcam feed for real-time detection.")

# Sidebar
st.sidebar.header("Settings")
source_option = st.sidebar.selectbox("Select Source", ["Webcam", "Upload Image"])

if source_option == "Upload Image":
    uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        frame = cv2.imread(tfile.name)

        results = model(frame)
        annotated_frame = results[0].plot()
        detected_classes = results[0].names

        st.image(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB), caption="Detected Objects")

        labels = [model.names[int(box.cls)] for box in results[0].boxes]
        if labels:
            spoken_text = ", ".join(labels)
            st.success(f"Detected: {spoken_text}")
            engine.say(spoken_text)
            engine.runAndWait()
        else:
            st.info("No objects detected.")

elif source_option == "Webcam":
    st.info("Starting your webcam... Press **Stop** to end stream.")

    run = st.checkbox("Run Detection")

    camera = cv2.VideoCapture(0)
    frame_window = st.image([])

    while run:
        ret, frame = camera.read()
        if not ret:
            st.error("Failed to access webcam.")
            break

        results = model(frame)
        annotated_frame = results[0].plot()

        # Get detected labels
        labels = [model.names[int(box.cls)] for box in results[0].boxes]
        if labels:
            spoken_text = ", ".join(labels)
            st.write(f"Detected: {spoken_text}")
            engine.say(spoken_text)
            engine.runAndWait()

        frame_window.image(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB))

    camera.release()
