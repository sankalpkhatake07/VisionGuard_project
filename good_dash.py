import streamlit as st
from ultralytics import YOLO
import cv2
import time
import pyttsx3
import tempfile
import numpy as np

# ===========================
# ⚙️ PAGE SETUP
# ===========================
st.set_page_config(page_title="Smart Object Detection", layout="wide")
st.markdown("""
    <style>
        .main {
            background: radial-gradient(circle at top left, #0f2027, #203a43, #2c5364);
            color: white;
        }
        h1, h2, h3, h4 {
            text-align: center;
            color: #00FFD1 !important;
            text-shadow: 0 0 20px #00FFD155;
        }
        .stButton>button {
            background-color: #00FFD1;
            color: black;
            font-weight: bold;
            border-radius: 10px;
            box-shadow: 0 0 15px #00FFD155;
            transition: 0.2s;
        }
        .stButton>button:hover {
            transform: scale(1.05);
            background-color: #00CCAA;
            color: white;
        }
    </style>
""", unsafe_allow_html=True)

# ===========================
# 🧠 MODEL + VOICE ENGINE
# ===========================
model = YOLO("runs/detect/custom_yolo_model/weights/best.pt")

engine = pyttsx3.init()
engine.setProperty('rate', 165)
engine.setProperty('volume', 1.0)

# ===========================
# 🎯 TITLE
# ===========================
st.title("🎯 Smart Object Detection")
st.markdown("##### Real-time Object Detection using YOLOv8 + Voice Feedback")

col1, col2 = st.columns([1, 1])

# ===========================
# 🎥 SELECT SOURCE
# ===========================
with col1:
    st.subheader("🧭 Choose Input Source")
    source_option = st.radio("Select Mode", ["🎦 Webcam", "🖼️ Upload Image"])

# ===========================
# 🧩 UPLOAD IMAGE MODE
# ===========================
if source_option == "🖼️ Upload Image":
    uploaded_file = col2.file_uploader("📤 Upload an image file", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        frame = cv2.imread(tfile.name)

        start_time = time.time()
        results = model(frame)
        end_time = time.time()

        fps = 1 / (end_time - start_time)
        annotated_frame = results[0].plot()
        annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)

        col1.image(annotated_frame, caption="🔍 Detection Results", use_column_width=True)

        detected_info = []
        for box in results[0].boxes:
            cls_id = int(box.cls)
            conf = float(box.conf)
            label = model.names[cls_id]
            detected_info.append(f"{label} ({conf*100:.1f}%)")

        if detected_info:
            text_out = ", ".join(detected_info)
            col2.success(f"🧠 Detected: {text_out}")
            col2.metric("⚡ FPS", f"{fps:.1f}")
            engine.say(text_out)
            engine.runAndWait()
        else:
            col2.warning("No objects detected in this image.")

# ===========================
# 🎦 WEBCAM MODE (OPTIMIZED)
# ===========================
elif source_option == "🎦 Webcam":
    st.info("🎥 Press **Start Detection** to activate your webcam")

    run_detection = st.button("🚀 Start Detection", key="start_button")

    if run_detection:
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        # Create smaller layout: left = video feed (1/4 width), right = stats
        left_col, right_col = st.columns([1, 3])
        frame_window = left_col.image([], width=320)  # smaller feed (1/4 screen)
        right_col.markdown("### 📊 Detection Info")
        spoken_labels = set()

        st.toast("🟢 YOLOv8 Live Detection Started")

        # Optimization: Reduce update rate to reduce lag
        fps_placeholder = right_col.empty()
        detect_placeholder = right_col.empty()

        stop_button = st.button("🛑 Stop Detection", key="stop_button")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                st.error("❌ Unable to access camera.")
                break

            # Resize frame for faster processing
            frame_resized = cv2.resize(frame, (640, 480))

            start_time = time.time()
            results = model(frame_resized, verbose=False)
            end_time = time.time()

            fps = 1 / (end_time - start_time + 1e-6)
            annotated_frame = results[0].plot()

            # Extract detected objects
            labels = []
            for box in results[0].boxes:
                cls_id = int(box.cls)
                conf = float(box.conf)
                label = f"{model.names[cls_id]} ({conf*100:.1f}%)"
                labels.append(label)

            # Show compact video feed
            frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            frame_window.image(frame_rgb, channels="RGB")

            if labels:
                detected_text = ", ".join(labels)
                detect_placeholder.markdown(f"**🧩 Detected:** {detected_text}")
                fps_placeholder.metric("⚡ FPS", f"{fps:.1f}")
                # Speak only for new detections
                for l in labels:
                    base_label = l.split('(')[0].strip()
                    if base_label not in spoken_labels:
                        engine.say(base_label)
                        engine.runAndWait()
                        spoken_labels.add(base_label)
            else:
                detect_placeholder.markdown("No objects detected.")
                fps_placeholder.metric("⚡ FPS", f"{fps:.1f}")

            # Break if Stop clicked
            if stop_button:
                st.toast("🛑 Detection Stopped")
                break

        cap.release()
3
