import streamlit as st
import cv2
import numpy as np
import json
from tensorflow.keras.models import load_model

# --- Load model dan label ---
model = load_model("expression_model.keras", compile=False)

with open("labels.json", "r") as f:
    class_indices = json.load(f)
labels = {v: k for k, v in class_indices.items()}

# --- Load face detector ---
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# --- Streamlit UI ---
st.title("Deteksi Ekspresi Wajah Real-Time")
st.markdown("Tekan **Kamera ON** untuk memulai deteksi webcam. Tekan **Kamera OFF** untuk menghentikan.")

# State kamera (disimpan di session_state)
if "camera_on" not in st.session_state:
    st.session_state.camera_on = False

# Tombol ON
if st.button("Kamera ON"):
    st.session_state.camera_on = True

# Tombol OFF
if st.button("Kamera OFF"):
    st.session_state.camera_on = False

# Placeholder untuk tampilan frame
frame_window = st.empty()

# --- Jalankan webcam jika kamera_on True ---
if st.session_state.camera_on:
    cap = cv2.VideoCapture(0)
    while st.session_state.camera_on:
        ret, frame = cap.read()
        if not ret:
            st.warning("Tidak dapat membaca dari kamera.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            face_roi = gray[y:y+h, x:x+w]
            face_resized = cv2.resize(face_roi, (48, 48)) / 255.0
            face_input = face_resized.reshape(1, 48, 48, 1)

            pred = model.predict(face_input, verbose=0)
            pred_idx = np.argmax(pred)
            pred_label = labels[pred_idx]
            confidence = np.max(pred)

            label_text = f"{pred_label} ({confidence*100:.1f}%)"
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)
            cv2.putText(frame, label_text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_window.image(frame_rgb)

        # Hentikan jika user menekan OFF saat streaming
        if not st.session_state.camera_on:
            break

    cap.release()
    st.success("Kamera dimatikan.")
else:
    frame_window.image([])  # Kosongkan tampilan
    st.warning("Kamera OFF. Tekan **Kamera ON** untuk memulai.")