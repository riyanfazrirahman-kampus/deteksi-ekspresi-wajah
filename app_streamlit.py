import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import av
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import json

# Load model dan label
model = load_model("expression_model.keras", compile=False)
with open("labels.json", "r") as f:
    class_indices = json.load(f)
labels = {v: k for k, v in class_indices.items()}

# Video transformer
class EmotionDetector(VideoTransformerBase):
    def transform(self, frame: av.VideoFrame) -> np.ndarray:
        img = frame.to_ndarray(format="bgr24")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml").detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            face_roi = gray[y:y+h, x:x+w]
            face_resized = cv2.resize(face_roi, (48, 48)) / 255.0
            face_input = face_resized.reshape(1, 48, 48, 1)

            pred = model.predict(face_input, verbose=0)
            pred_idx = np.argmax(pred)
            pred_label = labels[pred_idx]
            confidence = np.max(pred)

            label_text = f"{pred_label} ({confidence*100:.1f}%)"
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 255), 2)
            cv2.putText(img, label_text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)

        return img

st.title("Deteksi Ekspresi Wajah - Streamlit WebRTC")
webrtc_streamer(key="example", video_transformer_factory=EmotionDetector)
