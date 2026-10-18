import streamlit as st
import cv2
import os
import datetime
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import av

# Folder to save media
SAVE_DIR = "media"
os.makedirs(SAVE_DIR, exist_ok=True)

st.title("📷 Webcam Image & Video Recorder")

status_text = st.empty()

# Video transformer class
class VideoTransformer(VideoTransformerBase):
    def __init__(self):
        self.frame = None

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        self.frame = img
        return img

# Start webcam
ctx = webrtc_streamer(key="example", video_transformer_factory=VideoTransformer)

# Capture image
if st.button("📸 Capture Image"):
    if ctx.video_transformer and ctx.video_transformer.frame is not None:
        img = ctx.video_transformer.frame
        filename = os.path.join(SAVE_DIR, f"image_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg")
        cv2.imwrite(filename, img)
        status_text.success(f"✅ Image saved: {filename}")
    else:
        status_text.error("❌ No frame captured.")

# Initialize recording states
if "recording" not in st.session_state:
    st.session_state.recording = False
    st.session_state.out = None
    st.session_state.video_filename = ""

# Start/Stop recording
if st.button("🎥 Start/Stop Recording"):
    if not st.session_state.recording:
        st.session_state.video_filename = os.path.join(SAVE_DIR, f"video_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4")
        st.session_state.recording = True
        status_text.info("🔴 Recording started...")
    else:
        st.session_state.recording = False
        if st.session_state.out:
            st.session_state.out.release()
        status_text.success(f"✅ Recording stopped. Saved as: {st.session_state.video_filename}")

# Write frames if recording
if st.session_state.recording and ctx.video_transformer and ctx.video_transformer.frame is not None:
    img = ctx.video_transformer.frame
    if st.session_state.out is None:
        height, width, _ = img.shape
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        st.session_state.out = cv2.VideoWriter(st.session_state.video_filename, fourcc, 20.0, (width, height))
    st.session_state.out.write(img)

st.write("📂 Saved files will appear in the 'media' folder.")
