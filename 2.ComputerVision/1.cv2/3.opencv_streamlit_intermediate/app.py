
import streamlit as st
import cv2
from utils import motion_tracking, document_scanner, qr_barcode_scanner, image_stitching, hand_gesture, color_detection, object_measurement
from PIL import Image
import numpy as np
import tempfile
import os

st.set_page_config(page_title="Intermediate OpenCV Project", layout="wide")

st.title("OpenCV Intermediate Project - Streamlit")

menu = ["Home", "Motion Tracking", "Document Scanner", "QR/Barcode Scanner", "Image Stitching", "Hand Gesture", "Color Detection", "Object Measurement"]
choice = st.sidebar.selectbox("Select Feature", menu)

if choice == "Home":
    st.write("### Welcome to OpenCV Intermediate Project")
    st.write("Choose a feature from the sidebar to get started.")

elif choice == "Motion Tracking":
    st.subheader("Motion Detection & Object Tracking (Webcam)")
    run = st.checkbox('Start Webcam')
    if run:
        cap = cv2.VideoCapture(0)
        stframe = st.empty()
        while run:
            ret, frame = cap.read()
            if not ret:
                break
            processed = motion_tracking.detect_motion(frame)
            stframe.image(processed, channels="BGR")
        cap.release()

elif choice == "Document Scanner":
    st.subheader("Upload an Image for Document Scanning")
    img_file = st.file_uploader("Upload Image", type=['jpg','jpeg','png'])
    if img_file:
        img = Image.open(img_file)
        img_np = np.array(img)
        scanned = document_scanner.scan_document(img_np)
        col1, col2 = st.columns(2)
        col1.image(img, caption="Original")
        col2.image(scanned, caption="Scanned")

elif choice == "QR/Barcode Scanner":
    st.subheader("QR & Barcode Detection")
    img_file = st.file_uploader("Upload Image", type=['jpg','jpeg','png'])
    if img_file:
        img = Image.open(img_file)
        img_np = np.array(img)
        decoded_img, codes = qr_barcode_scanner.decode_qr_barcode(img_np)
        col1, col2 = st.columns(2)
        col1.image(img, caption="Original")
        col2.image(decoded_img, caption="Processed")
        st.write("Detected Codes:", codes)

elif choice == "Image Stitching":
    st.subheader("Upload Multiple Images for Panorama Stitching")
    img_files = st.file_uploader("Upload Images", type=['jpg','jpeg','png'], accept_multiple_files=True)
    if img_files:
        imgs = [np.array(Image.open(f)) for f in img_files]
        stitched = image_stitching.stitch_images(imgs)
        st.image(stitched, caption="Panorama")

elif choice == "Hand Gesture":
    st.subheader("Hand Gesture Recognition (Webcam)")
    run = st.checkbox('Start Webcam')
    if run:
        cap = cv2.VideoCapture(0)
        stframe = st.empty()
        while run:
            ret, frame = cap.read()
            if not ret:
                break
            processed = hand_gesture.detect_gesture(frame)
            stframe.image(processed, channels="BGR")
        cap.release()

elif choice == "Color Detection":
    st.subheader("Color Detection (Webcam)")
    run = st.checkbox('Start Webcam')
    if run:
        cap = cv2.VideoCapture(0)
        stframe = st.empty()
        while run:
            ret, frame = cap.read()
            if not ret:
                break
            processed = color_detection.detect_color(frame)
            stframe.image(processed, channels="BGR")
        cap.release()

elif choice == "Object Measurement":
    st.subheader("Upload Image for Object Measurement")
    img_file = st.file_uploader("Upload Image", type=['jpg','jpeg','png'])
    if img_file:
        img = Image.open(img_file)
        img_np = np.array(img)
        measured = object_measurement.measure_object(img_np)
        col1, col2 = st.columns(2)
        col1.image(img, caption="Original")
        col2.image(measured, caption="Measured")
