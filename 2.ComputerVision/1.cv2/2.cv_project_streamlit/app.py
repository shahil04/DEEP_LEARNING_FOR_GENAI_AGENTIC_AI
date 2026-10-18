import streamlit as st
import cv2
import numpy as np
from utils import basic_ops, filters, shapes, face_detection, cartoonify
from PIL import Image
import os

st.set_page_config(page_title="OpenCV Streamlit App", layout="wide")

st.title("🖼 OpenCV Image Processing App")

# Ensure media folder exists
if not os.path.exists('media'):
    os.makedirs('media')

# Upload or Capture Image
st.sidebar.header("Upload or Capture Image")
input_option = st.sidebar.radio("Select Input Method", ("Upload Image", "Capture from Camera"))

image = None
if input_option == "Upload Image":
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg","jpeg","png"])
    if uploaded_file is not None:
        image = np.array(Image.open(uploaded_file))
elif input_option == "Capture from Camera":
    captured_image = st.camera_input("Capture an image")
    if captured_image is not None:
        image = np.array(Image.open(captured_image))

if image is not None:
    st.image(image, caption="Original Image", use_column_width=True)

    # Sidebar for Operations
    st.sidebar.header("Select Operation")
    operation_category = st.sidebar.selectbox(
        "Choose a Category",
        ("Basic Image Operations", "Image Filters", "Shapes & Contours", "Face Detection", "Cartoonify")
    )

    processed_image = None

    if operation_category == "Basic Image Operations":
        col1, col2, col3, col4 = st.columns(4)
        if col1.button("Resize"):
            processed_image = basic_ops.resize_image(image, 0.5)
        if col2.button("Rotate"):
            processed_image = basic_ops.rotate_image(image, 45)
        if col3.button("Flip"):
            processed_image = basic_ops.flip_image(image)
        if col4.button("Grayscale"):
            processed_image = basic_ops.convert_to_grayscale(image)

    elif operation_category == "Image Filters":
        col1, col2 = st.columns(2)
        if col1.button("Gaussian Blur"):
            processed_image = filters.gaussian_blur(image)
        if col2.button("Edge Detection"):
            processed_image = filters.canny_edges(image)

    elif operation_category == "Shapes & Contours":
        if st.button("Detect Contours"):
            processed_image = shapes.detect_contours(image)

    elif operation_category == "Face Detection":
        if st.button("Detect Faces"):
            processed_image = face_detection.detect_faces(image)

    elif operation_category == "Cartoonify":
        if st.button("Cartoonify"):
            processed_image = cartoonify.cartoonify_image(image)

    if processed_image is not None:
        col1, col2 = st.columns(2)
        col1.image(image, caption="Original", use_column_width=True)
        col2.image(processed_image, caption="Processed", use_column_width=True)

        # Save and Download
        output_path = os.path.join("media", "processed_image.jpg")
        cv2.imwrite(output_path, cv2.cvtColor(processed_image, cv2.COLOR_RGB2BGR))
        with open(output_path, "rb") as file:
            st.download_button("Download Processed Image", file, "processed_image.jpg")
