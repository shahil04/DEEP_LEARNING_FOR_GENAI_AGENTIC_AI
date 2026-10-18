import streamlit as st
from ocr_utils import *
import numpy as np
import cv2

st.set_page_config(page_title="OCR Text Recognition")
st.title("📝 OCR Text Recognition with Pytesseract")

uploaded_file = st.file_uploader("Upload an image with text", type=["jpg", "jpeg", "png"])
if uploaded_file:
    image = load_image(uploaded_file.read())
    st.image(image, caption="Original Image", use_column_width=True)

    thresh = preprocess_image(image)
    st.image(thresh, caption="Preprocessed Image", use_column_width=True)

    st.subheader("📄 Extracted Text")
    text = extract_text(thresh)
    st.code(text)

    st.subheader("🔍 Word Detection with Bounding Boxes")
    data = extract_data(thresh)
    n_boxes = len(data['level'])
    for i in range(n_boxes):
        (x, y, w, h) = (data['left'][i], data['top'][i], data['width'][i], data['height'][i])
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    st.image(image, caption="Detected Words", use_column_width=True)
