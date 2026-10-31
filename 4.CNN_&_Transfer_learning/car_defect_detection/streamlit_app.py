import streamlit as st
import cv2
import numpy as np

st.set_page_config(page_title="Car Defect Detection", layout="centered")
st.title("🚗 Car Defect Detection")
st.markdown("Detect car part issues using **Template Matching** and **Edge Detection** techniques.")

# Choose method
option = st.radio("Choose Detection Method", ["Template Matching", "Edge Detection"])

# Upload image
uploaded_file = st.file_uploader("Upload a car image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Read uploaded file as image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    if option == "Template Matching":
        st.subheader("🧩 Template Matching")
        try:
            template = cv2.imread('templates/logo.jpg', 0)
            if template is None:
                st.error("Template image not found in 'templates/logo.jpg'")
            else:
                result = cv2.matchTemplate(gray_image, template, cv2.TM_CCOEFF_NORMED)
                _, _, _, max_loc = cv2.minMaxLoc(result)
                h, w = template.shape
                cv2.rectangle(image, max_loc, (max_loc[0] + w, max_loc[1] + h), (0, 255, 0), 2)
                st.success("Template matched and highlighted.")
        except Exception as e:
            st.error(f"Error during template matching: {e}")

        # Show image
        st.image(image, channels="BGR", caption="Template Matching Result", use_column_width=True)

    elif option == "Edge Detection":
        st.subheader("🪞 Edge Detection")
        low = st.slider("Canny Edge Threshold - Low", 0, 255, 50)
        high = st.slider("Canny Edge Threshold - High", 0, 255, 150)

        blurred = cv2.GaussianBlur(gray_image, (5, 5), 0)
        edges = cv2.Canny(blurred, low, high)

        # Show grayscale result
        st.image(edges, caption="Edge Detection Result", use_column_width=True, clamp=True)
