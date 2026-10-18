import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from streamlit_drawable_canvas import st_canvas

def app():
    # Load model
    model = load_model("mnist_ann_model.h5")

    st.title("MNIST Digit Recognizer")
    st.markdown("Draw a digit below 👇")

    # Canvas for drawing
    canvas_result = st_canvas(
        fill_color="black",
        stroke_width=10,
        stroke_color="white",
        background_color="black",
        height=280,
        width=280,
        drawing_mode="freedraw",
        key="canvas",
    )

    if canvas_result.image_data is not None:
        img = canvas_result.image_data

        # Preprocess the image
        img = cv2.cvtColor(img.astype('uint8'), cv2.COLOR_RGBA2GRAY)
        img = cv2.resize(img, (28, 28))
        img = img / 255.0
        # img = img.reshape(1, 28, 28, 1)
        # Flatten to match ANN input (784,)
        img = img.reshape(1, 784)


        # Predict
        pred = model.predict(img)
        pred_class = np.argmax(pred)

        st.subheader(f"Predicted Digit: {pred_class}")
app()