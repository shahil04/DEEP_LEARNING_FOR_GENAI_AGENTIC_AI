import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from streamlit_drawable_canvas import st_canvas
import matplotlib.pyplot as plt

def app():
    # Load model
    model = load_model("mnist_ann_model.h5")

    st.title("MNIST Digit Recognizer")
    st.markdown("🖊️ Draw a digit below and click **Submit** to predict.")

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

    # Buttons
    col1, col2 = st.columns(2)
    submit = col1.button("✅ Submit")
    clear = col2.button("🧹 Clear Canvas")

    if submit:
        if canvas_result is not None and hasattr(canvas_result, "image_data"):
            image_data = canvas_result.image_data

            if image_data is not None:
                try:
                    # Preprocessing pipeline
                    img = image_data.astype("uint8")
                    img = cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)

                    # Threshold to remove light strokes
                    _, img = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY)

                    # Resize to MNIST format
                    img = cv2.resize(img, (28, 28), interpolation=cv2.INTER_AREA)

                    # Center the digit using contours (basic method)
                    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    if contours:
                        x, y, w, h = cv2.boundingRect(contours[0])
                        digit = img[y:y+h, x:x+w]
                        padded = cv2.copyMakeBorder(digit, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=0)
                        img = cv2.resize(padded, (28, 28))

                    # Normalize and reshape
                    img = img.astype("float32") / 255.0
                    img_flat = img.reshape(1, 784)

                    # Show processed image
                    st.image(img, width=150, caption="Processed Digit (Input to Model)")

                    # Prediction
                    pred = model.predict(img_flat)
                    pred_class = np.argmax(pred)

                    st.subheader(f"✅ Predicted Digit: {pred_class}")

                    # Show probability chart
                    st.markdown("### 📊 Prediction Probabilities")
                    fig, ax = plt.subplots()
                    ax.bar(range(10), pred[0], color="skyblue")
                    ax.set_xticks(range(10))
                    ax.set_xlabel("Digit")
                    ax.set_ylabel("Probability")
                    st.pyplot(fig)

                except Exception as e:
                    st.error(f"⚠️ Error during prediction: {e}")
            else:
                st.warning("⚠️ Please draw a digit before submitting.")
        else:
            st.warning("⚠️ Canvas is not ready yet.")

    if clear:
        # Streamlit doesn't have a built-in way to clear canvas, so we rerun
        st.experimental_rerun()

app()
