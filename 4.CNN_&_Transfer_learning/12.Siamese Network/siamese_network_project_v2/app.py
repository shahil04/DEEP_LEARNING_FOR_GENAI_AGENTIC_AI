
import streamlit as st
from inference import predict_similarity
from train import train_model
import torch

st.title("🧠 Siamese Network - Similarity Check")

option = st.radio("Choose an option", ("Inference (Image Matching)", "Training (Train the Model)"))

if option == "Inference (Image Matching)":
    st.subheader("Upload two images to check similarity")
    img1 = st.file_uploader("Upload first image", type=["jpg", "png", "jpeg"], key="1")
    img2 = st.file_uploader("Upload second image", type=["jpg", "png", "jpeg"], key="2")
    
    if st.button("Check Similarity"):
        if img1 and img2:
            pred = predict_similarity(img1, img2)
            st.success(f"Similarity Score: {pred:.2f}")
        else:
            st.error("Please upload both images!")

else:
    st.subheader("Training Mode")
    if st.button("Start Training"):
        loss = train_model()
        st.success(f"Training Finished! Final Loss: {loss:.4f}")
