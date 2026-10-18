
import streamlit as st
from pages import face_mesh, hand_tracking, pose_estimation, holistic_model

st.title("MediaPipe Multi-Model Demo")
option = st.radio("Choose a model", ("FaceMesh", "Hand Tracking", "Pose Estimation", "Holistic Model"))

if option == "FaceMesh":
    face_mesh.run()
elif option == "Hand Tracking":
    hand_tracking.run()
elif option == "Pose Estimation":
    pose_estimation.run()
elif option == "Holistic Model":
    holistic_model.run()
