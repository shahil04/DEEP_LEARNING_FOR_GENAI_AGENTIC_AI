
import cv2
import mediapipe as mp
import streamlit as st

def run():
    st.subheader("FaceMesh")
    run_btn = st.button("Start FaceMesh")
    if run_btn:
        mp_face_mesh = mp.solutions.face_mesh
        drawing = mp.solutions.drawing_utils
        face_mesh = mp_face_mesh.FaceMesh()
        cap = cv2.VideoCapture(0)
        stframe = st.empty()
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(image)
            if results.multi_face_landmarks:
                for landmarks in results.multi_face_landmarks:
                    drawing.draw_landmarks(frame, landmarks, mp_face_mesh.FACEMESH_TESSELATION)
            stframe.image(frame, channels="BGR")
