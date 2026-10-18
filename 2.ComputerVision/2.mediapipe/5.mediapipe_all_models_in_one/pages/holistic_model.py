
import cv2
import mediapipe as mp
import streamlit as st

def run():
    st.subheader("Holistic Model")
    run_btn = st.button("Start Holistic")
    if run_btn:
        mp_holistic = mp.solutions.holistic
        drawing = mp.solutions.drawing_utils
        holistic = mp_holistic.Holistic()
        cap = cv2.VideoCapture(0)
        stframe = st.empty()
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(image)
            if results.face_landmarks:
                drawing.draw_landmarks(frame, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION)
            if results.left_hand_landmarks:
                drawing.draw_landmarks(frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
            if results.right_hand_landmarks:
                drawing.draw_landmarks(frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
            if results.pose_landmarks:
                drawing.draw_landmarks(frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
            stframe.image(frame, channels="BGR")
