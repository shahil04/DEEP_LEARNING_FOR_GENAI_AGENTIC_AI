
import cv2
import mediapipe as mp
import streamlit as st

def run():
    st.subheader("Pose Estimation")
    run_btn = st.button("Start Pose Estimation")
    if run_btn:
        mp_pose = mp.solutions.pose
        drawing = mp.solutions.drawing_utils
        pose = mp_pose.Pose()
        cap = cv2.VideoCapture(0)
        stframe = st.empty()
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(image)
            if results.pose_landmarks:
                drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            stframe.image(frame, channels="BGR")
