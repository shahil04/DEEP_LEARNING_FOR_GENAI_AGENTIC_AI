
import cv2
import mediapipe as mp
import streamlit as st

def run():
    st.subheader("Hand Tracking")
    run_btn = st.button("Start Hand Tracking")
    if run_btn:
        mp_hands = mp.solutions.hands
        drawing = mp.solutions.drawing_utils
        hands = mp_hands.Hands()
        cap = cv2.VideoCapture(0)
        stframe = st.empty()
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(image)
            if results.multi_hand_landmarks:
                for handLms in results.multi_hand_landmarks:
                    drawing.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)
            stframe.image(frame, channels="BGR")
