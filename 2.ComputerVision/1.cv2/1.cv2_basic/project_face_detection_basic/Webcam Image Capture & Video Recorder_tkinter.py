import cv2
import datetime
import threading
import tkinter as tk
from tkinter import Label, Button
from PIL import Image, ImageTk

# Global variables
cap = None
recording = False
out = None

def start_camera():
    global cap
    cap = cv2.VideoCapture(0)
    update_frame()

def update_frame():
    global cap
    if cap is not None and cap.isOpened():
        ret, frame = cap.read()
        if ret:
            # Convert to RGB for Tkinter
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame)
            imgtk = ImageTk.PhotoImage(image=img)
            lbl_video.imgtk = imgtk
            lbl_video.configure(image=imgtk)
        lbl_video.after(10, update_frame)

def capture_image():
    global cap
    if cap is not None and cap.isOpened():
        ret, frame = cap.read()
        if ret:
            filename = f"image_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
            cv2.imwrite(filename, frame)
            status_label.config(text=f"Image saved: {filename}")

def toggle_recording():
    global recording, out, cap
    if not recording:
        if cap is not None and cap.isOpened():
            video_filename = f"video_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.avi"
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out = cv2.VideoWriter(video_filename, fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))
            recording = True
            status_label.config(text="Recording Started...")
            threading.Thread(target=record_video).start()
    else:
        recording = False
        status_label.config(text="Recording Stopped.")

def record_video():
    global recording, out, cap
    while recording and cap.isOpened():
        ret, frame = cap.read()
        if ret:
            out.write(frame)
    if out:
        out.release()

def close_app():
    global cap, recording
    recording = False
    if cap is not None and cap.isOpened():
        cap.release()
    root.destroy()

# Tkinter UI
root = tk.Tk()
root.title("Webcam Image & Video Recorder")

lbl_video = Label(root)
lbl_video.pack()

btn_start = Button(root, text="Start Camera", command=start_camera)
btn_start.pack(pady=5)

btn_capture = Button(root, text="Capture Image", command=capture_image)
btn_capture.pack(pady=5)

btn_record = Button(root, text="Start/Stop Recording", command=toggle_recording)
btn_record.pack(pady=5)

status_label = Label(root, text="Status: Ready")
status_label.pack(pady=5)

btn_exit = Button(root, text="Exit", command=close_app)
btn_exit.pack(pady=5)

root.mainloop()
