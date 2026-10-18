import streamlit as st
import cv2
import numpy as np

# Load pre-trained models (Haar Cascade, MobileNet SSD, YOLO)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
net = cv2.dnn.readNetFromCaffe('models/deploy.prototxt', 'models/mobilenet_iter_73000.caffemodel')
yolo_net = cv2.dnn.readNet("models/yolov3.weights", "models/yolov3.cfg")
layer_names = yolo_net.getLayerNames()
output_layers = [layer_names[i - 1] for i in yolo_net.getUnconnectedOutLayers()]

# Function to perform face detection using Haar Cascade
def haar_cascade_detection(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
    return image

# Function to perform object detection using MobileNet SSD
def mobilenet_ssd_detection(image):
    h, w = image.shape[:2]
    blob = cv2.dnn.blobFromImage(image, 0.007843, (300, 300), 127.5, 127.5, 127.5, 0)
    net.setInput(blob)
    detections = net.forward()
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.2:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            cv2.rectangle(image, (startX, startY), (endX, endY), (0, 255, 0), 2)
    return image

# Function to perform object detection using YOLO
def yolo_detection(image):
    h, w = image.shape[:2]
    blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    yolo_net.setInput(blob)
    outs = yolo_net.forward(output_layers)
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * w)
                center_y = int(detection[1] * h)
                box_width = int(detection[2] * w)
                box_height = int(detection[3] * h)
                x = int(center_x - box_width / 2)
                y = int(center_y - box_height / 2)
                cv2.rectangle(image, (x, y), (x + box_width, y + box_height), (0, 255, 0), 2)
    return image

# Streamlit UI
st.title('Object Detection Streamlit App')

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

# Radio buttons for object detection methods
method = st.radio(
    "Select object detection method",
    ('Haar Cascade', 'MobileNet SSD', 'YOLO')
)

if uploaded_file is not None:
    # Read image from file
    image = np.array(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(image, 1)

    # Perform selected object detection method
    if method == 'Haar Cascade':
        result_image = haar_cascade_detection(image)
    elif method == 'MobileNet SSD':
        result_image = mobilenet_ssd_detection(image)
    elif method == 'YOLO':
        result_image = yolo_detection(image)

    # Display result
    st.image(result_image, caption='Processed Image', channels="BGR")
