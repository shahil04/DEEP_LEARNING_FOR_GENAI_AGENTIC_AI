import cv2

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def detect_faces(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    output = image.copy()
    for (x,y,w,h) in faces:
        cv2.rectangle(output, (x,y), (x+w, y+h), (255,0,0), 2)
    return output
