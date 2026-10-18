import cv2
import pytesseract
import numpy as np

def load_image(image_bytes):
    file_bytes = np.asarray(bytearray(image_bytes), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    return image

def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 3)
    _, thresh = cv2.threshold(gray, 111, 255, cv2.THRESH_BINARY)
    return thresh

def extract_text(image):
    return pytesseract.image_to_string(image)

def extract_data(image):
    return pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
