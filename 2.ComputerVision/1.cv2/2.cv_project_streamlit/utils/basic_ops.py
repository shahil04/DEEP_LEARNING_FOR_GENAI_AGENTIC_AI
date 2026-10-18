import cv2
import numpy as np

def resize_image(image, scale=0.5):
    return cv2.resize(image, None, fx=scale, fy=scale)

def rotate_image(image, angle=45):
    h, w = image.shape[:2]
    M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1)
    return cv2.warpAffine(image, M, (w, h))

def flip_image(image):
    return cv2.flip(image, 1)

def convert_to_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
