import cv2

def gaussian_blur(image, ksize=(15,15)):
    return cv2.GaussianBlur(image, ksize, 0)

def canny_edges(image, threshold1=100, threshold2=200):
    return cv2.Canny(image, threshold1, threshold2)
