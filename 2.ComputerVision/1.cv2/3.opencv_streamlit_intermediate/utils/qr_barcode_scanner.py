import cv2
from pyzbar import pyzbar

def decode_qr_barcode(img):
    codes = pyzbar.decode(img)
    return img, [code.data.decode('utf-8') for code in codes]
