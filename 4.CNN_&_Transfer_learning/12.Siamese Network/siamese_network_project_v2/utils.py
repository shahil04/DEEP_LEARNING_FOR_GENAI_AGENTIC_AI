
import cv2
import numpy as np
from PIL import Image
import torch
from torchvision import transforms

def preprocess(image_file):
    img = Image.open(image_file).convert('L')
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor()
    ])
    return transform(img).unsqueeze(0)
