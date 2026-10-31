import numpy as np
import cv2
import os

os.makedirs("images/defected", exist_ok=True)
os.makedirs("templates", exist_ok=True)

# White car dummy image
cv2.imwrite("images/defected/car1.jpg", np.full((100, 200, 3), 255, dtype=np.uint8))

# Black logo dummy image
cv2.imwrite("templates/logo.jpg", np.full((20, 40, 3), 0, dtype=np.uint8))
