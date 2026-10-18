# Edge Detection for crack/scratch detection
import cv2

# Load image
image = cv2.imread('images/defected/car1.jpg', 0)

# Apply Gaussian blur and Canny edge detection
blurred = cv2.GaussianBlur(image, (5, 5), 0)
edges = cv2.Canny(blurred, 50, 150)

# Show result
cv2.imshow('Edge Detection', edges)
cv2.waitKey(0)
cv2.destroyAllWindows()
