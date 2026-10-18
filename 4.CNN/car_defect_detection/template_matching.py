# Template Matching for car part detection
import cv2

# Load input image and template
image = cv2.imread('images/defected/car1.jpg')
template = cv2.imread('templates/logo.jpg')
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray_template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

# Match template
result = cv2.matchTemplate(gray_image, gray_template, cv2.TM_CCOEFF_NORMED)
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

# Draw rectangle around match
h, w = gray_template.shape
top_left = max_loc
bottom_right = (top_left[0] + w, top_left[1] + h)
cv2.rectangle(image, top_left, bottom_right, (0, 255, 0), 2)

# Show result
cv2.imshow('Template Matching', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
