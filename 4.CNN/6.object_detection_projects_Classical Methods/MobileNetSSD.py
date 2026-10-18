import cv2
# Load MobileNet SSD pre-trained model
net = cv2.dnn.readNetFromCaffe('deploy.prototxt', 'mobilenet_iter_73000.caffemodel')

# Load image
image = cv2.imread('image.jpg')
(h, w) = image.shape[:2]

# Prepare image for MobileNet SSD
blob = cv2.dnn.blobFromImage(image, 0.007843, (300, 300), 127.5, 127.5, 127.5, 0)

# Perform detection
net.setInput(blob)
detections = net.forward()

# Loop through detections and draw bounding boxes
for i in range(detections.shape[2]):
    confidence = detections[0, 0, i, 2]
    if confidence > 0.2:
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")
        cv2.rectangle(image, (startX, startY), (endX, endY), (0, 255, 0), 2)

cv2.imshow("MobileNet SSD", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
