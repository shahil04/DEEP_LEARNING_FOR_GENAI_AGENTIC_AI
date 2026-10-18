import cv2

def stitch_images(images):
    stitcher = cv2.Stitcher_create()
    (status, stitched) = stitcher.stitch(images)
    return stitched if status == 0 else images[0]
