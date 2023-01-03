# pb edge detection script

import cv2
import numpy as np

# read image
img = cv2.imread("sample_image.jpeg", cv2.IMREAD_COLOR)

# # display original image
# cv.imshow("original image", img)

# convert to grayscale
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# display grayscale image
cv2.imshow("grayscale image", gray_img)


# Oriented DoG filter using sobel and gaussian kernel
    # gaussian blur
    # getting edges using sobel
    # step1: blur the grayscale image
    # step2: apply sobel edge detection
    # apply canny edge detection

img_blur = cv2.GaussianBlur(gray_img, (3,3), 0)
cv2.imshow('Gaussian Blur', img_blur)


# Sobel Edge Detection
sobelx = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=5) # Sobel Edge Detection on the X axis
sobely = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=5) # Sobel Edge Detection on the Y axis
sobelxy = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=5) # Combined X and Y Sobel Edge Detection

edges = sobelxy
edges2 = np.sqrt(sobelx ** 2 + sobely ** 2)
# # Canny Edge Detection
# edges = cv2.Canny(image=img_blur, threshold1=100, threshold2=200) # Canny Edge Detection/
# Display Canny Edge Detection Image
cv2.imshow('Sobel Edge Detection', edges)
cv2.imshow('Sobel Edge Detection 2', edges2)

cv2.waitKey(0)
cv2.destroyAllWindows()