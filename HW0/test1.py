import numpy as np
import cv2

# Define the number of scales and orientations
s = 2
o = 16

# Scaling factor for the filters
k = 1

# Create an empty list to store the filters
filters = []

for i in range(s):
    # Calculate the standard deviation for the current scale
    std = (k / np.sqrt(2)) * (2**(1/s))**i

    # Create a Gaussian kernel with the current standard deviation
    gaussian_kernel = cv2.getGaussianKernel(5, std)

    # Convolve the Gaussian kernel with a Sobel filter to create a DoG filter
    dog_filter_x = cv2.filter2D(gaussian_kernel, -1, np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]))
    dog_filter_y = cv2.filter2D(gaussian_kernel, -1, np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]))

    # Rotate the DoG filter o times to create filters for all orientations
    for j in range(o):
        angle = 360 / o * j
        rot_mat = cv2.getRotationMatrix2D((2, 2), angle, 1)
        rotated_filter_x = cv2.warpAffine(dog_filter_x, rot_mat, (5, 5))
        rotated_filter_y = cv2.warpAffine(dog_filter_y, rot_mat, (5, 5))
        filters.append((rotated_filter_x, rotated_filter_y))

# Print the filters
for i in range(len(filters)):
    print("Filter", i+1)
    print(filters)

# Load the image
img = cv2.imread('sample_image.jpeg')

# Iterate through the filters in the filter bank
for i in range(len(filters)):
    # Apply the filter to the image
    output_x = cv2.filter2D(img, -1, filters[i][0])
    output_y = cv2.filter2D(img, -1, filters[i][1])
    output = np.sqrt(output_x**2 + output_y**2)
    
    # Save the output image
    output = output.astype(np.uint8)
    cv2.imwrite('output_image.jpeg'.format(i+1), output)

img_output = cv2.imread("output_image.jpeg", cv2.IMREAD_COLOR)
cv2.imshow('DoG filter', img_output)