import cv2
import cv2 as cv
from matplotlib import pyplot as plt
import numpy as np

# detector parameters
block_size = 3
sobel_size = 3
k = 0.06

image = cv.imread('imgs/Scenery.jpeg')

print(image.shape)
height = image.shape[0]
width = image.shape[1]
channels = image.shape[2]
print("width: %s  height: %s  channels: %s"%(width, height, channels))
   
gray_img = cv.cvtColor(image, cv2.COLOR_BGR2GRAY)


# modify the data type setting to 32-bit floating point 
gray_img = np.float32(gray_img)

# detect the corners with appropriate values as input parameters
corners_img = cv.cornerHarris(gray_img, block_size, sobel_size, k)

# result is dilated for marking the corners, not necessary
kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3, 3))
dst = cv.dilate(corners_img, kernel)

# Threshold for an optimal value, marking the corners in Green
#image[corners_img>0.01*corners_img.max()] = [0,0,255]

for r in range(height):
        for c in range(width):
            pix=dst[r,c]
            if pix>0.05*dst.max():
               cv2.circle(image,(c,r),5,(0,0,255),0)

image = cv.cvtColor(image, cv2.COLOR_BGR2RGB)
plt.imshow(image)
plt.show()
