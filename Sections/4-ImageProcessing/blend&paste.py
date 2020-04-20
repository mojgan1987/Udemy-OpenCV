import numpy as np
import cv2

img1 = cv2.imread('C://Users//Mojgan//Documents//Python/Udemy/OpenCV/Resources/original/Computer-Vision-with-Python/DATA/dog_backpack.jpg')
img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
img2 = cv2.imread('C://Users//Mojgan//Documents//Python/Udemy/OpenCV/Resources/original/Computer-Vision-with-Python/DATA/watermark_no_copy.png')
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

import matplotlib.pyplot as plt
plt.imshow(img1)
plt.imshow(img2)

img1.shape
img2.shape

# blendinhg images of the same size
img1 = cv2.resize(img1, (1200, 1200))
img2 = cv2.resize(img2, (1200, 1200))

blended = cv2.addWeighted(src1=img1, alpha=0.5, src2=img2, beta=0.5, gamma=0)
plt.imshow(blended)

blended = cv2.addWeighted(src1=img1, alpha=0.8, src2=img2, beta=0.1, gamma=0)
plt.imshow(blended)

#overlay a small image on top of a larger image (no blending)
# numpy reassignment
img1 = cv2.imread('C://Users//Mojgan//Documents//Python/Udemy/OpenCV/Resources/original/Computer-Vision-with-Python/DATA/dog_backpack.jpg')
img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
img2 = cv2.imread('C://Users//Mojgan//Documents//Python/Udemy/OpenCV/Resources/original/Computer-Vision-with-Python/DATA/watermark_no_copy.png')
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

img2 = cv2.resize(img2, (600, 600))
plt.imshow(img2)

large_img = img1
small_img = img2

x_offset = 0
y_offset = 0
x_end = x_offset + small_img.shape[1]
y_end = y_offset + small_img.shape[0]

large_img[y_offset:y_end, x_offset:x_end] = small_img
plt.imshow(large_img)

# how to blend together images of different sizes
img1 = cv2.imread('C://Users//Mojgan//Documents//Python/Udemy/OpenCV/Resources/original/Computer-Vision-with-Python/DATA/dog_backpack.jpg')
img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
img2 = cv2.imread('C://Users//Mojgan//Documents//Python/Udemy/OpenCV/Resources/original/Computer-Vision-with-Python/DATA/watermark_no_copy.png')
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

img2 = cv2.resize(img2, (600, 600))
plt.imshow(img2)

# paste at the bottom right with white shiny borders
img1.shape
x_offset = 934 - 600
y_offset = 1401 - 600

# create ROI
img2.shape

rows, cols, channels = img2.shape
roi = img1[y_offset:1401, x_offset:943]
plt.imshow(roi)

img2gray = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
plt.imshow(img2gray, cmap='gray')

mask_inverse = cv2.bitwise_not(img2gray)
plt.imshow(mask_inverse, cmap='gray')

mask_inverse.shape # hass no color channel
white_background = np.full(img2.shape, 255,dtype=np.uint8)
white_background.shape
plt.imshow(white_background)

# creat bachground
bkg = cv2.bitwise_or(white_background,white_background, mask=mask_inverse)
bkg.shape
plt.imshow(bkg)

# create foreground
fgd = cv2.bitwise_or(img2,img2, mask=mask_inverse)
plt.imshow(fgd)

final_roi = cv2.bitwise_or(roi, fgd)
plt.imshow(final_roi)

# blending images
large_img = img1
small_img = final_roi
large_img[y_offset:y_offset+small_img.shape[0], x_offset:x_offset+small_img.shape[1]] = small_img
plt.imshow(large_img)



