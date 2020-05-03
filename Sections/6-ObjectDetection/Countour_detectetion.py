# internal vs external countour

import cv2
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

img = cv2.imread('../DATA/internal_external.png', 0) # reading grayscale
img.shape

plt.imshow(img, cmap='gray')

contours, heirarchy = cv2.findContours(img, cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE) # both internal and external

# creating a black bg to draw extracted contours
external_contours = np.zeros((652,1080))


# external contours
for i in range(len(contours)):
    # external contours
    if heirarchy[0][i][3] == -1:
        cv2.drawContours(external_contours, contours,i,255,-1)
plt.imshow(external_contours, cmap='gray')        


# internal contours
internal_contours = np.zeros((652,1080))

for i in range(len(contours)):
    # internal contours
    if heirarchy[0][i][3] != -1:
        cv2.drawContours(internal_contours, contours,i,255,-1)
plt.imshow(internal_contours, cmap='gray')        


# grouping
grouped_contours = np.zeros((652,1080))

for i in range(len(contours)):
    # internal contours
    if heirarchy[0][i][3] == 4: # group
        cv2.drawContours(internal_contours, contours,i,255,-1)
plt.imshow(internal_contours, cmap='gray')        
