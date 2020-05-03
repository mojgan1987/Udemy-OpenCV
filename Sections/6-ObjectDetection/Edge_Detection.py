# Canny method: multi-stage

# 1. apply guassian to smooth noise (or other filters)
# 2. find intensity gradient of image
# 3. apply non-maximum supression
# 4. apply double treshold to determine potential edges
# 5. track by suppressing all weak edges and finding the strong one

# for high resolution imgaes, apply a bluring

import cv2
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

img = cv2.imread('../DATA/sammy_face.jpg')
plt.imshow(img)

edges = cv2.Canny(image=img, threshold1=127, threshold2=127)
plt.imshow(edges)

# method 1: blurring image
# method 2: playing around treshold values
# here combination

edges = cv2.Canny(image=img, threshold1=127, threshold2=255)
plt.imshow(edges)

# cal median pixel value
med_val = np.median(img)
med_val

# lower threshold to ether 0 or 70% of the med value
lower = int(max(0,0.7*med_val))
# upper threshold to either 130% of median (30% abouvbe median) or whichever is smaller
upper = int(min(255, 1.3*med_val))


edges = cv2.Canny(image=img, threshold1=lower, threshold2=upper+100)
plt.imshow(edges)


# blurr
blurred_img = cv2.blur(img, ksize=(7,7))

edges = cv2.Canny(image=blurred_img, threshold1=lower, threshold2=upper+50)
plt.imshow(edges)
