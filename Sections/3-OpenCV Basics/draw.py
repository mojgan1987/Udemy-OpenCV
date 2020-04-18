import cv2
import numpy as np

import matplotlib.pyplot as plt
%matplotlib inline

blank_image = np.zeros(shape=(512,512,3), dtype=np.int16)
blank_image.shape
plt.imshow(blank_image)

cv2.rectangle(blank_image, pt1=(384,0), pt2=(500,150), color=(0,255,0), thickness=10)
plt.imshow(blank_image)
cv2.rectangle(blank_image, pt1=(200, 200), pt2=(300,300), color=(0,0,255), thickness=10)
plt.imshow(blank_image)

cv2.circle(img=blank_image, center=(100,100),radius=50, color=(255,0,0), thickness=5)
plt.imshow(blank_image)

cv2.circle(img=blank_image, center=(400,400),radius=50, color=(255,0,0), thickness=-1)
plt.imshow(blank_image)

cv2.line(blank_image, pt1=(0,0),pt2=(512,512), color=(102,255,255), thickness=5)
plt.imshow(blank_image)

font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(blank_image, text='hello',org=(10,500),fontFace=font, fontScale=4, color=(255,255,255), thickness=3, lineType = cv2.LINE_AA)
plt.imshow(blank_image)

blank_img2 = np.zeros(shape=(512,512,3), dtype=np.int32)
plt.imshow(blank_img2)
vertices = np.array([[100,300],[200,200],[400,300],[200,400]],dtype=np.int32)
vertices
vertices.shape

# for OpenCV
pts = vertices.reshape((-1,1,2))
pts.shape
pts

cv2.polylines(blank_img2, [pts],isClosed=True, color=(255,0,0), thickness=5)
plt.imshow(blank_img2)
