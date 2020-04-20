import cv2
import matplotlib.pyplot as plt
%matplotlib inline

img = cv2.imread('C://Users//Mojgan//Documents//Python/Udemy/OpenCV/Resources/original/Computer-Vision-with-Python/DATA/dog_backpack.jpg')

img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
plt.imshow(img)
