import cv2
import matplotlib.pyplot as plt
import numpy as np
%matplotlib inline

def display_img(img,cmap=None):
    fig = plt.figure(figsize=(12,10))
    ax = fig.add_subplot(111)
    ax.imshow(img,cmap)
    
giraffe = cv2.imread('C://Users//Mojgan//Documents//Python/Udemy/OpenCV/Resources/original/Computer-Vision-with-Python/DATA/giraffes.jpg')
show_giraffe = cv2.cvtColor(giraffe, cv2.COLOR_BGR2RGB)
giraffe_gray = cv2.imread('C://Users//Mojgan//Documents//Python/Udemy/OpenCV/Resources/original/Computer-Vision-with-Python/DATA/giraffes.jpg',0)
display_img(giraffe_gray,cmap='gray')
display_img(show_giraffe)

# apply binary threshold
ret, th1 = cv2.threshold(giraffe_gray, 127, 255, cv2.THRESH_BINARY)
display_img(th1, cmap='gray')

# convert colorspace to HSV
giraffe_hsv = cv2.cvtColor(giraffe, cv2.COLOR_BGR2HSV)
display_img(giraffe_hsv)

# use 2-D Convolution to blur (k size 4)
kernel = np.ones((4,4), dtype=np.float32) / 10
result = cv2.filter2D(show_giraffe,-1, kernel)
display_img(result)
# also brighter!

# Horizontal Sobe on the grayscale
sobelx = cv2.Sobel(giraffe_gray, cv2.CV_64F, 1, 0,ksize=5)
display_img(sobelx,cmap='gray')

# histogram
color = ['B','G','R']
for i,col in enumerate(color):
    histr = cv2.calcHist([giraffe], [i],None, [256],[0,256])
    plt.plot(histr,color=col)
    
    
