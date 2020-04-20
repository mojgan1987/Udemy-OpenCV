# Sobel–Feldman operator

import cv2
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

def display_img(img):
    fig = plt.figure(figsize=(12,10))
    ax = fig.add_subplot(111)
    ax.imshow(img, cmap='gray')
    
img = cv2.imread('C://Users//Mojgan//Documents//Python/Udemy/OpenCV/Resources/original/Computer-Vision-with-Python/DATA/sudoku.jpg',0)
display_img(img)

sobelx = cv2.Sobel(img,cv2.CV_64F,1,0, ksize=5) # cal x gradient
display_img(sobelx)

sobelx = cv2.Sobel(img,cv2.CV_64F,0,1, ksize=5) # cal y gradient
display_img(sobelx)


# laplace
laplacian = cv2.Laplacian(img, cv2.CV_64F)
display_img(laplacian)

ret, th1 = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY)
display_img(th1)


# blending x,y
blended = cv2.addWeighted(src1=sobelx, alpha=0.5,src2=sobely, beta=0.5, gamma=0)
display_img(blended)

kernel = np.ones((4,4), np.uint8)
gradient = cv2.morphologyEx(blended, cv2.MORPH_GRADIENT, kernel)
display_img(gradient)
