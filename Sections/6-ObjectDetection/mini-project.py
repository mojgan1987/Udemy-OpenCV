import cv2
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

car_plate = plt.imread('../data/car_plate.jpg')

def display(img):
  fig = plt.figure(figsize=(11,12))
  ax = fig.add_subplot(111)
  ax.imshow(img, cmap='gray')
