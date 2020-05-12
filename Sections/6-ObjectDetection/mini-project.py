import cv2
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

car_plate = plt.imread('../data/car_plate.jpg')

def display(img):
  fig = plt.figure(figsize=(11,12))
  ax = fig.add_subplot(111)
  ax.imshow(img, cmap='gray')
  
  
display(img)

# 
plate_cascade = cv2.CascadeClassifier('../data/haarcascades/haarcascade_russian_plate_number.xml')

def detect_plate(img):
    img_cpy = img.copy()
    
    plate_rect = plate_cascade.detectMultiScale(img_cpy,scaleFactor=1.2, minNeighbors=5)
    
    for x,y,w,h in plate_rect:
        cv2.rectangle(img_cpy, (x,y), (x+w,y+h),(0,0,255),3)
  
    return img_cpy
  
  result = detect_plate(img)
  display(result)

# Blurr the plate number
def detect_and_blur_plate(img):
    
    plate_img = img.copy()
    roi = img.copy()
    
    plate_rect = plate_cascade.detectMultiScale(plate_img, scaleFactor=1.3, minNeighbors=3)
    
    for (x,y,w,h) in plate_rect:
        roi = roi[y:y+h,x:x+w]
        blurred_roi = cv2.medianBlur(roi,7)
        
        plate_img[y:y+h,x:x+w] = blurred_roi
    
    return plate_img
  
  result = detect_and_blur_plate(img)
  display(result)
  
