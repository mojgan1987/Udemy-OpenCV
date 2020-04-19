import cv2
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

###########################
###### FUNCTION ###########
###########################
def draw_circle(event, x,y, falgs, params):
    
    if event == cv2.EVENT_LBUTTONDOWN:
        cv2.circle(img, (x,y), 100, (0,255,0), thickness=3)
    
       
    
cv2.namedWindow(winname = 'moj_drw')

cv2.setMouseCallback('moj_drw', draw_circle)

###########################
####### Show image ########
###########################
img = cv2.imread('C://Users//Mojgan//Documents//Python/Udemy/OpenCV/Resources/original/Computer-Vision-with-Python/DATA/dog_backpack.jpg')

while True:
    cv2.imshow('moj_drw', img)
    
    if cv2.waitKey(20) & 0xFF == 27:
        break
        
cv2.destroyAllWindows()
