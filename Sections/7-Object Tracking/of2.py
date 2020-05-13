# tracking movements by color change

import cv2
import numpy as np

cap = cv2.VideoCapture(0)

ret,frame1 = cap.read()
prvsImg = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)

# HSV based mask
hsv_mask = np.zeros_like(frame1)
hsv_mask[:,:,1] = 255 # along saturation channel, fully saturated

while True:
    ret,frame2 = cap.read() # compare prev with current
    nextImg = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    
    # calculate flow
    flow = cv2.calcOpticalFlowFarneback(prvsImg, nextImg,None, 0.5,3,15,3,5,1.2,0)
    
    # convert flo to color coordinate
    mag, ang = cv2.cartToPolar(flow[:,:,0],flow[:,:,1], angleInDegrees=True) # cartesian, 0 horizenta, 1 verical
    
    hsv_mask[:,:,0] = ang/2 # 0 grab the hue
    
    hsv_mask[:,:,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX) # 2 actual value
    
    bgr = cv2.cvtColor(hsv_mask,cv2.COLOR_HSV2BGR)
    cv2.imshow('frame',bgr)
    
    k = cv2.waitKey(10) & 0xFF
    if k== 27:
        break
    
    prvsImg = nextImg
cap.release()
cv2.destoyAllWindows()
