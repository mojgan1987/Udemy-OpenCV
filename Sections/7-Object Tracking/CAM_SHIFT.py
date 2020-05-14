import numpy as np
import cv2

cap = cv2.VideoCapture(0)

ret,frame = cap.read()

# face tracking
# first object detection and then track
face_cascade = cv2.CascadeClassifier('../data/haarcascades/haarcascade_frontalface_default.xml')
face_rects = face_cascade.detectMultiScale(frame) 

# first face
(face_x,face_y,w,h) = tuple(face_rects[0]) 
track_window = (face_x,face_y, w, h)

# track teh rectangle
roi = frame[face_y:face_y+h, face_x:face_x+w]

hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

# 
roi_hist = cv2.calcHist([hsv_roi],[0],None,[180],[0,180]) # 0 one channel

# normalize hist
cv2.normalize(roi_hist,roi_hist,0,255,cv2.NORM_MINMAX)

# seting up termination criteria
term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1) # 10 iteration, 1 epsilon

while True:
    
    ret,frame = cap.read()
    
    
    if ret == True:
        # grab the frame and convert 2 hsv
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # calculate back projection based on roi
        dst = cv2.calcBackProject([hsv],[0], roi_hist,[0,180],1)
        
#         ret , track_window = cv2.meanShift(dst, track_window,term_crit)
        
#         # draw new rect
#         x,y,w,h = track_window
#         img2 = cv2.rectangle(frame, (x,y) , (x+w,y+h), (0,0,255),5)

        ret, track_window = cv2.CamShift(dst, track_window, term_crit)
        
        pts = cv2.boxPoints(ret)
        pts = np.int0(pts)
        
        img2 = cv2.polylines(frame, [pts], True, (0,0,255),5)
        
        cv2.imshow('img', img2)
        
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break
    else:
        break # if not connected to camera
        
cv2.destroyAllWindows()
cap.release()
