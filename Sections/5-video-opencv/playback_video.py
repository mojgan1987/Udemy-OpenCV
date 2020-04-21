import cv2
import time

cap = cv2.VideoCapture('myvideoOCV.mp4')

if cap.isOpened()==False:
    print('File not found! Or wrong CODEC used!') 

while cap.isOpened():
    ret,frame = cap.read()
    
    if ret == True:
        # control playback speed: same frame rate recorded
        time.sleep(1/20) # we recorded with 20 fps
        cv2.imshow('frame',frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
        else:
            break # when finished 

            
cap.release()
cv2.destroyAllWindows()
