# connect openCV to Webcam

import cv2

cap = cv2.VideoCapture(0)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
# framerate: CAP_PROP_FRAME_COUNT

# MAC or Linux *'XVID'
#running on WIndows : *'DVIX'
writer = cv2.VideoWriter('myvideoOCV.mp4',cv2.VideoWriter_fourcc(*'DVIX'), 20,(width,height)) 

while True:    
    ret,frame = cap.read()
    
#     OPERATIONS (drawing)
    writer.write(frame)

    cv2.imshow('frame',frame) # full color
    
# grayscale
#     gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
#     cv2.imshow('frame',gray)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
        
cap.release()
writer.release()
cv2.destroyAllWindows()        
