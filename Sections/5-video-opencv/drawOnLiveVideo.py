# draw on video
import cv2

cap = cv2.VideoCapture(0)

# PAY ATTENTION to int conversion
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# draw rectangle on the live video
# calculate top left points of the rectangle
x = width // 2 # integer
y = height // 2

# width and height of rectangle
w = width//4
h = height//4

# bottom right x+h, y+h

while True:
    ret, frame = cap.read()
    
    cv2.rectangle(frame, (x, y), (x+w, y+h), color=(0,0,255),thickness= 4)
    
    cv2.imshow('frame',frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
        
cap.release()
cv2.destroyAllWindows()
    
