# interactive drawing
import cv2

## CALLBACK Function to draw rectangle
def draw_rectagle(event, x, y, flags, param):
    global pt1, pt2, topLeft_clicked, botRight_Clicked
    
    if event == cv2.EVENT_LBUTTONDOWN:
        # reset the rectangle
        if topLeft_clicked == True and botRight_Clicked == True:
            pt1 = (0,0)
            pt2 = (0,0)
            topLeft_clicked = False
            botRight_Clicked = False
    
    if topLeft_clicked == False:
        pt1 = (x,y)
        topLeft_clicked = True
        
    elif botRight_Clicked == False:
        pt2 = (x,y)
        botRight_Clicked = True
            

## GLOBAL VARIABLE
pt1 = (0,0)
pt2 = (0,0)
topLeft_clicked = False
botRight_Clicked = False

## connect to callback
cap = cv2.VideoCapture(0)
cv2.namedWindow('Test')
cv2.setMouseCallback('Test', draw_rectagle)

while True:
    ret,frame = cap.read()
    
    # drawing on the frame based on global variable
    if topLeft_clicked == True:
        cv2.circle(frame,center=pt1,radius=5,color=(0,0,255), thickness=-1)
    if topLeft_clicked and botRight_Clicked:
        cv2.rectangle(frame, pt1, pt2, (0,0,255),3)
    
    
    cv2.imshow('Test',frame)
    
    if cv2.waitKey(0) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
