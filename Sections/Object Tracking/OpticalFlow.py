import numpy as np
import cv2

# first technique : sparse

# shi tomasi to detect corners
corner_track_param = dict(maxCorners=10, qualityLevel = 0.3, minDistance=7, blockSize=7)
# we detect 10 corners


# lucas canade
lk_params = dict(winSize=(200,200), maxLevel=2, criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,10,0.03)) 
# smaller window: more sensitive to noise and miss larger motions
# max level: pyramid for image processing: 0:original image
# two critaria: max iterations 10, epsilon = 0.03 more iteration more exhastive, smaller eps finish earlier: speed and accuracy

# grab image from cam
cap = cv2.VideoCapture(0)
ret, prev_frame = cap.read()

prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

# what points we want to track
# choose top 10
prev_points = cv2.goodFeaturesToTrack(prev_gray,mask=None, **corner_track_param)

# display points and drw line
mask = np.zeros_like(prev_frame) # same shape

while True:
    # current frame
    ret, frame = cap.read()
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # cal optical flow
    next_points , status, err = cv2.calcOpticalFlowPyrLK(prev_gray, frame_gray, prev_points, None, **lk_params)
    
    # status: if the flow found, status = 1
    good_new = next_points[status==1]
    good_prv = prev_points[status==1]
    
    for i, (new,prv) in enumerate(zip(good_new,good_prv)):
        
        # cal x y to drwa
        # flatten
        x_new, y_new = new.ravel()
        x_prv, y_prv = prv.ravel()
        
        mask = cv2.line(mask, (x_new, y_new), (x_prv,y_prv), (0,255,0),3)
        
        frame = cv2.circle(frame, (x_new, y_new), 8, (0,0,255),-1)

    img = cv2.add(frame, mask)
    cv2.imshow('tracking', img)
    
    k = cv2.waitKey(30) & 0xFF
    if k == 27:
        break
    
    prev_gray = frame_gray.copy()
    prev_points = good_new.reshape(-1,1,2)
    
cv2.destroyAllWindows()
cap.release()
