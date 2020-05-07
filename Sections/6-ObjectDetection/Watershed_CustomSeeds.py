import cv2
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

road = cv2.imread('../data/road_image.jpg')
road_copy = np.copy(road)
plt.imshow(road_copy)

# create empty space 
road.shape[:2]

marker_image = np.zeros(road.shape[:2], dtype=np.int32)
marker_image.shape

segments = np.zeros(road.shape, dtype=np.uint8)
segments.shape

# qualitative colormaps
from matplotlib import cm
cm.tab10(0) #RGB, alpha

tuple(np.array(cm.tab10(0)[:3])*255) # distinct unit colors

def creat_rgb(i):
    return tuple(np.array(cm.tab10(i)[:3])*255) # distinct unit colors
    
colors = []
for i in range(10):
    colors.append(creat_rgb(i))

colors


##
# program 
##
# gloabal variables
# color choice
n_marker = 10 #0-9
current_marker = 1

# if markers have been updated by watershed algorithm
marks_updated = False

##
# callback function
def mouse_callback(event, x, y, flags, param):
    global marks_updated
    
    if event == cv2.EVENT_LBUTTONDOWN:
        #draw a circle, markers passed to the watershed algo
        cv2.circle(marker_image, (x,y), 10, (current_marker),-1)
        
        # what user sees 
        cv2.circle(road_copy, (x,y), 10, colors[current_marker], -1)
        
        marks_updated = True


##
# while true
cv2.namedWindow('Road Image')
cv2.setMouseCallback('Road Image', mouse_callback)

while True: 
    
    cv2.imshow('watershed segments', segments)
    cv2.imshow('Road Image', road_copy)
    
    #close all windows
    k = cv2.waitKey(1)
    
    if k == 27:
        break
        
    # clearing all the color on press C    
    elif k == ord('c'):
        road_copy = road.copy()
        marker_image = np.zeros(road.shape[:2], dtype=np.int32)
        segments = np.zeros(road.shape, dtype=np.uint8)

    # update color choice
    elif k>0 and chr(k).isdigit():
        current_marker = int(chr(k))
        
    
    
    # update the markers
    if marks_updated:
        marker_image_copy = marker_image.copy()
        cv2.watershed(road, marker_image_copy)
        
        segments = np.zeros(road.shape, dtype=np.uint8)
        
        for color_ind in range(n_marker):
            # coloring the segments
            segments[marker_image_copy==(color_ind)] = colors[color_ind]
    
cv2.destroyAllWindows()    
