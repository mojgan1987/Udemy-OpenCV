import cv2
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

# first we check the manual approach without using watershed
def display(img, cmap='gray'):
    fig = plt.figure(figsize=(12,10))
    ax = fig.add_subplot(111)
    ax.imshow(img,cmap='gray')
    
sep_coins = cv2.imread('../data/pennies.jpg')
display(sep_coins)

# segmenting image: coins and background
# apply median blur
sep_blur = cv2.medianBlur(sep_coins, 25)
display(sep_blur)

# grayscale
gray_sep_coins =cv2.cvtColor(sep_blur, cv2.COLOR_BGR2GRAY)
display(gray_sep_coins)

# binary threshold 
ret, sep_thresh = cv2.threshold(gray_sep_coins, 160,255,cv2.THRESH_BINARY_INV)
display(sep_thresh)
# find contours
imgae,contour = cv2.findContours(sep_thresh.copy(), cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)


# read image
img = cv2.imread('../data/pennies.jpg')

# median blur
img = cv2.medianBlur(img, 35) # huge imag 35
# display(img)

# convert to gray
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(gray,127,255,cv2.THRESH_BINARY_INV)
display(thresh)

# to get distinct circles: works good with watershed
ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
display(thresh)

# or we can apply a noise removal
kernel = np.ones((3,3),np.uint8)
openning = cv2.morphologyEx(thresh, cv2.MORPH_OPEN,kernel,iterations=2)
display(openning)

# we need 6 seeds
# check which one is in the background and which in the foreground: distance transformation
dist_transform = cv2.distanceTransform(openning, cv2.DIST_L2,5)
display(dist_transform)

ret, sure_fg = cv2.threshold(dist_transform,0.7*dist_transform.max(),255,0)
display(sure_fg) # these are for sure in the foreground

# sure background area
sure_bg = cv2.dilate(openning,kernel,iterations=3)
display(sure_bg)

# now we have both BG and FG, the between in unkown region
sure_fg = np.uint8(sure_fg)
unkown = cv2.subtract(sure_bg,sure_fg)
display(unkown) # we are not sure if these are in the NG or FG

# to make lablel markers and consider them as seeds to have segments
# getting marekers
ret, markers = cv2.connectedComponents(sure_fg)
markers = markers+1 # not to get confused with black BG
markers[unkown==255] = 0
display(markers)

markers = cv2.watershed(img, markers)
display(markers)

# at the ed we can apply contours to draw red line around 
