# edge: sudden change in brightness
# corner: significant change in all direction 
# Shi Tomasi Methos

import cv2
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

flach_chess = cv2.imread('../DATA/flat_chessboard.png')
flach_chess = cv2.cvtColor(flach_chess, cv2.COLOR_BGR2RGB)
plt.imshow(flach_chess)

gray_flat_chess = cv2.cvtColor(flach_chess, cv2.COLOR_BGR2GRAY)
plt.imshow(gray_flat_chess, cmap='gray')

real_chess = cv2.imread('../DATA/real_chessboard.jpg')
real_chess = cv2.cvtColor(real_chess, cv2.COLOR_BGR2RGB)
plt.imshow(real_chess)

gray_real_chess = cv2.cvtColor(real_chess, cv2.COLOR_BGR2GRAY)
plt.imshow(gray_real_chess, cmap='gray')

gray_flat_chess
gray = np.float32(gray_flat_chess)
gray

# corner detection
dst = cv2.cornerHarris(src=gray, blockSize=2,ksize=3,k=0.04)
dst = cv2.dilate(dst,None)

flach_chess[dst>0.01*dst.max()] = [255,0,0] #if more than 1 percent, corner is detected
plt.imshow(flach_chess)

# now in the real chess pic
gray = np.float32(gray_real_chess)
dst = cv2.cornerHarris(src=gray, blockSize=2, ksize=3, k=0.04)
dst = cv2.dilate(dst, None)

real_chess[dst>0.01*dst.max()] = [255,0,0]
plt.imshow(real_chess)


# Method 2
# reload images
flach_chess = cv2.imread('../DATA/flat_chessboard.png')
flach_chess = cv2.cvtColor(flach_chess, cv2.COLOR_BGR2RGB)
gray_flat_chess = cv2.cvtColor(flach_chess, cv2.COLOR_BGR2GRAY)

real_chess = cv2.imread('../DATA/real_chessboard.jpg')
real_chess = cv2.cvtColor(real_chess, cv2.COLOR_BGR2RGB)
gray_real_chess = cv2.cvtColor(real_chess, cv2.COLOR_BGR2GRAY)

corners = cv2.goodFeaturesToTrack(gray_flat_chess,55,0.01,10) # -1 all corners
corners=np.int0(corners)
corners

for i in corners:
    x,y = i.ravel() # flatten the array
    cv2.circle(flach_chess, (x,y),3,(255,0,0), -1)

plt.imshow(flach_chess)


# On the real image
corners = cv2.goodFeaturesToTrack(gray_real_chess,100,0.01,10) # -1 all corners
corners=np.int0(corners)

for i in corners:
    x,y = i.ravel() # flatten the array
    cv2.circle(real_chess, (x,y),3,(255,0,0), -1)
plt.imshow(real_chess)    

