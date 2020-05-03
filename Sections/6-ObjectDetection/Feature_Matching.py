import cv2
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

def display(img, cmap='gray'):
    fig = plt.figure(figsize=(12,10))
    ax = fig.add_subplot(111)
    ax.imshow(img,cmap='gray')
    

reeses = cv2.imread('../DATA/reeses_puffs.png',0)
display(reeses)

cereals = cv2.imread('../DATA/many_cereals.jpg',0)
display(cereals)

# ORDB
orb = cv2.ORB_create() # detector object

kp1, des1 = orb.detectAndCompute(reeses,None)
kp2, des2 = orb.detectAndCompute(cereals,None)

# creating matching object
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

matches = bf.match(des1,des2)
single_match = matches[0]
single_match.distance

# sort in order of distance
matches = sorted(matches, key=lambda x:x.distance)

len(matches)

reeses_matches = cv2.drawMatches(reeses,kp1,cereals,kp2,matches[:25], None, flags=2)
display(reeses_matches)


# not a very good job, we move to a more sophisticated method: SIFT: available in older version of OpenCV
sift = cv2.xfeatures2d.SIFT_create()
