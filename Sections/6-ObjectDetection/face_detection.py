import cv2
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

nadia = cv2.imread('../data/Nadia_Murad.jpg', 0)
denis = cv2.imread('../data/Denis_Mukwege.jpg', 0)
solvay = cv2.imread('../DATA/solvay_conference.jpg',0)
plt.imshow(solvay, cmap='gray')

face_cascade = cv2.CascadeClassifier('../data/haarcascades/haarcascade_frontalface_default.xml')
# contains 6000 classifiers and if it fits all features, indicates that there should be a face there!

# function to draw a rectangle
def detect_face(img):
    
    face_img = img.copy()
    
    face_rectangle = face_cascade.detectMultiScale(face_img) # x,y + w,h
    
    for (x,y,w,h) in face_rectangle:
        cv2.rectangle(face_img, (x,y), (x+w,y+h), (255,255,255),10)
    
    return face_img

result = detect_face(denis)
plt.imshow(result, cmap='gray')

result = detect_face(nadia)
plt.imshow(result, cmap='gray')

result = detect_face(solvay)
plt.imshow(result, cmap='gray')

# add parameters to detect more precisely
# scale factor and min neighburs
# function to draw a rectangle
def detect_face(img):
    
    face_img = img.copy()
    
    face_rectangle = face_cascade.detectMultiScale(face_img, scaleFactor=1.2, minNeighbors=5) 
    
    for (x,y,w,h) in face_rectangle:
        cv2.rectangle(face_img, (x,y), (x+w,y+h), (255,255,255),10)
    
    return face_img

result = detect_face(solvay)
plt.imshow(result, cmap='gray')

eye_cascade = cv2.CascadeClassifier('../data/haarcascades/haarcascade_eye.xml')

# function to draw a rectangle
def detect_eye(img):
    
    face_img = img.copy()
    
    eye_rectangle = eye_cascade.detectMultiScale(face_img) # x,y + w,h
    
    for (x,y,w,h) in eye_rectangle:
        cv2.rectangle(face_img, (x,y), (x+w,y+h), (255,255,255),10)
    
    return face_img

result = detect_eye(nadia)
plt.imshow(result, cmap='gray')

result = detect_eye(denis)
plt.imshow(result, cmap='gray')

result = detect_eye(solvay)
plt.imshow(result, cmap='gray')

# face detection from video, sp webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read(0)
    
    frame = detect_face(frame)
    
    cv2.imshow('Video Face Detect', frame)
    
    k = cv2.waitKey(1)
    if k ==27:
        break
        
cap.release()
cv2.destroyAllWindows()
