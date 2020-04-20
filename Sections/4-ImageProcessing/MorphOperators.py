import cv2
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

def load_img():
    blank_img = np.zeros((600,600))
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(blank_img,text='ABCDE',org=(50,300),fontFace=font, fontScale=5,color=(255,255,255),thickness=25,lineType=cv2.LINE_AA)
    return blank_img
    
def display_img(img):
    fig = plt.figure(figsize=(12,10))
    ax = fig.add_subplot(111)
    ax.imshow(img,cmap='gray')

img = load_img()
display_img(img)

kernel = np.ones((5,5), dtype = np.uint8)
kernel

result = cv2.erode(img, kernel, iterations=1)
display_img(result)

result = cv2.erode(img, kernel, iterations=4)
display_img(result)

# openning: ero + dil
img = load_img()
# creat white noise
white_noise = np.random.randint(low=0,high=2,size=(600,600))
white_noise
white_noise = white_noise * 255
display_img(white_noise)

noisy_img = white_noise+img
display_img(noisy_img)

# openning to get rid of noise in the BG
openning = cv2.morphologyEx(noisy_img, cv2.MORPH_OPEN, kernel)
display_img(openning)

# removing foreground noise
img = load_img()
black_noise = np.random.randint(low=0, high=2, size=(600,600))
black_noise = black_noise * -255
black_noise_img = img+black_noise
display_img(black_noise_img)
black_noise_img[black_noise_img==-255]=0
black_noise_img.min()
display_img(black_noise_img)

# closing to remove foreground noise
closing = cv2.morphologyEx(black_noise_img, cv2.MORPH_CLOSE, kernel)
display_img(closing)


# morph gradient
img = load_img()
display_img(img)

# edge detection
gradients = cv2.morphologyEx(img, cv2.MORPH_GRADIENT,kernel)
display_img(gradients)

