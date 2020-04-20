import cv2
import matplotlib.pyplot as plt
%matplotlib inline

# read a sgrayscale 0
img = cv2.imread('C://Users//Mojgan//Documents//Python/Udemy/OpenCV/Resources/original/Computer-Vision-with-Python/DATA/rainbow.jpg',0)
plt.imshow(img, cmap ='gray')

# 255: img.max
ret, thresh1 = cv2.threshold(img, 127, 255, cv2.THRESH_TRUNC) #thresh trunc, zero, etc
ret
plt.imshow(thresh1, cmap ='gray')

# img 2
img = cv2.imread('C://Users//Mojgan//Documents//Python/Udemy/OpenCV/Resources/original/Computer-Vision-with-Python/DATA/crossword.jpg',0)
plt.imshow(img, cmap='gray')

# changing display size
def show_pic(img):
    fig = plt.figure(figsize=(15,15))
    ax = fig.add_subplot(111)
    ax.imshow(img, cmap='gray')
show_pic(img)

# binary
ret, th1 = cv2.threshold(img, 180, 255, cv2.THRESH_BINARY)
show_pic(th1)

# adaptive threshold: giassian and c_means
th2 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY, 11, 8) #11 num of neiughburs odd, constant is subtracted from mean
show_pic(th2)

# blend threshold 1 and 2
blended = cv2.addWeighted(src1=th1, alpha = 0.6, src2=th2, beta=0.4, gamma=0)
show_pic(blended)
