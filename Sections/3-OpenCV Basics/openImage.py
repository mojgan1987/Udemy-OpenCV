import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

img = cv2.imread('C://Users//Mojgan//Documents//Python/Udemy/OpenCV/Resources/original/Computer-Vision-with-Python/DATA/00-puppy.jpg')
type(img) #if NoneType: wrong file

plt.imshow(img)

fixed_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(fixed_image)

img_gray = cv2.imread('C://Users//Mojgan//Documents//Python/Udemy/OpenCV/Resources/original/Computer-Vision-with-Python/DATA/00-puppy.jpg', cv2.IMREAD_GRAYSCALE)
plt.imshow(img_gray, cmap='gray')

img_gray.shape
fixed_image.shape

new_image = cv2.resize(fixed_image, (1000, 400))
plt.imshow(new_image)

w_ration = 0.5
h_ratio = 0.5
n_img = cv2.resize(fixed_image, (0,0), fixed_image, w_ration, h_ratio)
plt.imshow(n_img)
n_img.shape

new_imga = cv2.flip(fixed_image, -1)
plt.imshow(new_imga)

cv2.imwrite('totaly_new.jpg', fixed_image)

fig = plt.figure(figsize=(2,8))
ax = fig.add_subplot(111)
ax.imshow(fixed_image)
