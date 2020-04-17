import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

from PIL import Image
pic = Image.open('C:/Users/Mojgan/Documents/Python/Udemy/OpenCV/Resources/original/Computer-Vision-with-Python/DATA/00-puppy.jpg')
pic

type(pic)

pic_arr = np.asarray(pic)
pic_arr.shape

plt.imshow(pic_arr)

pic_red = pic_arr.copy()
# R G B
plt.imshow(pic_red[:,:,0])
# red channel values 0-255
plt.imshow(pic_red[:,:,0], cmap = 'gray')
plt.imshow(pic_red[:,:,1], cmap = 'gray')

# GREEN CHANNEL
pic_red[:,:,1] = 0
plt.imshow(pic_red[:,:,2])

pic_red[:,:,1].shape
