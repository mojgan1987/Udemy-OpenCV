import matplotlib.pyplot as plt
import cv2
%matplotlib inline

cat4 = cv2.imread('../DATA/CATS_DOGS/train/CAT/4.jpg')
cat4 = cv2.cvtColor(cat4, cv2.COLOR_BGR2RGB)
plt.imshow(cat4)

cat4.shape

dog2  = cv2.imread('../DATA/CATS_DOGS/train/DOG/2.jpg')
dog2 = cv2.cvtColor(dog2, cv2.COLOR_BGR2RGB)
plt.imshow(dog2)

dog2.shape

# generate flow of batches, preprocessing package
from keras.preprocessing.image import ImageDataGenerator

# random fluctuation of generated image to have more robust by seeing more data
image_gen = ImageDataGenerator(rotation_range=30,
                              width_shift_range=0.1,
                              height_shift_range=0.1,
                              rescale=1/255,
                              shear_range=0.2,
                              zoom_range=0.2,
                              horizontal_flip=True,
                              fill_mode='nearest')

plt.imshow(image_gen .random_transform(dog2))

image_gen.flow_from_directory('../DATA/CATS_DOGS/train') 

from keras.models import Sequential
from keras.layers import Activation, Dropout, Dense, Flatten, Conv2D, MaxPooling2D

input_shape = (150,150,3)
model = Sequential()

model.add(Conv2D(filters=32, kernel_size=(3,3), input_shape=(150,150,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(filters=64, kernel_size=(3,3), input_shape=(150,150,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(filters=64, kernel_size=(3,3), input_shape=(150,150,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())

model.add(Dense(128))
model.add(Activation('relu'))

model.add(Dropout(0.5)) # neurons randomly turn off to avoid over fitting

model.add(Dense(1)) # one neuron for binary class
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
             optimizer='adam',
             metrics=['accuracy'])
             
model.summary()

# training
# choose a batch size: if huge: to long time takes
batch_size = 16

training_image_gen = image_gen.flow_from_directory('../DATA/CATS_DOGS/train',
                                                  target_size=(150,150),
                                                  batch_size=batch_size,
                                                  class_mode='binary') #input_shape[:2]

test_image_gen = image_gen.flow_from_directory('../DATA/CATS_DOGS/test',
                                                  target_size=input_shape[:2],
                                                  batch_size=batch_size,
                                                  class_mode='binary') 

training_image_gen.class_indices

results = model.fit_generator(training_image_gen, epochs=1, steps_per_epoch=150,
                             validation_data=test_image_genst_image_gen, validation_steps=12)
                             
# to ignore warnings:
import warnings 
warnings.filterwarnings('ignore')

# evaluate teh model
results.history['acc'] # accuracy

# load an already learned model
from keras.models import load_model

new_model = load_model('cat_dog_100epochs.h5')
# predicting unseen image
dog_file = '../DATA/CATS_DOGS/test/DOG/10006.jpg'

from keras.preprocessing import image

dog_image = image.load_img(dog_file, target_size=(150,150))

dog_image = image.img_to_array(dog_image)

dog_image.shape

import numpy as np
dog_image = np.expand_dims(dog_image, axis=0)
dog_image.shape

dog_image = dog_image/255

new_model.predict_classes(dog_image)
new_model.predict(dog_image)

# compare to model with 1 ipoch
