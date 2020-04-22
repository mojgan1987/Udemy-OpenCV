# one hot encoding

from keras.datasets import mnist

(X_train, y_train), (X_test, y_test) = mnist.load_data()

import matplotlib.pyplot as plt
%matplotlib inline

X_train.shape

single_img = X_train[0]
plt.imshow(single_img, cmap='gray_r')

y_train
# one hot encoding
from keras.utils.np_utils import to_categorical

y_cat_test = to_categorical(y_test, 10)
y_cat_train = to_categorical(y_train, 10)

y_cat_train[0]

# processing x
# normalizing data
X_train = X_train/X_train.max()
X_test = X_test/X_test.max()

scaled_image = X_train[0]
scaled_image.max()
plt.imshow(scaled_image, cmap='gray_r')

# one dimension lost
# reshape data
X_train.shape

# include color channel
X_train = X_train.reshape(60000,28,28,1)
X_test = X_test.reshape(10000,28,28,1)
X_test.shape

# modeling
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten

model = Sequential()

# convo
model.add(Conv2D(filters=32, kernel_size=(4,4), input_shape=(28,28,1), activation = 'relu')) # complex images higher filter

# pooling layer
model.add(MaxPool2D(pool_size=(2,2)))

# flatten 2D to 1D output label
model.add(Flatten())

# Dense Layer
model.add(Dense(128, activation='relu'))

model.add(Dense(10, activation='softmax'))

model.compile(loss='categorical_crossentropy',
             optimizer='rmsprop',
             metrics=['accuracy'])
             
model.summary()

model.fit(X_train, y_cat_train, epochs=2)

model.metrics_names

model.evaluate(X_test, y_cat_test)

from sklearn.metrics import classification_report

predictions = model.predict_classes(X_test)

y_cat_test

# understands non one hot coding
print(classification_report(y_test,predictions))
