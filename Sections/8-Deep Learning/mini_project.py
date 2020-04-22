# Fashion MNIST dataset
from keras.datasets import fashion_mnist

# load data
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

import matplotlib.pyplot as plt
plt.imshow(x_train[0])

# Normalize
x_train.max()
x_train = x_train/255
x_test = x_test/255

from keras.utils import to_categorical
y_train_cat = to_categorical(y_train, 10)
y_test_cat = to_categorical(y_test, 10)

from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten

# Building the model
model = Sequential()

model.add(Conv2D(filters=32, kernel_size=(4,4), input_shape=(28,28,1), activation='relu'))
model.add(MaxPooling2D((2,2)))
model.add(Flatten())
model.add(Dense(128,activation='relu'))
model.add(Dense(10, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

# review the model
model.summary()

# training the model
model.fit(x_train, y_train_cat, epochs=10)

model.metrics_names
model.evaluate(x_test, y_test_cat)

# test
from sklearn.metrics import classification_report  
predictions = model.predict_classes(x_test)
print(classification_report(y_test, predictions))

