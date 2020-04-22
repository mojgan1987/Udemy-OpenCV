import numpy as np
from numpy import genfromtxt

data = genfromtxt('../DATA/bank_note_data.txt',delimiter=',')

# separating lable and features
labels = data[:,4]
labels
features = data[:,0:4]
features
X = features
y = labels

# split to train and test
from sklearn.model_selection import train_test_split # does ramdomize shuffling train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

len(X_train)
X_test.max()

from sklearn.preprocessing import MinMaxScaler # force all the feature data to follow certain range
scaler_object = MinMaxScaler()
scaler_object.fit(X_train) #finds the min and max
# NOTE! we fit only to train, not all or test bcz do not have any knowledge about test data

# transform:
scaled_X_train = scaler_object.transform(X_train)
scaled_X_test = scaler_object.transform(X_test)

sacled_X_train.min()

# creating a simple network with keras
from keras.models import Sequential
from keras.layers import Dense

model = Sequential()

model.add(Dense(4,input_dim=4,activation='relu'))

model.add(Dense(8,activation='relu'))

model.add(Dense(1,activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(scaled_X_train, y_train,epochs=50,verbose=2)#verbose:reporting back, reporting a long
model.predict_classes(scaled_X_test)

# evaluate
model.metrics_names
from sklearn.metrics import confusion_matrix,classification_report
predictions = model.predict_classes(scaled_X_test)
confusion_matrix(y_test, predictions)
print(classification_report(y_test, predictions))

model.save('myfirstmodel.h5')
from keras.models import load_model
newmodel = load_model('myfirstmodel.h5')
newmodel.predict_classes(scaled_X_test)
