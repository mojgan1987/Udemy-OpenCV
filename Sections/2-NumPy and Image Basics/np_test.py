import numpy as np
mylist = [1,2,3]
type(mylist)
# list

myarray = np.array(mylist)
type(myarray)
# numpy.ndarray

np.arange(1,10)
np.zeros(shape=(10,5))
np.ones((2,5))

np.random.seed(101)
arr = np.random.randint(0,100,10)
arr

arr.max()
arr.argmax()

arr.shape
arr.reshape((2,5))

mat= np.arange(0,100).reshape(10,10)
mat
row = 0
col = 1
mat[row, col]
mat[:,1]

mat[:,1].reshape(1,10)
mat[0:3,0:3]



