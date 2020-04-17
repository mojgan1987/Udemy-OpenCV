import numpy as np
myarr=np.ones((5,5))
myarr*10

# This line sets a "seed" so you get the same random numbers we do
np.random.seed(101)
# This line creates an array of random numbers
arr = np.random.randint(low=0,high=100,size=(5,5))


mycopy = pic_arr.copy()

# RED TO ZERO
mycopy[:,:,0] = 0
# Green TO ZERO
mycopy[:,:,1] = 0

plt.imshow(mycopy)
