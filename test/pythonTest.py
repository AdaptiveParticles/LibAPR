# Find path to script - all test files are kept there
import os
myPath = os.path.dirname(os.path.abspath(__file__))

# Add path to build APR pythond module
import sys
sys.path.insert(0, os.getcwd())

# Check if APR can be read and if gives correct dimensions
import pyApr
import numpy as np
apr=pyApr.AprShort()

apr.read(myPath + '/files/Apr/sphere_120/sphere_apr.h5')
img=np.array(apr.reconstruct(), copy = False)
print(img.shape)

img=np.arange(1000, dtype = np.uint16).reshape(10,10,10)

if(apr.readArr(img, False)):
    print('array successfully read into c++')
else:
    print('Error: array could not be read into c++')


img=np.array(apr, copy = False)
print(img.shape)
# plot a 2D slice of the image

