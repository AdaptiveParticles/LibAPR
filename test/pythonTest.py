# Find path to script - all test files are kept there
import os
myPath = os.path.dirname(os.path.abspath(__file__))
print("Python script directory [" + myPath + "]")

print("DIR: [" + os.getcwd() + "]")
files = [f for f in os.listdir('.') if os.path.isfile(f)]
for f in files:
    print("F: [" + f + "]");

# Check if APR can be read and if gives correct dimensions
import pyApr
import numpy as np
apr=pyApr.AprShort()
apr.read(myPath + '/files/Apr/sphere_120/sphere_apr.h5')
img=np.array(apr, copy = False)
assert img.shape == (120, 120, 120)
