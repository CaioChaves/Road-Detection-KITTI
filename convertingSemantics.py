"""
20/05/2019
This code simply converts the semantics of the dataset KITTI
from the classification in several classes
to a binary one (navigable ground or non-navigable ground)
The be executed only one time, once you download the original dataset 
to adapt it to our problem.
"""

import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
import time


def clear_all():
    """Clears all the variables from the workspace of the spyder application."""
    gl = globals().copy()
    for var in gl:
        if var[0] == '_': continue
        if 'func' in str(globals()[var]): continue
        if 'module' in str(globals()[var]): continue

        del globals()[var]
if __name__ == "__main__":
    clear_all()


clear_all()

os.chdir("/home/caio/workspace/pre/data_semantics/training")

# Create directory
dirName = 'semantics_binary_ground'
try:
    # Create target Directory
    os.mkdir(dirName)
    print("Directory " , dirName ,  " Created ") 
except FileExistsError:
    print("Directory " , dirName ,  " already exists")
    
directory = "/home/caio/workspace/pre/data_semantics/training/semantic_rgb"
os.chdir(directory)

# List of pixels identified as navigable ground
NavigableGroundPixel = np.array([128,64,128])
cont = 1

for filename in os.listdir(directory):
     img = np.asarray(Image.open(filename))
     print("Working on image:",filename," ",cont,"/200")
     numberPixelsRow = np.shape(img)[0]
     numberPixelsColumn = np.shape(img)[1]
     numberPixelsChannel = np.shape(img)[2]
     binaryImage = np.zeros([numberPixelsRow,numberPixelsColumn])
     
     for i in range(numberPixelsRow):
         for j in range(numberPixelsColumn):
             actualPixel = img[i,j,:]
             if all(actualPixel == NavigableGroundPixel):
                 binaryImage[i,j] = 255
             else:
                 binaryImage[i,j] = 0

     cont = cont + 1
     imgBinary = Image.fromarray(np.uint8(binaryImage))
     imgBinary.save("BinaryGround_"+filename,"PNG")
 





    