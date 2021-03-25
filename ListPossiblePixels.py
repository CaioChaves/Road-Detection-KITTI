"""
20/05/2019
This code analyses the 200 images from the semantic division of the KITTI
dataset and makes a list of all color RGB that appear at least once in an image.
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

directory = os.chdir(os.getcwd()+'/data_semantics/training/semantic_rgb')

listofPixels = []
isEqual = np.array([])
cont = 0

for filename in os.listdir(directory):
    if filename.endswith(".png"):
        # Beginning of loop for every file .png in the directory
        # img = mpimg.imread(filename)
        
        img = np.asarray(Image.open(filename))
        print("Working on image:",filename)
        numberPixelsRow = np.shape(img)[0]
        numberPixelsColumn = np.shape(img)[1]
        numberPixelsChannel = np.shape(img)[2]
        
        if numberPixelsChannel != 3:
            print("ERROR! Input image does not have 3 channels (RGB) as expected!")
            break
        
               
        for i in range(numberPixelsRow):
            for j in range(numberPixelsColumn):
                isEqual = np.array([])
                actualPixel = img[i,j,:]
                         
                if np.shape(listofPixels)[0] == 0:  #Empty list
                    listofPixels.insert(np.shape(listofPixels)[0],actualPixel)
                    cont = cont+1
                    print("New pixel: ",actualPixel,"detected at: (",i,",",j,")")
                    continue    
                                    
                for pxl in listofPixels:
                     if all( (pxl == actualPixel) == True):
                         isEqual = np.concatenate((isEqual,[1]),axis=0)
                     else:
                         isEqual = np.concatenate((isEqual,[0]),axis=0)
                         
        
                if all(isEqual == 0):
                    cont = cont + 1
                    listofPixels.insert(np.shape(listofPixels)[0],actualPixel)
                    print("New pixel: ",actualPixel,"detected at: (",i,",",j,")")
                    
                
                
    
        
        
        ## at the end im = Image.fromarray(np.uint8(img))
        
        # mpimg.imsave()
        # End of the loop for every file .png in the directory
        continue
    else:
        continue
    
    

