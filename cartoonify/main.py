#the purpose of this script is to cartoonify an image 

#imports 
import cv2
import numpy as np 
import matplotlib.pyplot as plt 
import imageio
import sys
import easygui

"""
CV2: Imported to use OpenCV for image processing
easygui: Imported to open a file box. It allows us to select any file from our system.
Numpy: Images are stored and processed as numbers. These are taken as arrays. We use NumPy to deal with arrays.
Imageio: Used to read the file which is chosen by file box using a path.
Matplotlib: This library is used for visualization and plotting. Thus, it is imported to form the plot of images.
OS: For OS interaction. Here, to read the path and save images to that path.

"""

def chooseFile():
    imgPath=opengui.fileopenbox()
