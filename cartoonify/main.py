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
    """
    Allows user to choose file
    @return imgPath: Path to image
    """
    #open file box
    imgPath=easygui.fileopenbox()
    return imgPath


def cartoonify(imgPath):
    """
    Cartoonifies image stored in specified image path
    @param imgPath: path to image file
    """
    #read image 
    img=cv2.imread(imgPath)
    if img is None:
        print("The image file could not be found")
        sys.exit()
    #convert to RGB
    img_cvt_clr=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    #resize image
    img_resized=cv2.resize(img_cvt_clr,(900,900))
    #show image 
    grey_img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img_resized_grey=cv2.resize(grey_img,(900,900))
    #apply median blur
    greyScale_smooth=cv2.medianBlur(grey_img,5)
    resized_smooth=cv2.resize(greyScale_smooth,(900,900))
#Allow user to choose image to cartoonify
imgPath=chooseFile()
cartoonify(imgPath)