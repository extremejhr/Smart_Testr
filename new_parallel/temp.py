# -*- coding: utf-8 -*-
"""
Created on Wed Jul 25 12:12:56 2018

@author: xqk9qq
"""


from PIL import Image, ImageGrab 

# Cancel the image max size limit
#Image.MAX_IMAGE_PIXELS = None

import pytesseract
import cv2
import numpy as np
import win32gui
import Levenshtein
import os
import time

import pythoncom
import PyHook3

import Smart_Testr_Library_parallel as ST

'''
for i in range(2):

    img1,_ = ST.Image_Capture('Simcenter').ICapture()
    
    cv2.imwrite('img'+str(i)+'.png',img1)
    
    time.sleep(5)
'''
img1 = cv2.imread('img0.png')

img2 = cv2.imread('img1.png')

diff = cv2.absdiff(img1,img2)


import cv2
import numpy as np

img1 = cv2.imread("img0.png")
img2 = cv2.imread("img0.png")
diff = cv2.absdiff(img1, img2)
mask = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
th = 1
imask =  mask>th

canvas = np.zeros_like(img2, np.uint8)
canvas[imask] = img2[imask]

cv2.imwrite("result.png", canvas)

cv2.imshow('',img)