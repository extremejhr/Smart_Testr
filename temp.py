# -*- coding: utf-8 -*-
"""
Created on Thu Jun 28 16:31:42 2018

@author: xqk9qq

"""

from PIL import Image, ImageGrab 
# pytesseract import module should be modified if 'Image has no attribute Image' appears.

import pytesseract
import cv2
import numpy as np
from matplotlib import pyplot as plt
import Levenshtein
pytesseract.pytesseract.tesseract_cmd = 'E:\\Development\\Tesseract-OCR\\tesseract'

import pyautogui
import win32gui # OCR combined with win32 get handler and position.

import time


image = Image.open('icon\\icon77.png')

image1 = np.array(image)

_, image1 = cv2.threshold(image1, 10, 255, cv2.THRESH_BINARY)

image1 = Image.fromarray(image1)

image1 = image1.resize((image1.size[0]*5,image1.size[1]*5),Image.ANTIALIAS)

cv2.imshow('',np.array(image1))

OCR_string = pytesseract.image_to_string(image)
