# -*- coding: utf-8 -*-
"""
Created on Wed Jul 11 14:57:59 2018

@author: xqk9qq
"""
###############################################################################  

###############################################################################
#
# Import Library
#
###############################################################################

from PIL import Image, ImageGrab 
import pytesseract
import cv2
import numpy as np
import win32gui
import Levenshtein

import Smart_Testr_Library as ST

pytesseract.pytesseract.tesseract_cmd = 'E:\\Development\\Tesseract-OCR\\tesseract'

###############################################################################  

img, window_position = ST.Image_Capture('NX 1847').ICapture()    