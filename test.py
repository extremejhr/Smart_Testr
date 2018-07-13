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

kernel_dilate = np.uint8(np.ones((3,4)))      
kernel_dilate[1,:]=0
segmentation_threshold_group=[190, 160, 180]
kernel_dilate_reg_group=[kernel_dilate]
kernel_dilate_box_group=[kernel_dilate]
scale_group=[[0.05,0.05],[0.1,0.1]]

optimize_index=ST.HyPara_Optimize('NX 1847',segmentation_threshold_group,kernel_dilate_reg_group,kernel_dilate_box_group,scale_group).IGrid_Search()   