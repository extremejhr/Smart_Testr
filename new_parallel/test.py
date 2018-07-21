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

# Cancel the image max size limit
#Image.MAX_IMAGE_PIXELS = None

import pytesseract
import cv2
import numpy as np
import win32gui
import Levenshtein
import os
import time

import Smart_Testr_Library_parallel as ST

#pytesseract.pytesseract.tesseract_cmd = 'E:\\Machine Learning\\Tesseract-OCR\\tesseract'

############################################################################### 

#time.sleep(5)

if __name__ == '__main__':
    
    segmentation_threshold_group=160

    kernel_dilate_reg2 = np.uint8(np.ones((3,4)))      
    kernel_dilate_reg2[1,:]=0  
    
    kernel_dilate_reg_group=kernel_dilate_reg2  
    
    scale_group=[0.1,0.25]
    
    Action_Sequence =[['MB1', 'Element Quality'], ['MB1', 'Selected'], ['MB1', 'Displayed']]
    
    title = ['NX 1847.1500','Element Quality','Element Quality']
    
   # title.append('2D Mapped Mesh')
    
    #ST.Tesseract_Training('Physical Property',segmentation_threshold_group,kernel_dilate_reg_group,scale_group).ISample()  

    #optimize_index = ST.HyPara_Optimize('Simcenter 12',segmentation_threshold_group,kernel_dilate_reg_group,kernel_dilate_box_group,scale_group).IGrid_Search()

    for i in range(len(title)):
    
        ST.Search_Engine(title[i],Action_Sequence[i],segmentation_threshold_group,kernel_dilate_reg_group,scale_group).IOperate()  
        
        time.sleep(1)
