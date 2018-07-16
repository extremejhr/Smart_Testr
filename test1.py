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
import os
import time

import Smart_Testr_Library as ST

#pytesseract.pytesseract.tesseract_cmd = 'E:\\Machine Learning\\Tesseract-OCR\\tesseract'

############################################################################### 

#time.sleep(5)

def main():
    
    segmentation_threshold_group=160
        
    kernel_dilate_reg1 = np.uint8(np.ones((2,3)))      
    kernel_dilate_reg1[0,:]=0  
    kernel_dilate_reg2 = np.uint8(np.ones((3,4)))      
    kernel_dilate_reg2[1,:]=0  
    kernel_dilate_reg3 = np.uint8(np.ones((4,5)))      
    kernel_dilate_reg3[2,:]=0  
    
    kernel_dilate_reg_group=kernel_dilate_reg2
    
    kernel_dilate_box1 = np.uint8(np.ones((5,5)))      
    kernel_dilate_box1[2,:] = 0
    
    kernel_dilate_box_group=kernel_dilate_box1
    
    scale_group=[0,0.25]
    
    #optimize_index = ST.HyPara_Optimize('Simcenter 12',segmentation_threshold_group,kernel_dilate_reg_group,kernel_dilate_box_group,scale_group).IGrid_Search()
    
    ST.Search_Engine('Spyder',['MB1','Manage materials'],segmentation_threshold_group,kernel_dilate_reg_group,kernel_dilate_box_group,scale_group).IOperate()  
    
if __name__ == '__main__':
    
    main()