# -*- coding: utf-8 -*-
"""
Created on Wed Jun 27 12:29:34 2018

@author: xqk9qq
"""

###############################################################################
#                                                                             
# Class Library                                                                           
#                                                                             
###############################################################################

###############################################################################
#
# Import Modules
#
###############################################################################
from PIL import Image 
# pytesseract import module should be modified if 'Image has no attribute Image' appears.

import pytesseract
import cv2
import numpy as np
from matplotlib import pyplot as plt
import Levenshtein
pytesseract.pytesseract.tesseract_cmd = 'E:\\Development\\Tesseract-OCR\\tesseract'

import pyautogui
import win32gui # OCR combined with win32 get handler and position.

###############################################################################

###############################################################################
#
# Image Segmentation -> Search Regions (22.06.2018)
#
###############################################################################

class Image_Segmentation(object):
    
    def __init__(self,Operation_Image):
        
        self.Operation_Image = Operation_Image
        
    def Get_Region(self, plot_flag = None, Morph_kernel = None):
        
        if plot_flag is None:
            
            plot_flag = False
            
        if Morph_kernel is None:
            
            Morph_kernel = (5,5)     
                   
        self.img_initial = cv2.imread(self.Operation_Image)
        
        h, w, _ = self.img_initial.shape
        
        gray = cv2.cvtColor(self.img_initial, cv2.COLOR_BGR2GRAY)
        
        # Delete the Long Edge
        
        (_, thresh) = cv2.threshold(gray, 140, 255, cv2.THRESH_BINARY_INV)
        
        edges = cv2.Canny(thresh,100,200)
        
        lines = cv2.HoughLinesP(edges,1,np.pi/180,10, minLineLength = 20, maxLineGap = 0)
        
        for i in range(len(lines)):
            line = lines[i]    
            cv2.line(thresh,(line[0][0],line[0][1]), (line[0][2],line[0][3]),(0,0,0),5)
            
        # Segmentation
        # -> Can be optimized further in order to find more accurate parameter of morphology.
        # -> For iteration, stop at the maximum box numbers.
            
        thresh_copy = thresh.copy()
        
        kernel_morph = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, Morph_kernel)
        
        closed = cv2.morphologyEx(thresh_copy, cv2.MORPH_CLOSE, kernel_morph) 
        
        kernel_dilate = np.uint8(np.ones((6,7)))
        kernel_dilate[3,:]=0
        
        closed = cv2.dilate(closed, kernel_dilate, 5)
        
        #blurred = cv2.GaussianBlur(closed, (9, 9),0) # Gaussian Blur reduce noise
        blurred = closed
        
        (_, cnts, _) = cv2.findContours(blurred, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        c = sorted(cnts, key=cv2.contourArea, reverse=True)
        
        # compute the rotated bounding box of the largest contour
        
        thresh_cover = cv2.bitwise_not(thresh)
        
        Search_Region_Coordinates = np.int0(np.zeros((1,4)))
        
        
        for i in range(len(c)):
            
            rect = cv2.minAreaRect(c[i])
            
            box = np.int0(cv2.boxPoints(rect))
            
            Xs = [i[0] for i in box]
            Ys = [i[1] for i in box]
            
            x1 = (min(Xs) + abs(min(Xs)))/2
            x2 = max(Xs) if max(Xs) <= w else w
            y1 = (min(Ys) + abs(min(Ys)))/2
            y2 = max(Ys) if max(Ys) <= h else h
            
            box = np.int0([[x1,y1],[x2,y1],[x2,y2],[x1,y2]])
            corner = np.int0([[x1,y1,x2,y2]])
            
            if i == 0:
                
                Search_Region_Coordinates[i][:4] = corner
            
            else:
                
                if abs(x1-x2) >= 2 and abs(y1-y2) >= 2 :
                
                    Search_Region_Coordinates = np.append(Search_Region_Coordinates, corner,axis = 0)
                    
                    draw_img = cv2.drawContours(self.img_initial, [box], -1, (0, 0, 255), 3)
                    
        
        if plot_flag is True:            
        #cv2.imshow("blurred",blurred)      
            cv2.imshow("draw_img", self.img_initial)
            cv2.waitKey()            
            
        return Search_Region_Coordinates

###############################################################################
        
import pyautogui
import win32gui # OCR combined with win32 get handler and position.

a = Image_Segmentation('Image\\Search.png')

b=a.Get_Region()

label = 'NX 1847.1400 / Analysis_nx13.226' 

hld = win32gui.FindWindow(None, label)

#Tab_handle = win32gui.FindWindowEx(win32gui.FindWindow('#32770','???????V7.35'),None,'SysTabControl32','Tab1')

left, top, right, bottom = win32gui.GetWindowRect(hld)

Target_Keyword = 'Open'

for i in range(len(b)):
    
    if b[i][0]>=544 or b[i][1]>=163:
        
        continue
    
    else:
    
        x1 = b[i][0]
        y1 = b[i][1]
        x2 = b[i][2]
        y2 = b[i][3]
        
        hight = y2-y1
        width = x2-x1
        
        
        crop_img= thresh_cover[y1:y1+hight, x1:x1+width]
        
        OCR_string = pytesseract.image_to_string(crop_img)
        
        print(OCR_string)
        
        if Levenshtein.ratio(OCR_string, Target_Keyword)>0.8:
            #print(OCR_string)
            #print(i)
            #cv2.imwrite('icon\\icon'+str(i)+'.png',thresh_cover[y1:y1+hight, x1:x1+width])
            pyautogui.moveTo(left+x1+abs(x2-x1)/2, top+y1+abs(y2-y1)/2)
            pyautogui.click()  
            break
    
    
        
a = Image_Segmentation('Image\\Search.png')

b=a.Get_Region()