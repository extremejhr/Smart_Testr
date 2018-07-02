# -*- coding: utf-8 -*-
"""
Created on Thu Jun 28 15:59:03 2018

@author: xqk9qq
"""

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

###############################################################################

###############################################################################
#
# Window_Capture -> Image Capture (28.06.2018)
#
###############################################################################    
    
class Window_Capture(object):
    
    def __init__(self,title):
        
        self.title = title
        
        self.handle = None
        
    def Handle_Catch(self,hwnd,lParam):
        
        if win32gui.IsWindowVisible(hwnd):
            
            if self.title in win32gui.GetWindowText(hwnd):
                
                self.handle = hwnd     
   
    def Image_Capture(self):
        
        win32gui.EnumWindows(self.Handle_Catch, None)
        
        if self.handle is None:
            
            print('No Window Found!')
            
        else:
            
            bbox = win32gui.GetWindowRect(self.handle)
            self.img = np.array(ImageGrab.grab(bbox))
        
        return self.img, self.handle
    
###############################################################################  

###############################################################################
#
# Image Segmentation -> Search Regions (22.06.2018)
#
###############################################################################

class Image_Segmentation(object):
    
    def __init__(self,Operation_Image):
        
        self.Operation_Image = Operation_Image
        
    def Get_Region(self, plot_flag = None, Morph_kernel = None, Binary_Threshold = None):
        
        if plot_flag is None:
            
            plot_flag = False
            
        if Morph_kernel is None:
            
            Morph_kernel = (2,2)  
            
        if Binary_Threshold is None:
            
            Binary_Threshold = 180               
                   
        #self.img_initial = cv2.imread(self.Operation_Image)
        
        self.img_initial = self.Operation_Image
        
        h, w, _ = self.img_initial.shape
        
        gray = cv2.cvtColor(self.img_initial, cv2.COLOR_BGR2GRAY)
        
        # Delete the Long Edge
        
        (_, thresh) = cv2.threshold(gray, Binary_Threshold, 255, cv2.THRESH_BINARY_INV)     
        
        edges = cv2.Canny(thresh,100,200)
        
        lines = cv2.HoughLinesP(edges,1,np.pi/180,10, minLineLength = 20, maxLineGap = 0)
        
        if lines is not None:

            for i in range(len(lines)):
                line = lines[i]    
                cv2.line(thresh,(line[0][0],line[0][1]), (line[0][2],line[0][3]),(0,0,0),5)
            
        # Segmentation
        # -> Can be optimized further in order to find more accurate parameter of morphology.
        # -> For iteration, stop at the maximum box numbers.
            
        thresh_copy = thresh.copy()
        
        kernel_morph = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, Morph_kernel)
        
        closed = cv2.morphologyEx(thresh_copy, cv2.MORPH_CLOSE, kernel_morph) 
        
        kernel_dilate = np.uint8(np.ones((3,6)))
        kernel_dilate[1,:]=0
        
        closed = cv2.dilate(closed, kernel_dilate, 1)
        
        #blurred = cv2.GaussianBlur(closed, (9, 9),0) # Gaussian Blur reduce noise
        blurred = closed
        
        (_, cnts, _) = cv2.findContours(blurred, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        c = sorted(cnts, key=cv2.contourArea, reverse=True)
        
        # compute the rotated bounding box of the largest contour
        
        thresh_cover = cv2.bitwise_not(thresh)
        
        Search_Region_Coordinates = np.int0(np.zeros((1,4)))
        
        #cv2.imshow('ss',blurred)
        #cv2.waitKey()
        
        for i in range(len(c)):
            
            rect = cv2.minAreaRect(c[i])
            
            box = np.int0(cv2.boxPoints(rect))
            
            Xs = [i[0] for i in box]
            Ys = [i[1] for i in box]            
            
            x1 = (min(Xs) + abs(min(Xs)))/2
            x2 = max(Xs) if max(Xs) <= w else w
            y1 = (min(Ys) + abs(min(Ys)))/2
            y2 = max(Ys) if max(Ys) <= h else h
            
            h1 = y2 - y1
            w1 = x2 - x1
            
            ratio = abs(h1/w1) if abs(h1/w1)>=1 else abs(w1/h1)
            
            scale = 0.1
            
            box = np.int0([[x1,y1],[x2,y1],[x2,y2],[x1,y2]])
            
            if (x1 - w1*scale) >= 0 and (y1 - h1*scale) >= 0:
            
                corner = np.int0([[(x1),(y1),(x2 + w1*scale),(y2 + h1*scale)]])
            
            else:
                
                corner = np.int0([[x1,y1,(x2 + w1*scale),(y2 + h1*scale)]])
            
            if i == 0:
                
                Search_Region_Coordinates[i][:4] = corner
            
            else:
                
                if abs(h1*w1) < 5000 and abs(h1*w1) > 10 and ratio <= 20 :
                
                    Search_Region_Coordinates = np.append(Search_Region_Coordinates, corner,axis = 0)
                    
                    #draw_img = cv2.drawContours(self.img_initial, [box], -1, (0, 0, 255), 1)
                    
        if plot_flag is True:            
            #cv2.imshow("blurred",blurred)      
            cv2.imshow("draw_img", self.img_initial)
            cv2.waitKey()            
            
        return Search_Region_Coordinates, thresh_cover
    
    def Group_Region (self, plot_flag = None):
        
        if plot_flag is None:
            
            plot_flag = False
        
        SRC, thresh_cover = self.Get_Region()
        
        SRC = SRC.tolist()
        
        Search_index = list(range(len(SRC)))
                
        Group_Region_Coordinates = np.int0(np.zeros((1,4)))
        
        while len(Search_index) > 0 :
            
            j = Search_index[0]
            
            Group_index = [j]
                          
            for k in Search_index[1:]:
                
                Expand_scale = 0.5
                
                l1 = [int(SRC[j][0]*(1-Expand_scale)),int(SRC[j][1]*(1-Expand_scale))]
                r1 = [int(SRC[j][2]*(1+Expand_scale)),int(SRC[j][3]*(1+Expand_scale))]
                l2 = [int(SRC[k][0]*(1-Expand_scale)),int(SRC[k][1]*(1-Expand_scale))]
                r2 = [int(SRC[k][2]*(1+Expand_scale)),int(SRC[k][3]*(1+Expand_scale))]
                               
                if (l1[0]>r2[0] or l2[0]>r1[0] or l1[1]<r2[1] or l2[1]<l1[1]) == False :
            
                    Group_index.append(k)      
                    
            for k in Group_index:                      
                
                del Search_index[Search_index.index(k)]                                      
            
            if len(Group_index) > 1:
                
                x1_range = [[],]
                x2_range = [[],]
                y1_range = [[],]
                y2_range = [[],]                
                                
                for k in Group_index:
                    
                    x1_range.append(SRC[k][0])
                    y1_range.append(SRC[k][1]) 
                    x2_range.append(SRC[k][2])
                    y2_range.append(SRC[k][3])
                
                x1 = min(x1_range[1:])
                y1 = min(y1_range[1:])
                x2 = max(x2_range[1:])
                y2 = max(y2_range[1:])
                
                
            else:
                
                x1, y1, x2, y2 = SRC[j]
                
            box = np.int0([[x1,y1],[x2,y1],[x2,y2],[x1,y2]])     
            
            draw_img = cv2.drawContours(self.img_initial, [box], -1, (0, 0, 255), 1)
            
            if j == 0:
                
                Group_Region_Coordinates[0][:4] = np.int0([[x1,y1,x2,y2]])
                
            else:
            
                Group_Region_Coordinates = np.append(Group_Region_Coordinates, np.int0([[x1,y1,x2,y2]]),axis = 0)               
            
            Group_index = []
            
        if plot_flag is True:            
            #cv2.imshow("blurred",blurred)      
            cv2.imshow("draw_img", self.img_initial)
            cv2.waitKey()            
            
        return Group_Region_Coordinates, thresh_cover
                              
###############################################################################
        
        
###############################################################################
#
# OCR matching -> Coordinates Location (28.06.2018)
#
###############################################################################     

class Operation_Location(object):
    
    def __init__(self, Search_Region,Image,hwnd,Target_Keyword):
        
        self.SR = Search_Region
        self.Image = Image
        self.hwnd = hwnd
        self.Target_Keyword = Target_Keyword
        
    def Get_Location(self) :
        
        left , top , _ , _ = win32gui.GetWindowRect(self.hwnd)
        
        for i in range(len(self.SR)):
            
            x1 = self.SR[i][0]
            y1 = self.SR[i][1]
            x2 = self.SR[i][2]
            y2 = self.SR[i][3]
                
            hight = y2-y1
            width = x2-x1
                
                
            crop_img = self.Image[y1:y1+hight, x1:x1+width]
            

            crop_img1 = cv2.bitwise_not(crop_img[int(0.05*hight):int(0.95*hight),int(0.1*width):int(0.9*width)])
            
            #cv2.imshow('',crop_img1)
            #cv2.waitKey()
                
            OCR_string = pytesseract.image_to_string(Image.fromarray(crop_img))
            
            
            
            if len(OCR_string) == 0 or OCR_string.isspace():
                print('void')
                #OCR_string = pytesseract.image_to_string(crop_img1)
                cv2.imwrite('icon1\\icon'+str(i)+'.png', crop_img)
                
            else:
                
                print(OCR_string)
                cv2.imwrite('icon\\icon'+str(i)+'.png', crop_img)
                
            if Levenshtein.ratio(OCR_string, self.Target_Keyword)>0.6:
                pyautogui.moveTo(left+x1+abs(x2-x1)/2, top+y1+abs(y2-y1)/2)
                pyautogui.click()  
                break        
            
###############################################################################     


















lable = ['Close']

wintext = ['NX 1847']



for i in range(len(lable)) :
    
    window = Window_Capture(wintext[i])
    
    img_initial,hwnd = window.Image_Capture()
    
    img_process = Image_Segmentation(img_initial)
    
    regions, img_processed = img_process.Group_Region(plot_flag=True)
    
    e = Operation_Location(regions,img_processed ,hwnd,lable[i])
    
    e.Get_Location()
    
    time.sleep(1)

    