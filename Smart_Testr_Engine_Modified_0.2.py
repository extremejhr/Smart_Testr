# -*- coding: utf-8 -*-
"""
Created on Thu Jul 12 14:29:23 2018

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
        
    def img_process(self, Morph_kernel = None, Binary_Threshold = None):        
            
        if Morph_kernel is None:
            
            Morph_kernel = (2,2)  
            
        if Binary_Threshold is None:
            
            Binary_Threshold = 180               
                   
        #self.img_initial = cv2.imread(self.Operation_Image)
        
        self.img_initial = self.Operation_Image
        
        gray = cv2.cvtColor(self.img_initial, cv2.COLOR_BGR2GRAY)
        
        # Delete the Long Edge
        
        
        (_, self.thresh) = cv2.threshold(gray, Binary_Threshold, 255, cv2.THRESH_BINARY_INV) 
        
        #(_, self.thresh_recog) = cv2.threshold(gray,180, 255, cv2.THRESH_BINARY_INV)
        
        #self.thresh =cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,11,2)
               
        self.thresh_recog =cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV,11,2)
        
        edges = cv2.Canny(self.thresh,100,200)
        
        edges_1 = cv2.Canny(self.thresh_recog,100,200)
        
        lines = cv2.HoughLinesP(edges,1,np.pi/180,10, minLineLength = 20, maxLineGap = 0)
        
        lines_1 = cv2.HoughLinesP(edges_1,1,np.pi/180,10, minLineLength = 20, maxLineGap = 0)
        
        if lines is not None:
            for i in range(len(lines)):
                line = lines[i]    
                cv2.line(self.thresh,(line[0][0],line[0][1]), (line[0][2],line[0][3]),(0,0,0),5)
       
        if lines_1 is not None:
            for i in range(len(lines_1)):
                line = lines_1[i]    
                cv2.line(self.thresh_recog,(line[0][0],line[0][1]), (line[0][2],line[0][3]),(0,0,0),4)
                
        thresh_copy = self.thresh.copy()
          
        kernel_morph = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, Morph_kernel)
        
        self.closed = cv2.morphologyEx(thresh_copy, cv2.MORPH_CLOSE, kernel_morph) 
        
        #self.thresh_cover = cv2.bitwise_not(self.thresh)
    
    def img_group(self, thresh_region, kernel_dilate):              
        
        # Segmentation
        # -> Can be optimized further in order to find more accurate parameter of morphology.
        # -> For iteration, stop at the maximum box numbers      
        img_initial_copy = self.img_initial.copy()
        
        h, w, _ = img_initial_copy.shape
        
        closed = cv2.dilate(thresh_region, kernel_dilate,1)
        
        #blurred = cv2.GaussianBlur(closed, (9, 9),0) # Gaussian Blur reduce noise
        blurred = closed
        
        (_, cnts, _) = cv2.findContours(blurred, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        c = sorted(cnts, key=cv2.contourArea, reverse=True)
        
        # compute the rotated bounding box of the largest contour
        
        Region_Coordinates = np.int0(np.zeros((1,4)))
        
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
            
            scaley = 0.2
            scalex = 0
            
            box = np.int0([[x1,y1],[x2,y1],[x2,y2],[x1,y2]])
            
            if (x1 - w1*scalex) >= 0 and (y1 - h1*scaley) >= 0:
            
                corner = np.int0([[(x1- w1*scalex),(y1- h1*scaley),(x2 + w1*scalex),(y2 + h1*scaley)]])
                
                x1 = x1- w1*scalex
                
                y1 = y1- h1*scaley
                
                x2 = x2 + w1*scalex
                
                y2 = y2 + h1*scaley
            
            else:
                
                corner = np.int0([[x1,y1,(x2 + w1*scalex),(y2 + h1*scaley)]])
                
            box1 =  np.int0([[x1,y1],[x2,y1],[x2,y2],[x1,y2]])  
            
            if i == 0:
                
                Region_Coordinates[i][:4] = corner
            
            else:
                
                if abs(h1*w1) > 10 and abs(h1*w1)<abs(h*w)/15:
                
                    Region_Coordinates = np.append(Region_Coordinates, corner,axis = 0)
                    
                    cv2.drawContours(thresh_region, [box], -1, (255, 255, 255), -1)
                        
                    cv2.drawContours(img_initial_copy, [box1], -1, (0, 0, 255), 1)
                                  
        return Region_Coordinates, thresh_region, img_initial_copy
         
    
    def Get_Region(self, plot_flag = None):
        
        self.img_process()
        
        thresh_copy  = self.closed.copy()
        
        thresh_cover = cv2.bitwise_not(self.thresh_recog.copy())

#******************************************************************************        
        kernel_dilate = np.uint8(np.ones((3,4)))      
        kernel_dilate[1,:]=0
#******************************************************************************
           
        Small_box, Small_region, plot_img_small = self.img_group(thresh_copy,kernel_dilate)
                    
        kernel_dilate = np.uint8(np.ones((4,4)))      
        kernel_dilate[2,:]=0
        
        Big_box, Big_region, plot_img_big = self.img_group(Small_region,kernel_dilate)                   
                                    
        if plot_flag is True:            
            
            cv2.imshow("small", plot_img_small)      
            cv2.imshow("big", plot_img_big)
            cv2.waitKey()            
   
        return Big_box, Small_box, thresh_cover
    
###############################################################################
         
        
###############################################################################
#
# OCR matching -> Coordinates Location (28.06.2018)
#
###############################################################################     

class Operation_Location(object):
    
    def __init__(self, Search_Region_L,Search_Region_S,Image,hwnd,Target_Keyword):
        
        self.SRL = Search_Region_L
        self.SRM = Search_Region_S
        self.Image = Image
        self.hwnd = hwnd
        self.Target_Keyword = Target_Keyword
        self.region = region
        
    def Get_Location(self) :
        
        Target_Keyword = self.Target_Keyword.lower()
        
        left , top , _ , _ = win32gui.GetWindowRect(self.hwnd)
        
        SRL = self.SRL
        
        SRM = self.SRM
        
        h, w= self.Image.shape  
        
        break_flag = 0
        
        for i in range(len(SRL)): 
            
            if break_flag == 1: 
                
                break
            
            k_tag = list(np.zeros((len(Target_Keyword.split()),1)))
            
            Medical_results = list(np.zeros((len(SRM),1)))
            
            positive_index =[]            
                      
            x1 = SRL[i][0]
            y1 = SRL[i][1]
            x2 = SRL[i][2]
            y2 = SRL[i][3]  
            
            hight = y2-y1
            width = x2-x1            
            
            
            crop_img1 = self.Image[y1:y1+hight, x1:x1+width] 
            
            #cv2.imshow('',crop_img1)
            #cv2.waitKey()
            
            for j in range(len(SRM)):   
                
                if SRM[j][0]>= x1 and SRM[j][1]>= y1 and SRM[j][2]<= x2 and SRM[j][3]<= y2 :
                    
                
                    x1m = SRM[j][0]
                    y1m = SRM[j][1]
                    x2m = SRM[j][2]
                    y2m = SRM[j][3]
                        
                    hightm = y2m-y1m
                    widthm = x2m-x1m             
                        
                    crop_img = self.Image[y1m:y1m+hightm, x1m:x1m+widthm]  
                    
                    cv2.imwrite('icon\\icon'+str(j)+'.png', crop_img)
                    
                    OCR_string = pytesseract.image_to_string(Image.fromarray(crop_img),lang='eng').lower()
                    
                    OCR_string_porcessed = ''.join(e for e in OCR_string if e.isalnum() or e.isspace())
                    
                    Target_Keyword_i = ''.join(e for e in Target_Keyword if e.isalnum() or e.isspace())
                    
                    Target_Keyword_i = Target_Keyword.split()  
                    
                    if len(OCR_string_porcessed.split()) > 0:
                   
                        print(OCR_string_porcessed)                                 
            
                    for k in range(len(Target_Keyword_i)):
                        
                        if Levenshtein.ratio(OCR_string_porcessed, Target_Keyword_i[k])>0.9:               
                            
                            Medical_results[j] = k+1
                            
                            print(Medical_results[j])
                   
                    for k in range(len(Target_Keyword_i)):  
                        
                        if len([i for i, e in enumerate(Medical_results) if e == k+1])>0 :
                            
                            k_tag[k] = 1
                    
                    
                    if sum(k_tag) == len(Target_Keyword_i):
                        
                        break_flag = 1  
                        
                        pyautogui.moveTo(left+x1+abs(x2m-x1m)/2, top+y1m+abs(y2m-y1m)/2)
                        
                        pyautogui.click() 
                        
                        
                        break
                
############################################################################### 

lable = ['Element Quality', 'Selected']

wintext = ['NX 1847','Element Quality']



for i in range(len(lable)) :
    
    window = Window_Capture(wintext[i])
    
    img_initial,hwnd = window.Image_Capture()
    
    img_process = Image_Segmentation(img_initial)
    
    big_regions,small_regions, img_processed = img_process.Get_Region(plot_flag=True)
    
    e = Operation_Location(big_regions,small_regions,img_processed ,hwnd,lable[i])
    
    e.Get_Location()
    
    time.sleep(2)

    