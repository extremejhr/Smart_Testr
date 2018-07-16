# -*- coding: utf-8 -*-
"""
Created on Fri Jul 13 12:31:16 2018

@author: xqk9qq
"""
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

import os

import shutil

import time

from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, Executor


###############################################################################
#
# Image Capture - Only suitable for Win32 application
#
###############################################################################
   
class Image_Capture(object):
    
    def __init__(self,title):
        
        self.title = title
        
        self.handle = None
        
    def Handle_Catch(self,hwnd,lParam):
        
        if win32gui.IsWindowVisible(hwnd):
            
            if self.title in win32gui.GetWindowText(hwnd):
                
                self.handle = hwnd     
   
    def ICapture(self):
        
        win32gui.EnumWindows(self.Handle_Catch, None)
        
        if self.handle is None:
            
            print('No Window Found!')
            
        else:
            
            bbox = win32gui.GetWindowRect(self.handle)
            img = np.array(ImageGrab.grab(bbox))
            
        window_position = win32gui.GetWindowRect(self.handle)    
        
        return img, window_position
    
###############################################################################  

###############################################################################
#
# Image Process - Binarization and Hough Transform
#
###############################################################################    
    
    
class Image_Process(object):
    
    def __init__(self,Image_Initial, Segmentation_Threshold):
        
        self.img_initial = Image_Initial
        
        self.segmentation_threshold = Segmentation_Threshold
        
    def IProcess(self):
        
        segmentation_threshold = self.segmentation_threshold
        
        img_initial = self.img_initial.copy()
        
        gray = cv2.cvtColor(img_initial, cv2.COLOR_BGR2GRAY)
              
        (_, thresh_seg) = cv2.threshold(gray, segmentation_threshold, 255, cv2.THRESH_BINARY_INV) 
               
        thresh_ocr =cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV,3,2)
             
        edges = cv2.Canny(thresh_seg,100,200)
        
        edges_1 = cv2.Canny(thresh_ocr,100,200)
        
        lines = cv2.HoughLinesP(edges,1,np.pi/180,10, minLineLength = 20, maxLineGap = 0)
        
        lines_1 = cv2.HoughLinesP(edges_1,1,np.pi/180,10, minLineLength = 20, maxLineGap = 0)
        
        if lines is not None:
            
            for i in range(len(lines)):
                
                line = lines[i]    
                cv2.line(thresh_seg,(line[0][0],line[0][1]), (line[0][2],line[0][3]),(0,0,0),4)
       
        if lines_1 is not None:
            
            for i in range(len(lines_1)):
                
                line = lines_1[i]    
                cv2.line(thresh_ocr,(line[0][0],line[0][1]), (line[0][2],line[0][3]),(0,0,0),4)
                
        kernel_morph = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2,2))
        
        thresh_seg = cv2.morphologyEx(thresh_seg, cv2.MORPH_CLOSE, kernel_morph)     
        
        thresh_ocr = cv2.bitwise_not(thresh_ocr)
        
        #cv2.imshow('',thresh_ocr)
        #cv2.waitKey()
                
        return thresh_seg, thresh_ocr
    
###############################################################################  

###############################################################################
#
# Image Analyze - Seperate text regions and calculate the boundary
#
###############################################################################    
    
class Image_Analyze(object): 
    
    def __init__(self,Thresh_Seg, Kernel_Dilate_Region, Kernel_Dilate_Box, Scale):
        
        self.thresh_seg = Thresh_Seg
        
        self.kernel_dilate_reg = Kernel_Dilate_Region
        
        self.kernel_dilate_box = Kernel_Dilate_Box
        
        self.scalex = Scale[0] 

        self.scaley = Scale[1]         

    def IRegion(self,thresh_region,kernel_dilate):              
        
        scalex = self.scalex

        scaley = self.scaley      
        
        h, w = thresh_region.shape
        
        closed = cv2.dilate(thresh_region, kernel_dilate,1)
        
        (_, cnts, _) = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        c = sorted(cnts, key=cv2.contourArea, reverse=True)
        
        region_coordinates = np.int0(np.zeros((1,4)))
        
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
            
            box_b =  np.int0([[x1,y1],[x2,y1],[x2,y2],[x1,y2]]) 
            
            if (x1 - w1*scalex) >= 0 and (y1 - h1*scaley) >= 0:
                
                x1 = x1- w1*scalex
                    
                y1 = y1- h1*scaley

            x2 = x2 + w1*scalex
                
            y2 = y2 + h1*scaley    
                   
            corner = np.int0([[x1,y1,x2,y2]])
                
            box =  np.int0([[x1,y1],[x2,y1],[x2,y2],[x1,y2]])  
            
            if i == 0:
                
                region_coordinates[i][:4] = corner
            
            else:
                
                if abs(h1*w1) > 50 and abs(h1*w1)<abs(h*w)/15:
                
                    region_coordinates = np.append(region_coordinates, corner,axis = 0)
                    
                    cv2.drawContours(thresh_region, [box_b], -1, (255, 255, 255), -1)
                    
                    #cv2.drawContours(thresh_seg,[box],-1, (255, 255, 255), -1)
      
        #cv2.imshow('',thresh_region)
        #cv2.waitKey()
        return region_coordinates, thresh_region    
    
    def IBoundary(self):
        
        kernel_dilate_reg = self.kernel_dilate_reg
        
        kernel_dilate_box = self.kernel_dilate_box
        
        thresh_seg = self.thresh_seg 
        
        region_coordinates, thresh_region = self.IRegion(thresh_seg,kernel_dilate_reg)
        
        box_coordinates, _ = self.IRegion(thresh_region, kernel_dilate_box) 
        
        group_coordinates = []

        for i in range(len(box_coordinates)): 
                      
            x1 = box_coordinates[i][0]
            y1 = box_coordinates[i][1]
            x2 = box_coordinates[i][2]
            y2 = box_coordinates[i][3]  

            group_pack = []                                    

            for j in range(len(region_coordinates)):   
                 
                if region_coordinates[j][0]>= x1 and region_coordinates[j][1]>= y1 and region_coordinates[j][2]<= x2 and region_coordinates[j][3]<= y2 :

                    x1m = region_coordinates[j][0]
                    y1m = region_coordinates[j][1]
                    x2m = region_coordinates[j][2]
                    y2m = region_coordinates[j][3]
                    
                    group_pack.append([x1m,y1m,x2m,y2m])
            
            group_coordinates.append(group_pack) 
   
        return group_coordinates
    
###############################################################################  

###############################################################################
#
# Image Segmentation - OCR image and return the Keyword position
#
###############################################################################    

class Image_OCR(object):
    
    def __init__(self,Thresh_OCR, Window_Position ,Group_Coordinates, Target_Keyword):
        
        self.group_coordinates = Group_Coordinates
        
        self.thresh_ocr = Thresh_OCR
        
        self.window_position = Window_Position
        
        self.target_keyword = Target_Keyword
        
    def IOCR(self, Group_Coordinates) :
        
        group_coordinates = Group_Coordinates
        
        thresh_ocr = self.thresh_ocr
        
        positive_value = 0
        
        medical_matrix = []
        
        for i in range(len(group_coordinates)): 

            for j in range(len(group_coordinates[i])):  
                
                medical_results =[]
                 
                x1m = group_coordinates[i][j][0]
                y1m = group_coordinates[i][j][1]
                x2m = group_coordinates[i][j][2]
                y2m = group_coordinates[i][j][3]
                    
                hightm = y2m-y1m
                widthm = x2m-x1m             
                    
                crop_img = thresh_ocr[y1m:y1m+hightm, x1m:x1m+widthm] 
                
                OCR_string = pytesseract.image_to_string(Image.fromarray(crop_img),lang='eng').lower()
                
                OCR_string_porcessed = ''.join(e for e in OCR_string if e.isalnum() or e.isspace())
                
                # Grid Search Criteria
                
                if len(OCR_string_porcessed.split()) > 0:
                    
                    positive_value = positive_value + 1  
                    
                    medical_results.append([OCR_string_porcessed,x1m,y1m,x2m,y2m])
                    
                    #cv2.imwrite('icon\\'+OCR_string_porcessed+'.png', crop_img) 
                
                else:
                   
                    #cv2.imwrite('icon1\\'+str(j)+'.png', crop_img)
                    pass
            
            medical_matrix.append(medical_results)
                       
        return medical_matrix
   
    def ISearch(self, Multi_Threads = None):
        
        multi_threads = Multi_Threads 
        
        if multi_threads is None:
            
            multi_threads = 8      
            
        step = int(len(self.group_coordinates)/multi_threads)
        
        group_coordinates = [] 
        
        for i in range(multi_threads):
            
            if i < multi_threads-1:
            
                group_coordinates.append(self.group_coordinates[step*i:step*(i+1)])
            
            else:
                
                group_coordinates.append(self.group_coordinates[step*i:])   
    
        start = time.time()
        
        pool = ProcessPoolExecutor(max_workers=multi_threads)
        
        medical_matrix = list(pool.map(self.IOCR, group_coordinates))
        
        print(medical_matrix)
        
        end = time.time()
        
        print('OCR Collapse Time = ',end-start)
    
        target_keyword = self.target_keyword.lower()
        
        Target_Keyword_i = target_keyword.split()
        
        window_position = self.window_position
        
        thresh_ocr = self.thresh_ocr
        
        left = window_position[0]
        
        top = window_position[1]
        
        h, w= thresh_ocr.shape  
        
        break_flag = 0
        
        operation_coordinates=[]
        
        medical_matrix, positive_value =  self.IOCR()
        
        #shutil.rmtree('icon')
        
        #os.mkdir('icon')
        
        for i in range(len(medical_matrix)): 
            
            x1m = medical_matrix[i][j][0]
            y1m = medical_matrix[i][j][1]
            
            x2m = medical_matrix[i][j][2]
            y2m = medical_matrix[i][j][3]
            
            if break_flag == 1: 
                
                break

            k_tag = list(np.zeros((len(target_keyword.split()),1)))
            
            medical_region_results = list(np.zeros((len(medical_matrix[i]),1)))                                   
                
            for j in range(len(medical_matrix[i])):
                
                for k in range(len(Target_Keyword_i)):
                
                    if Levenshtein.ratio(medical_matrix[i][0], Target_Keyword_i[k])>0.8:               
                        
                        medical_region_results[j] = k+1
                    
           
            for m in range(len(Target_Keyword_i)):  
                
                if len([i for i, e in enumerate(medical_region_results) if e == m+1])>0 :
                    
                    k_tag[m] = 1
            
            
            if sum(k_tag) == len(Target_Keyword_i):
                
                break_flag = 1  
                
                operation_coordinates = [left+x1+abs(x2m-x1m)/2, top+y1m+abs(y2m-y1m)/2]                      
                                        
                break    
                    
        return operation_coordinates
##############################################################################  


###############################################################################
#
# Mouse&Keyboard Action Event
#
###############################################################################  

class MK_Event(object):
    
    def __init__(self, Action_Key, Operation_Coordinates):
        
        self.operation_coordinates = Operation_Coordinates
        
        self.action_key = Action_Key
        
    def IOperate(self):
        
        action_key = self.action_key
        
        operation_coordinates = self.operation_coordinates
        
        x=operation_coordinates[0]
        
        y=operation_coordinates[1]
        
        if action_key == 'MB1':
                        
             pyautogui.click(x, y)
            
        elif action_key == 'MB3':
            
             pyautogui.rightClick(x, y)
             
        elif action_key == 'KIn': 
            
             pyautogui.Click(x+50, y)
             
             pyautogui.typewrite(action_key[1])
        
        else:
            
            print('Wrong Action Key!')
             
###############################################################################  

###############################################################################
#
# Image Segmentation - Wrapper of classes (capture, process, ocr)
#
###############################################################################   

class Search_Engine(object):
    
        def __init__(self,Title,Action_Sequence,Segmentation_Threshold,Kernel_Dilate_Region,Kernel_Dilate_Box,Scale):
        
            self.title = Title
            
            self.action_key = Action_Sequence[0]
            
            self.target_keyword = Action_Sequence[1] 
            
            self.segmentation_threshold = Segmentation_Threshold
            
            self.kernel_dilate_reg = Kernel_Dilate_Region
        
            self.kernel_dilate_box = Kernel_Dilate_Box
            
            self.scale = Scale
                  
        def ILocate(self):            
            
            title = self.title
            
            target_keyword = self.target_keyword
            
            segmentation_threshold = self.segmentation_threshold
            
            kernel_dilate_reg = self.kernel_dilate_reg 
            
            kernel_dilate_reg = self.kernel_dilate_reg
            
            kernel_dilate_box = self.kernel_dilate_box
            
            scale = self.scale          
            
            img, window_position = Image_Capture(title).ICapture()
    
            thresh_seg, thresh_ocr = Image_Process(img, segmentation_threshold).IProcess()
            
            group_coordinates = Image_Analyze(thresh_seg, kernel_dilate_reg, kernel_dilate_box, scale).IBoundary()
            
            operation_coordinates = Image_OCR(thresh_ocr, window_position ,group_coordinates,target_keyword).ISearch()
        
            return operation_coordinates, positive_value
        
        def IOperate(self):
            
            action_key = self.action_key
            
            operation_coordinates, _ = self.ILocate()
           
            MK_Event(action_key, operation_coordinates).IOperate()
                    

###############################################################################  

###############################################################################
#
# Hyperparameters Optimization - Using Grid Search algorithm
#
###############################################################################

class HyPara_Optimize(object): 

    def __init__(self, Title, Segmentation_Threshold_Group, Kernel_Dilate_Region_Group , Kernel_Dilate_Box_Group , Scale_Group ):
        
        self.title = Title
        
        self.segmentation_threshold_group = Segmentation_Threshold_Group
        
        self.kernel_dilate_reg_group = Kernel_Dilate_Region_Group
    
        self.kernel_dilate_box_group = Kernel_Dilate_Box_Group
        
        self.scale_group = Scale_Group        
        
    def IGrid_Search(self):
        
        title = self.title
        
        segmentation_threshold_group = self.segmentation_threshold_group
        
        kernel_dilate_reg_group = self.kernel_dilate_reg_group
        
        kernel_dilate_box_group = self.kernel_dilate_box_group
        
        scale_group = self.scale_group  
        
        target_keyword = 'GRIDSearchWordTemp'
        
        p1_group = list(range(len(segmentation_threshold_group)))
        p2_group = list(range(len(kernel_dilate_reg_group)))
        p3_group = list(range(len(kernel_dilate_box_group)))
        p4_group = list(range(len(scale_group)))
        
        positive_value_p = 0
        
        optimize_index = []
        
        grid_index = [(i,j,m,n) for i in p1_group for j in p2_group for m in p3_group for n in p4_group]
        
        
        for i in range(len(grid_index)):
            
            print('Process = ', i, '/',len(grid_index))
            
            _ , positive_value = Search_Engine(title,target_keyword, segmentation_threshold_group[grid_index[i][0]]\
                                           ,kernel_dilate_reg_group[grid_index[i][1]], kernel_dilate_box_group[grid_index[i][2]]\
                                           ,scale_group[grid_index[i][3]]).ILocate()
            
            if positive_value_p < positive_value:
            
                positive_value_p = positive_value
                
                optimize_index = grid_index[i]
                
                print('!!!!OCR Success Num = ', positive_value)
                
        
        return optimize_index
            
###############################################################################  

###############################################################################
#
# Input file format
#
###############################################################################  
'''
Not finished    

'''           
###############################################################################  