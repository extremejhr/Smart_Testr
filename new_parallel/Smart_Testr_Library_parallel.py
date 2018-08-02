# -*- coding: utf-8 -*-
"""
Created on Fri Jul 13 12:31:16 2018

@author: xqk9qq
"""
###############################################################################
# Import Modules
# pytesseract import module should be modified if 'Image has no attribute Image' appears.
# pytesseract.pytesseract.tesseract_cmd = 'E:\\Development\\Tesseract-OCR\\tesseract'
# pywinauto.application has been modified : add new funcition to achieve the control identifiers.
###############################################################################
from PIL import Image, ImageGrab 
import pytesseract
import cv2
import numpy as np
#from matplotlib import pyplot as plt
import Levenshtein
import pyautogui
import pywinauto
import win32gui 
import pythoncom
import PyHook3
import os
import shutil
import time
import concurrent.futures
import ctypes

import copy

import itertools
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
            
            print('No Main Window Found!')
            
        else:
            
            bbox = win32gui.GetWindowRect(self.handle)
        
            img = np.array(ImageGrab.grab(bbox))
            
        window_position = win32gui.GetWindowRect(self.handle) 
        
        cv2.imwrite('running_temp_file\\'+self.title+'img_previous.png',img)
        
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
               
        thresh_ocr =cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV,3,1)
             
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
    
    def __init__(self,Thresh_Seg, Kernel_Dilate_Region, Scale):
        
        self.thresh_seg = Thresh_Seg
        
        self.kernel_dilate_reg = Kernel_Dilate_Region
        
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
    
    def Box_Eater(self, Region_Coordinates):
        
        region_coordinates = Region_Coordinates
        
        region_coordinates_temp = region_coordinates
        
        break_flag=0
        
        for i in range(len(region_coordinates_temp)):
            
            if break_flag == 1:
                
                break
            
            for j in range(len(region_coordinates_temp)):
                
                x10 = region_coordinates_temp[i][0]
                y10 = region_coordinates_temp[i][1]
                x20 = region_coordinates_temp[i][2]
                y20 = region_coordinates_temp[i][3]  
                
                h1 = y20 - y10
                w1 = x20 - x10
                
                x11 = region_coordinates_temp[j][0]
                y11 = region_coordinates_temp[j][1]
                x21 = region_coordinates_temp[j][2]
                y21 = region_coordinates_temp[j][3] 

                h2 = y21 - y11
                w2 = x21 - x11   
                          
                    
                if i!=j and (abs(x20-x11) <= (w1+w2)) and (abs(x21 - x10) <= (w1+w2)) and (abs(y21-y10) <= (h1+h2)) and (abs(y20-y11) <= (h1+h2)):

                    x1 = min(x10,x11)
                    y1 = min(y10,y11)
                    x2 = max(x20,x21)
                    y2 = max(y20,y21)
                    
                    region_coordinates = np.delete(region_coordinates,[i,j],axis = 0)               
                    region_coordinates = np.append(region_coordinates,np.int0([[x1,y1,x2,y2]]),axis=0) 
                    
                    break_flag = 1
                    
                    break
                         
        return region_coordinates
    
    def getKey(self,item):
        
        return len(item)
    
    def IBoundary(self,thresh_ocr):
        
        kernel_dilate_reg = self.kernel_dilate_reg
        
        thresh_seg = self.thresh_seg 
        
        region_coordinates, thresh_region  = self.IRegion(thresh_seg,kernel_dilate_reg)
        
        region_coordinates_prev = region_coordinates
     
        box_diff = 1
        
        while box_diff > 0 :
                 
            region_coordinates_new = self.Box_Eater(region_coordinates_prev)
            
            #print(len(region_coordinates_prev),len(region_coordinates_new))
            
            box_diff = abs(len(region_coordinates_prev)-len(region_coordinates_new))
            
            region_coordinates_prev = region_coordinates_new
            
         
        for i in range(len(region_coordinates)):
                
                x1 = region_coordinates[i][0]
                y1 = region_coordinates[i][1]
                x2 = region_coordinates[i][2]
                y2 = region_coordinates[i][3]
                
                box_b =  np.int0([[x1,y1],[x2,y1],[x2,y2],[x1,y2]]) 
                
                cv2.drawContours(thresh_ocr, [box_b], -1, (0, 0, 0), 1)    
                
        cv2.imshow('',thresh_ocr)
        cv2.waitKey()
        
        box_coordinates = region_coordinates_prev
        
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
             
            if len(group_pack) > 0 :
            
                group_coordinates.append(group_pack)
                
                group_coordinates = sorted(group_coordinates,key=self.getKey, reverse = True)
   
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
        
    def IOCR(self, Region_Coordinates,training_flag = None,save_path = None) :
        
        region_coordinates = Region_Coordinates
        
        thresh_ocr = self.thresh_ocr

        x1m = region_coordinates[0]
        y1m = region_coordinates[1]
        x2m = region_coordinates[2]
        y2m = region_coordinates[3]
            
        hightm = y2m-y1m
        widthm = x2m-x1m             
            
        crop_img = thresh_ocr[y1m:y1m+hightm, x1m:x1m+widthm] 
        
        OCR_string = pytesseract.image_to_string(Image.fromarray(crop_img),lang='eng',config='--psm 6')
        
        OCR_string_processed = ''.join(e for e in OCR_string if e.isalnum() or e.isspace())
        
        if len(OCR_string_processed)>0 and training_flag is True:
        
            cv2.imwrite('Sample\\'+save_path+'\\'+OCR_string_processed+'.png',crop_img)
                       
        return OCR_string_processed, region_coordinates
   
    def ISearch(self, Multi_Threads = None):
        
        multi_threads = Multi_Threads 
        
        if multi_threads is None:
            
            multi_threads = 8      
        
        group_coordinates = self.group_coordinates 
        
        target_keyword = self.target_keyword     
                           
        target_keyword_i = ''.join(e for e in target_keyword if e.isalnum() or e.isspace())
      
        target_keyword_i = target_keyword.split()
        
        window_position = self.window_position
        
        left = window_position[0]
        
        top = window_position[1]
    
        start = time.time()
        
        for i in range(len(group_coordinates)):
            
            operation_coordinates = []
            
            medical_results_h = [[]]*len(target_keyword_i)
            
            medical_results_l = [[]]*len(target_keyword_i)
            
            Region_Coordinates = group_coordinates[i]
            
            OCR_string_processed = []
            
            region_coordinates = []
            
            futures = []
 
            with concurrent.futures.ThreadPoolExecutor(max_workers=multi_threads) as pool:
            
                for region_coordinates_i in Region_Coordinates:
                
                    future = pool.submit(self.IOCR, region_coordinates_i)
                    
                    futures.append(future)
                
                for future in futures:
                    
                    ocr_string = list(future.result())
                    
                    if len(ocr_string[0])>0:
                        
                        ocr_string[0] = ''.join(e for e in ocr_string[0] if e.isalnum())
                        
                        OCR_string_processed.append(ocr_string[0])
                    
                        region_coordinates.append(ocr_string[1])
            
            if len(OCR_string_processed) > 0:
                
                print(OCR_string_processed)
        
                for j in range(len(target_keyword_i)):
                    
                    medical_results_h_k = []
                    medical_results_l_k = []

                    for k in range(len(OCR_string_processed)):
                        
                        indicator = Levenshtein.ratio(OCR_string_processed[k], target_keyword_i[j])
    
                        if indicator >= 0.7:
    
                            medical_results_h_k.append([indicator,region_coordinates[k],OCR_string_processed[k]]) 
                            
                        if indicator >=0.5:
                            
                            medical_results_l_k.append([indicator,region_coordinates[k],OCR_string_processed[k]]) 
                            
                    
                    medical_results_h[j] = medical_results_h_k                        
                    
                    medical_results_l[j] = medical_results_l_k  
                       
                    
                def takeFirst(elem):
                    
                    return elem[0] 
                
                kid = 1
                
                length = []
                
                for i in range(len(target_keyword_i)):
                
                    medical_results_h[i].sort(key=takeFirst,reverse=True)
                    medical_results_l[i].sort(key=takeFirst,reverse=True)
                    
                medical_results = medical_results_h            
                
                for i in range(len(target_keyword_i)):
                    
                    kid = kid * len(medical_results[i])
                    
                if kid == 0:
                    
                    medical_results =  medical_results_l
                    
                    kid = 1
                                     
                for i in range(len(target_keyword_i)):
                    
                    kid = kid * len(medical_results[i])   
                    
                    length.append(len(medical_results[i]))    
                    
                print('Medical Results = ',medical_results)
                
                if kid>0:
    
                    index_num = []
                    
                    if len(target_keyword_i)>1:
                        
                        index_num = [[i,j] for i in range(length[0]) for j in range(length[1])]
                        
                        index_num_final =[]
                        
                        if len(length) > 2:
                            
                            for j in range(2,len(length)):
                                
                                for m in range(length[j]):
                                    
                                    for i in range(len(index_num)):
                                        
                                        temp = []
                                        
                                        temp = copy.deepcopy(index_num[i])
                        
                                        temp.append(m)
                                        
                                        index_num_final.append(temp)
                                        
                                index_num = index_num_final                             
                        
                        
                        rect_area = []
                        
                        for i in range(len(index_num)):
                            
                            index = index_num[i]
                            
                            rect_temp = []
                            
                            for j in range(len(index)):
                        
                                rect_temp.append(medical_results[j][index[j]][1])                               
                            
                            rect_temp = np.array(rect_temp)
                            
                            x1 = min(rect_temp[:,0])
                            y1 = min(rect_temp[:,1])
                            x2 = max(rect_temp[:,2])
                            y2 = max(rect_temp[:,3])
                            
                            rect_area.append(abs(x2-x1)*abs(y2-y1))
                            
                        
                        target_index = index_num[rect_area.index(min(rect_area))]
                        
                        
                        print('Cluster Group = ',rect_area)
                        
                        print('Positive Indicator = ',target_index)
                        
                        distance_logic = 0
                        
                        distance_index = list(itertools.combinations(list(range(len(target_index))),2))
                        
                        for k in range(len(distance_index)):
                            
                            i,j = distance_index[k]
                            
                            [xi1,yi1,xi2,yi2] = medical_results[i][target_index[i]][1]
                            
                            mid1 = [(xi2+xi1)/2,(yi1+yi2)/2]
                            
                            [xj1,yj1,xj2,yj2] = medical_results[j][target_index[j]][1]
                            
                            mid2 = [(xj2+xj1)/2,(yj1+yj2)/2]
                            
                            distance_logic = min(abs(mid1[0]-mid2[0]),abs(mid1[1]-mid2[1]))
                        
                        op_coors = []
                        
                        for i in range(len(target_index)):
                            
                            op_coors.append(medical_results[i][target_index[i]][1])
                            
                        op_coors = np.array(op_coors)
                       
                        x1m = min(op_coors[:,0])
                        y1m = min(op_coors[:,1])
                        x2m = max(op_coors[:,2])
                        y2m = max(op_coors[:,3])
                       
                        
                        passport = 0
                        
                        for i in range(len(target_index)):
                        
                            if medical_results[i][target_index[i]][0]>0.7:
                                
                                passport = 1/(len(target_index)-1)+passport
                                    
                    else:
                        
                        passport = 0
                        
                        distance_logic = 0
                        
                        if medical_results[0][0][0]>=0.8:
                            
                            passport = 1
                            
                        [x1m,y1m,x2m,y2m] = medical_results[0][0][1]    
                                       
                    if passport >=1 and distance_logic <=10:
                        
                        operation_coordinates = [left+x1m+abs(x2m-x1m)/2, top+y1m+abs(y2m-y1m)/2]                   
                                                
                        break    
        
        end = time.time()
        
        print('OCR Collapse Time = ',end-start)
    
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
# Icon Template Maching - For non-text operation
#
###############################################################################  

class Icon_Capture(object):
    
    def __init__(self):
        
        self.hm = None
        
    def on_mouse_event(self,event):
         
        if event.MessageName == "mouse left down":
            
            self.old_x, self.old_y = event.Position
            
        if event.MessageName == "mouse left up":
            
            self.new_x, self.new_y = event.Position

            self.hm.UnhookMouse()
            
            self.hm = None

        self.image = ImageGrab.grab((self.old_x, self.old_y, self.new_x, self.new_y))
        
        self.image.show()
        
        if self.image.shape[0]>0:
            
            ctypes.windll.user32.PostQuitMessage(0)
            
            print('Already Quit!')
        
        return True
    
    def capture(self):
        
        self.hm = PyHook3.HookManager()
        
        self.hm.MouseAll = self.on_mouse_event
        
        self.hm.HookMouse()
        
        pythoncom.PumpWaitingMessages()
        
        
          
###############################################################################
            

###############################################################################
#
# Image Segmentation - Wrapper of classes (capture, process, ocr)
#
###############################################################################   

class Search_Engine(object):
    
        def __init__(self,Title,Action_Sequence,Segmentation_Threshold,Kernel_Dilate_Region,Scale):
        
            self.title = Title
            
            self.action_key = Action_Sequence[0]
            
            self.target_keyword = Action_Sequence[1] 
            
            self.segmentation_threshold = Segmentation_Threshold
            
            self.kernel_dilate_reg = Kernel_Dilate_Region
            
            self.scale = Scale
                  
        def ILocate_OCR(self):            
            
            title = self.title
            
            target_keyword = self.target_keyword
            
            segmentation_threshold = self.segmentation_threshold
            
            kernel_dilate_reg = self.kernel_dilate_reg 
            
            scale = self.scale          
            
            img, window_position = Image_Capture(title).ICapture()
    
            thresh_seg, thresh_ocr = Image_Process(img, segmentation_threshold).IProcess()
            
            group_coordinates = Image_Analyze(thresh_seg, kernel_dilate_reg, scale).IBoundary(thresh_ocr)
            
            operation_coordinates = Image_OCR(thresh_ocr, window_position ,group_coordinates,target_keyword).ISearch()
        
            return operation_coordinates
        
        def ILocate_MT(self):
            
            pass
        
        def IOperate(self):
            
            action_key = self.action_key
            
            operation_coordinates= self.ILocate_OCR()
           
            MK_Event(action_key, operation_coordinates).IOperate()
                    

###############################################################################  

###############################################################################
#
# Hyperparameters Optimization - Using Grid Search algorithm
#
###############################################################################

class HyPara_Optimize(object): 

    def __init__(self, Title, Segmentation_Threshold_Group, Kernel_Dilate_Region_Group , Scale_Group ):
        
        self.title = Title
        
        self.segmentation_threshold_group = Segmentation_Threshold_Group
        
        self.kernel_dilate_reg_group = Kernel_Dilate_Region_Group
        
        self.scale_group = Scale_Group        
        
    def IGrid_Search(self):
        
        title = self.title
        
        segmentation_threshold_group = self.segmentation_threshold_group
        
        kernel_dilate_reg_group = self.kernel_dilate_reg_group
        
        scale_group = self.scale_group  
        
        target_keyword = 'GRIDSearchWordTemp'
        
        p1_group = list(range(len(segmentation_threshold_group)))
        p2_group = list(range(len(kernel_dilate_reg_group)))
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
# Tesseract 4.0 Training - image & text capture
#
###############################################################################

class Tesseract_Training(object):
    
    def __init__(self,Title,Segmentation_Threshold,Kernel_Dilate_Region,Scale):
    
        self.title = Title
        
        self.segmentation_threshold = Segmentation_Threshold
        
        self.kernel_dilate_reg = Kernel_Dilate_Region
        
        self.scale = Scale
              
    def ISample(self):            
        
        title = self.title
        
        target_word = 'Trainingkeyword'
        
        segmentation_threshold = self.segmentation_threshold
        
        kernel_dilate_reg = self.kernel_dilate_reg 
        
        scale = self.scale          
        
        img, window_position = Image_Capture(title).ICapture()

        thresh_seg, thresh_ocr = Image_Process(img, segmentation_threshold).IProcess()
        
        group_coordinates = Image_Analyze(thresh_seg, kernel_dilate_reg, scale).IBoundary(thresh_ocr)
        
        Image = Image_OCR(thresh_ocr, window_position ,group_coordinates,target_word)
 
        if os.path.isdir('Sample\\'+title):
            
            shutil.rmtree('Sample\\'+title)
            
        os.makedirs('Sample\\'+title)          
        
        for i in range(len(group_coordinates)):
            
            for j in range(len(group_coordinates[i])):
                
                Image.IOCR(group_coordinates[i][j],training_flag=True,save_path=title)
            
           
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