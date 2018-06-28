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
            
            Morph_kernel = (5,5)  
            
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
            
            Xs = np.int0([i[0] for i in box])
            Ys = np.int0([i[1] for i in box])
            
            x1 = np.int0((min(Xs) + abs(min(Xs)))/2)
            x2 = np.int0(max(Xs) if max(Xs) <= w else w)
            y1 = np.int0((min(Ys) + abs(min(Ys)))/2)
            y2 = np.int0(max(Ys) if max(Ys) <= h else h)
            
            hight = np.int0(y2-y1)
            width = np.int0(x2-x1)
            
            print(y1,x1)
            
            box = np.int0([[x1,y1],[x2,y1],[x2,y2],[x1,y2]])
            
            corner = thresh_cover[y1:y1+hight, x1:x1+width]
            
            if i == 0:
                
                Search_Region_Coordinates[i][:] = corner
            
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
        
###############################################################################
#
# Image Segmentation -> Search Regions (22.06.2018)
#
###############################################################################    
    
class Window_Capture(object):
    
    def __init__(self,title):
        
        self.title = title
        
        self.handle = None
        
    def Handle_Catch(self,hwnd,lParam):
        
        if win32gui.IsWindowVisible(hwnd) and win32gui.IsWindowEnabled(hwnd):
            
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
        
                

a = Window_Capture('Pre/Post')

b,hwnd = a.Image_Capture()

c = Image_Segmentation(b)

d = c.Get_Region(plot_flag=True)