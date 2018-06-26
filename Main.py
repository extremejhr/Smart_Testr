# -*- coding: utf-8 -*-
###############################################################################
#                                                                             
# Smart Tester for Simcenter 3D (Version 0.0.1)                                            
#                                                                             
# First written by Haoran Ju at 21.06.2018;                                   
#                                                                             
###############################################################################

###############################################################################
#
# Testing Algorithm & Functionality: 
#
# 1. Screen Capturer -> Image Segmentation -> OCR&MatchTemplate Matching ->
#    -> Coordinates of Controller -> GUI Ooperation; 
#    (Win32 Controller ID & Handle Cooperation)
# 2. Keyword&Icon Driven;
# 3. Workflow Written Input -> Result Image/Operation Log/Error List;
# 4. Dialog Transverse Testing;
# 5. Easily Maintainence of New Functionalities.
#
###############################################################################

###############################################################################
#
# Road Map:
#
# 1. Operation Coordinates Confirmation (Keyword Driven):
#    -> Interface_Capturer/Image_Seperator ...
#    -> OCR Matching & improvement -> Operation_Locator etc.
# 2. User Input Formatting and Library enlargement;
# 3. Operation Logic Tree;
# 3. Dialog Transverse Testing;
# 4. Results Comparision;
# 5. Testing Report Generation;
# 6. Operation Record
# 7. Package....
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

###############################################################################

###############################################################################
#
# Image Segmentation -> Search Regions (22.06.2018)
#
###############################################################################

Operation_Image = 'Image\\Search.png'

img_initial = cv2.imread(Operation_Image)

h, w, _ = img_initial.shape

gray = cv2.cvtColor(img_initial, cv2.COLOR_BGR2GRAY)

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

kernel_morph = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5, 5))

closed = cv2.morphologyEx(thresh_copy, cv2.MORPH_CLOSE, kernel_morph) 

kernel_dilate = np.uint8(np.ones((8,10)))
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
        
            Search_Region_Coordinates = np.append(Search_Region_Coordinates, corner)
            
            draw_img = cv2.drawContours(img_initial, [box], -1, (0, 0, 255), 3)
        
    
cv2.imshow("blurred",blurred)      

cv2.imshow("draw_img", img_initial)
cv2.waitKey()

###############################################################################

###############################################################################
#
# OCR -> Text Matching (22.06.2018)
#
# Problem (22.06.2018): 
# 1. Efficiency seems a little bit low -> New algroithm need to be developed;
# 2. Comparision between Selection and Destination keyword need a more robust
#    approch.
#
###############################################################################
import pyautogui
import win32gui # OCR combined with win32 get handler and position.

label = 'NX 1847.1400 / Analysis_nx13.226' 

hld = win32gui.FindWindow(None, label)



Target_Keyword = 'Open'

for i in range(len(Search_Region_Coordinates)):
    
    x1 = Search_Region_Coordinates[i][0]
    y1 = Search_Region_Coordinates[i][1]
    x2 = Search_Region_Coordinates[i][2]
    y2 = Search_Region_Coordinates[i][3]
    
    hight = y2-y1
    width = x2-x1
    
    
    crop_img= Image.fromarray(thresh_cover[y1:y1+hight, x1:x1+width])
    
    OCR_string = pytesseract.image_to_string(crop_img)
    
    if Levenshtein.ratio(OCR_string, Target_Keyword)>0.6:
        #print(OCR_string)
        #print(i)
        #cv2.imwrite('icon\\icon'+str(i)+'.png',thresh_cover[y1:y1+hight, x1:x1+width])
        break
        

    #plt.ion()
    #plt.figure(i)
    #plt.imshow(crop_img)
    #plt.pause(5)
    #plt.close()
    

###############################################################################

###############################################################################
#
# SLED TESTING RECORDING (26.06.2018)
#
# Ideas: 
# 1. Record the mouse event and coordinates -> Capture the image and text:
# 2. OCR the sequecence image and text -> Recover the mouse and coordinates;
# 3. Capture the result image -> Manually comparasion.
#
# Comment:
# 1. Semi-Automation may be a good start of whole automation testing, cause it
#    seems easier to realize.
# 2. Search speed is much faster than whole automation testing.
# 3. Easy application.
#
#
###############################################################################

## Screen capture

#from PIL import ImageGrab
#import pyautogui

#print(pyautogui.position())

#im = pyautogui.screenshot()
#plt.imshow(im)
#pic = ImageGrab.grab()
#pic.save('1.jpg')

# PyHooK module listen mouse&keyboard event. - need install SWIG





###############################################################################




