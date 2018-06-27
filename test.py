# -*- coding: utf-8 -*-
"""
Created on Wed Jun 27 17:24:49 2018

@author: xqk9qq
"""

import win32gui

class Window_Capture(object):
    
    def __init__(self,title):
        
        self.title = title
        
    def Handle_Catch(self,hwnd,lParam):
        
        if win32gui.IsWindowVisible(hwnd):
            
            if self.title in win32gui.GetWindowText(hwnd):
                
                self.handle = hwnd
                
            else:
                
                self.handle = None
   
    def Image_Capture(self):
        
        win32gui.EnumWindows(self.Handle_Catch, None)
        
        if self.handle == None:
            
            print('No Window Found!')
            
        else:
            
            win32gui.SetForegroundWindow(self.handle)
            bbox = win32gui.GetWindowRect(self.handle)
            img = ImageGrab.grab(bbox)
            img.show()
    
        
                

a = Window_Capture('ss')