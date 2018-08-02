# -*- coding: utf-8 -*-
"""
Created on Thu Jul 26 09:06:40 2018

@author: xqk9qq
"""
import pywinauto
from pywinauto.application import Application

app = Application().connect(process = 11472)

dlg = app.window(title_re="Physical Property Table Manager")

#dlg.print_control_identifiers(filename='file.txt')


criteria = dlg.criteria
this_ctrl=dlg._WindowSpecification__resolve_control(dlg.criteria,)[-1]
all_ctrls = [this_ctrl, ] + this_ctrl.descendants()



for ctrl in all_ctrls1:
    
    if hasattr(ctrl.element_info, 'automation_id'):
    
        auto_id =ctrl.element_info.automation_id
