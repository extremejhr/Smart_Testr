import cv2
import os
import numpy as np
import global_variables as gl


def icon_match(target_string, path):

    coordinates = []

    for i in os.walk(path):

        for j in range(len(i[2])):

            a = i[2][j]

            if target_string == a.split('.')[0].lower():

                gray_org = cv2.imread(gl.get_value('scratch_path')+'previous.png')

                icon = cv2.imread(path+a)

                gray_icon = cv2.cvtColor(icon, cv2.COLOR_BGR2GRAY)

                gray_icon = cv2.resize(gray_icon, (gray_icon.shape[1]*eval(gl.get_value('scale')),
                                                   gray_icon.shape[0]*eval(gl.get_value('scale'))))

                minloc_list = []

                minval_list = []

                minloc_list1 = []

                minval_list1 = []

                for u in range(eval(gl.get_value('thresh_hold_iteration'))):

                    gray_icon_edged = []

                    margin = int(gray_icon.shape[0]*u*0.025)

                    for k in range(margin, gray_icon.shape[0]-margin):

                        gray_icon_edged.append(gray_icon[k][margin:gray_icon.shape[1]-margin])

                    gray_icon_edged = np.array(gray_icon_edged)

                    gray_org_processed = cv2.cvtColor(gray_org, cv2.COLOR_BGR2GRAY)

                    thresh_org = cv2.adaptiveThreshold(gray_org_processed, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 3, 3)

                    thresh_icon = cv2.adaptiveThreshold(gray_icon_edged, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 3, 3)

                    res = cv2.matchTemplate(gray_icon_edged, gray_org_processed, cv2.TM_SQDIFF_NORMED)

                    res1 = cv2.matchTemplate(thresh_icon, thresh_org, cv2.TM_SQDIFF_NORMED)

                    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

                    minloc_list.append(min_loc)

                    minval_list.append(min_val)

                minloc_list = np.array(minloc_list)

                minval_list = np.array(minval_list)

                w, h = gray_icon_edged.shape[::-1]

                index = np.where(minval_list == min(minval_list))

                top_left = minloc_list[index][0]

                bottom_right = (top_left[0] + w, top_left[1] + h)

                coordinates = (np.array(top_left) + np.array(bottom_right)) / 2

                break

    return coordinates

