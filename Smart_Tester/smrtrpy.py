# -*- coding: utf-8 -*-
"""
Smart Tester Engine 0.0.0.3

- Improve OCR Algorithm
- Add New Group Algorithm

"""

from PIL import ImageGrab
import cv2
import numpy as np
import win32gui
import win32con
import time
import pytesseract
import concurrent.futures
import itertools
import circumcircle
import pyautogui
import Levenshtein
import ocr_space
from skimage.measure import compare_ssim
import imutils
import global_variables as gl
import text_detection as td
import icon_template_match as itm


class ImageDiff(object):

    def diff_image(self, gray_orig, gray_mod):

        margin = eval(gl.get_value('margin'))

        gray_orig_1 = cv2.adaptiveThreshold(gray_orig, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 3, 3)

        gray_mod_1 = cv2.adaptiveThreshold(gray_mod, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 3, 3)

        area_max = 0

        rect = []

        rect1 = []

        (score, diff) = compare_ssim(gray_orig_1, gray_mod_1, full=True)

        diff = (diff * 255).astype("uint8")

        thresh = cv2.threshold(diff, 127, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        cnts = imutils.grab_contours(cnts)

        for c in cnts:

            (x, y, w, h) = cv2.boundingRect(c)

            area = w * h

            if area >= area_max:

                area_max = area

                rect = [x, y, w, h]

        gray_diff = []

        rect_m = [rect[0], rect[1], rect[2], rect[3]]

        for i in range(rect[1], rect[1] + rect[3]):

            tmp = gray_mod[i][rect[0]:rect[0] + rect[2]]

            gray_diff.append(tmp)

        gray_diff = np.array(gray_diff, dtype='uint8')

        thresh = cv2.adaptiveThreshold(gray_diff, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 3, 3)

        cnts1 = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        cnts2 = imutils.grab_contours(cnts1)

        area_max = 0

        for c in cnts2:

            (x, y, w, h) = cv2.boundingRect(c)

            area = w * h

            if area >= area_max:

                area_max = area

                rect = [x+rect_m[0], y+rect_m[1], w, h]

        gray_diff1 = []

        rect_m1 = [rect[0], rect[1], rect[2], rect[3]]

        for i in range(rect[1] + margin, rect[1] + rect[3] - margin):

            tmp = gray_mod[i][rect[0] + margin:rect[0] + rect[2] - margin]

            gray_diff1.append(tmp)

        cv2.imwrite(gl.get_value('scratch_path') + 'diff.png', np.array(gray_diff1))

        gl.set_value('DIFF_POSITION', rect_m1)

        return gray_diff1


class ImageCapture(object):

    def __init__(self, title):

        self.title = title

        self.handle = None

        self.title_flag_nm = title+'_flag'

    def handle_catch(self, hwnd, lParam):

        if win32gui.IsWindowVisible(hwnd):

            if self.title in win32gui.GetWindowText(hwnd).lower():

                self.handle = hwnd

    def icapture(self, window_type, diff_flag):

        size = []

        position = []

        if window_type == 'main':

            position = eval(gl.get_value('main_window_position'))

            size = eval(gl.get_value('main_window_size'))

        elif window_type == 'dialog':

            position = eval(gl.get_value('dialog_position'))

            size = eval(gl.get_value('dialog_size'))

        else:

            print('Window type should be only main or dialog !!')

        margin = eval(gl.get_value('margin'))

        scale = eval(gl.get_value('scale'))

        x_axis = 1*(size[0] - 2 * margin)

        y_axis = 1*(size[1] - 2 * margin)

        screen_shot = [[np.array([int(scale*x_axis), int(scale*y_axis), position[0], position[1], 0])], ]

        win32gui.EnumWindows(self.handle_catch, None)

        if self.handle is None:

            print('No Main Window Found!')

        else:

            reset_flag = gl.get_value(self.title_flag_nm)

            if reset_flag != 'close':

                win32gui.SetForegroundWindow(self.handle)

                gl.set_value(self.title_flag_nm, 'close')

            rect = win32gui.GetWindowRect(self.handle)

            window_position_current = rect[0:2]

            window_size_current = rect[2:4]

            if (list(window_position_current) != position) or (list(window_size_current) != size):

                win32gui.SetWindowPos(self.handle, win32con.HWND_TOPMOST, position[0], position[1], size[0], size[1],
                                      win32con.SWP_SHOWWINDOW)

                time.sleep(0.5)

            img = np.array(ImageGrab.grab((position[0]+margin, position[1]+margin, position[0]+size[0]-margin,
                                           position[1]+size[1]-margin)))

            img = cv2.resize(img, (int(scale*(size[0]-2*margin)), int(scale*(size[1]-2*margin))))

            cv2.imwrite(gl.get_value('scratch_path')+'workbench.png', img)

            gray_mod = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            if diff_flag is True:

                gray_orig = cv2.imread(gl.get_value('scratch_path') + 'previous.png')

                gray_orig = cv2.cvtColor(gray_orig, cv2.COLOR_BGR2GRAY)

                gray_diff = ImageDiff().diff_image(gray_orig, gray_mod)

                rect = gl.get_value('DIFF_POSITION')

                x_axis = 1 * (rect[2]-2*margin)

                y_axis = 1 * (rect[3]-2*margin)

                screen_shot1 = [[np.array([int(1 * x_axis), int(1 * y_axis), rect[0], rect[1], 0])], ]

                screen_shot1[0].append(np.array(gray_diff).ravel())

                screen_shot = screen_shot1

            else:

                screen_shot[0].append(np.array(gray_mod).ravel())

            cv2.imwrite(gl.get_value('scratch_path') + 'previous.png', gray_mod)

        return [screen_shot]


class UIScissors(object):

    def __init__(self, screen_shot):

            self.origin = screen_shot

            self.x_axis = screen_shot[0][0]

            self.y_axis = screen_shot[0][1]

            self.positionx = screen_shot[0][2]

            self.positiony = screen_shot[0][3]

            self.cflag = screen_shot[0][4]

            img = screen_shot[1].reshape(self.y_axis, self.x_axis)

            img_blur = cv2.GaussianBlur(img, eval(gl.get_value('gaussian_blur')), 5)

            #cv2.imshow('', img_blur)

            #cv2.waitKey()

            self.pic_blur = np.array(img_blur).ravel()

            self.pic_original = np.array(screen_shot[1])

    def cutting_line(self, axis):

        x_axis, y_axis = [0, 0]

        pic_blur = []

        pic_original = []

        if axis == 'x':

            x_axis = self.x_axis

            y_axis = self.y_axis

            pic_blur = self.pic_blur.copy()

            pic_original = self.pic_original.copy()

        elif axis == 'y':

            x_axis = self.y_axis

            y_axis = self.x_axis

            pic_blur = np.transpose(self.pic_blur.reshape(self.y_axis, self.x_axis)).ravel()

            pic_original = np.transpose(self.pic_original.reshape(self.y_axis, self.x_axis)).ravel()

        scan_lines_temp = [0]

        scan_lines = []

        gap = []

        for i in range(0, y_axis):

            percent = eval(gl.get_value('edge_remove_para'))

            std = np.std(pic_blur[i*x_axis+int(x_axis*percent):(i+1)*x_axis-int(x_axis*percent)])

            #print(std)

            if std <= eval(gl.get_value('cutting_threshold')):

                scan_lines_temp.append(i)

        if len(scan_lines_temp) > 0:

            scan_lines_temp.append(y_axis)

            for j in range(1, len(scan_lines_temp)):

                gap.append(scan_lines_temp[j]-scan_lines_temp[j-1])

            gap = np.array(gap)

            index = np.where(gap >= 2)[0]

            for k in range(len(index)):

                scan_lines.append([scan_lines_temp[index[k]], scan_lines_temp[index[k]+1]])

            scan_lines = np.array(scan_lines)

            if len(scan_lines) > 0:

                if y_axis - scan_lines[-1][1] > 0:

                    scan_lines = np.append(scan_lines, [[scan_lines[-1][1], y_axis]], axis=0)

        return scan_lines, x_axis, y_axis, pic_original

    def sep_reg(self, axis):

        scan_lines, x_axis, y_axis, screen_shot = self.cutting_line(axis)

        region = []

        if len(scan_lines) > 0:

            for i in range(len(scan_lines)):

                tmp = screen_shot[scan_lines[i][0] * x_axis:scan_lines[i][1]*x_axis]

                if axis == 'x':

                    x = x_axis

                    y = int(len(tmp)/x_axis)

                    positionx = self.positionx

                    positiony = scan_lines[i][0]+self.positiony

                    region_temp = tmp

                else:

                    x = int(len(tmp)/x_axis)

                    y = x_axis

                    positionx = scan_lines[i][0]+self.positionx

                    positiony = self.positiony

                    region_temp = np.transpose(tmp.reshape(x, y)).ravel()

                region.append([np.array([x, y, positionx, positiony, self.cflag]), np.array(region_temp)])

        return region

    def cutting_axis(self):

        region = []

        if self.cflag == 0:

            futures = []

            regions = []

            with concurrent.futures.ThreadPoolExecutor(1) as executor:

                for axis in ['x', 'y']:

                    future = executor.submit(self.sep_reg, axis)

                    futures.append(future)

                for future in concurrent.futures.as_completed(futures):

                    regions.append(future.result())

            if (len(regions[0]) <= 1) and (len(regions[1]) <= 1) and max((len(regions[0]), len(regions[1]))) > 0:

                if len(regions[0]) >= len(regions[1]):

                    regions[0][0][0][4] = 1

                else:

                    regions[1][0][0][4] = 1

            if len(regions[0]) >= len(regions[1]):

                region = regions[0]

            else:

                region = regions[1]

        elif self.cflag == 1:

            region = [self.origin]

        return region


class ImgTree(object):

    def img_group(self, regions):

        thresh = regions[0][1].reshape(regions[0][0][1], regions[0][0][0])

        thresh1 = cv2.adaptiveThreshold(thresh, 200, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 3, 3)

        thresh1 = cv2.GaussianBlur(thresh1, (7, 7), 1)

        cnts1 = cv2.findContours(thresh1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        cnts2 = imutils.grab_contours(cnts1)

        rect = []

        for c in cnts2:

            (x, y, w, h) = cv2.boundingRect(c)

            #cv2.rectangle(thresh, (x,y), (x+w,y+h), [0,0,255])

            rect.append([w, h, x, y+h])

        #cv2.imshow('', thresh)

        #cv2.waitKey()

        return rect

    def next_depth(self, rd1, depth):

        rd2 = []

        for i in range(len(rd1)):

            for j in range(len(rd1[i])):

                tmp = UIScissors(rd1[i][j]).cutting_axis()

                for k in range(len(tmp)):

                    rd2.append([tmp[k]])

        while depth-1 > 0:

            depth = depth - 1

            return self.next_depth(rd2, depth)

        return rd2

    def img_save(self, rd, out_flag):

        img_org = cv2.imread(gl.get_value('scratch_path') + 'workbench.png')

        path = gl.get_value('icon_path')

        for i in range(len(rd)):

            w, h, x, y, _ = rd[i][0][0]

            img = []

            for j in range(y, y+h):

                img.append(img_org[j][x:x+w])

            if out_flag == 'OI':

                if td.text_detector(np.array(img), 320, 320, 0.5) == 0:

                    cv2.imwrite(path+str(i)+'.png', np.array(img))

            elif out_flag == 'ALL':

                cv2.imwrite(path + str(i) + '.png', np.array(img))

        #cv2.imshow('', cv2.imread('running_temp_file\\Regions.png'))

        #cv2.waitKey()

    def tesseract_ocr(self, rd, i, j, path, img):

        p1 = (rd[0][2], rd[0][3])

        p2 = (rd[0][0] + rd[0][2], rd[0][1] + rd[0][3])

        x1 = int((abs(p1[1]-1)+p1[1]-1)/2)

        y1 = int((abs(p1[0]-1)+p1[0]-1)/2)

        if p2[1]+1 < img.shape[0]:

            x2 = p2[1]+1

        else:

            x2 = img.shape[0]

        if p2[0] + 1 < img.shape[1]:

            y2 = p2[0]+1

        else:

            y2 = img.shape[1]

        cv2.imwrite(path + str(i) + '-' + str(j) + '.png', img[x1:x2 + 1, y1:y2 + 1])


class AreaDivide(object):

    def region_init_sep(self):

        test_window = ImageCapture(gl.get_value('version').lower())

        region = test_window.icapture('main', False)

        tree = ImgTree()

        RD0 = tree.next_depth(region, 1)

        regions_up = []

        for i in range(len(RD0)):

            if RD0[i][0][0][1] >= 30:

                regions_up.append(RD0[i])

            if len(regions_up) > 4:

                break

        RD1 = tree.next_depth([regions_up[-1]], 1)

        regions = regions_up[0:len(regions_up)-1]

        for i in range(3):

            if len(regions) > 7:

                break

            if RD1[i][0][0][1] >= 30:

                regions.append(RD1[i])

        for i in range(len(regions)):

            cv2.imwrite('scratch\\'+str(i)+'.png', regions[i][0][1].reshape(regions[i][0][0][1], regions[i][0][0][0]))

        gl.set_value('top_border', regions[0][0][0])

        gl.set_value('ribbon_up', regions[1][0][0])

        gl.set_value('ribbon_down', regions[2][0][0])

        gl.set_value('quick_access', regions[3][0][0])

        gl.set_value('left_border', regions[4][0][0])

        gl.set_value('navigator', regions[5][0][0])

        gl.set_value('main_window', regions[6][0][0])

    def region_sep_main(self, act_type):

        test_window = ImageCapture(gl.get_value('version').lower())

        region = test_window.icapture('main', act_type)

        regions = region

        return regions

    def region_sep_dialog(self, title, act_type):

        test_window = ImageCapture(title)

        region = test_window.icapture('dialog', act_type)

        return region


class RegionOCR(object):

    def __init__(self):

        self.method = gl.get_value('ocr_method')

    def text_location(self, region):

        if self.method == 'tesseract':

            h, w = region.shape

            OCR_rlt = pytesseract.image_to_boxes(region)

            if len(OCR_rlt) > 0:

                OCR_sort = OCR_rlt.split('\n')

                for i in range(len(OCR_sort)):

                    OCR_sort[i] = OCR_sort[i].split(' ')

                    OCR_sort[i][1] = int(OCR_sort[i][1])

                    OCR_sort[i][2] = h - int(OCR_sort[i][2])

                    OCR_sort[i][3] = int(OCR_sort[i][3])

                    OCR_sort[i][4] = h - int(OCR_sort[i][4])

                    OCR_sort[i][5] = int(OCR_sort[i][5])

            else:

                OCR_sort = [['', 0, 0, 0, 0]]

        elif self.method == 'OCR_Space':

            path = gl.get_value('scratch_path')+str(time.time())+'.png'

            time.sleep(0.5)

            cv2.imwrite(path, region)

            OCR_sort = ocr_space.image_to_boxes(path, api_key='0d5c85b2d388957')

        else:

            OCR_sort = [['', 0, 0, 0, 0]]

        return OCR_sort


class StringLoc(object):

    def ocr_concurrent_wrap(self, i, rd, RD4, Whole_Region1):

        temp = rd

        block_size = 3 + i * 2

        Whole_Region = cv2.adaptiveThreshold(Whole_Region1, 200, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,
                                             block_size, block_size)

        OCR_sort = RegionOCR().text_location(Whole_Region)

        string_loc = []

        for k in range(len(RD4)):

            string = []

            p1 = (RD4[k][0][0][2] - temp[0][0][2], RD4[k][0][0][3] - temp[0][0][3])

            p2 = (RD4[k][0][0][0] + RD4[k][0][0][2] - temp[0][0][2], RD4[k][0][0][1] + RD4[k][0][0][3] - temp[0][0][3])

            for i in range(len(OCR_sort)):

                OCR_line = OCR_sort[i]

                x_l = int(OCR_line[1])
                y_t = int(OCR_line[2])

                x_r = int(OCR_line[3])
                y_b = int(OCR_line[4])

                c1 = x_r < p1[0]
                c2 = x_l > p2[0]

                c3 = y_t < p1[1]
                c4 = y_b > p2[1]

                if not (c1 or c2 or c3 or c4):

                    string.append(OCR_line[0])

            string1 = ''.join(e for e in string if e.isalnum())

            string_loc.append(string1)

        return string_loc

    def ocr_concurrent_wrap_img(self, i, rd, RD4, Whole_Region1):

        temp = rd

        block_size = 3 + i * 2

        Whole_Region = cv2.adaptiveThreshold(Whole_Region1, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,
                                             block_size, block_size)

        OCR_sort = RegionOCR().text_location(Whole_Region)

        string_loc = []

        for k in range(len(RD4)):

            string = []

            p1 = (RD4[k][2] - temp[0][0][2], RD4[k][3] - temp[0][0][3])

            p2 = (RD4[k][0] + RD4[k][2] - temp[0][0][2], -RD4[k][1] + RD4[k][3] - temp[0][0][3])

            for i in range(len(OCR_sort)):

                OCR_line = OCR_sort[i]

                x_l = int(OCR_line[1])
                y_t = int(OCR_line[2])

                x_r = int(OCR_line[3])
                y_b = int(OCR_line[4])

                c1 = x_r < p1[0]
                c2 = x_l > p2[0]

                c3 = y_t < p1[1]
                c4 = y_b > p2[1]

                if not (c1 or c2 or c3 or c4):

                    string.append(OCR_line[0])

            string1 = ''.join(e for e in string if e.isalnum())

            string_loc.append(string1)

        return string_loc

    def string_ocr_img(self, rd):

        futures = []

        string_loc = []

        tree = ImgTree()

        temp = rd

        block_size_iter = eval(gl.get_value('block_size_iter'))

        RD4 = tree.img_group(rd)

        Whole_Region1 = temp[0][1].reshape(temp[0][0][1], temp[0][0][0])

        #cv2.imshow('', np.array(Whole_Region1))

        #cv2.waitKey()

        with concurrent.futures.ThreadPoolExecutor() as executor:

            for i in range(block_size_iter[0], block_size_iter[1]):

                future = executor.submit(self.ocr_concurrent_wrap_img, i, rd, RD4, Whole_Region1)

                futures.append(future)

            for future in concurrent.futures.as_completed(futures):

                string_loc.append(future.result())

        strings_all = []

        for i in range(len(RD4)):

            string_cl = []

            for j in range(len(string_loc)):

                string_cl.append(string_loc[j][i])

            strings_all.append([string_cl, [RD4[i][0] / 2 + RD4[i][2], RD4[i][1] / 2 + RD4[i][3]]])

        return strings_all

    def string_ocr(self, rd):

        futures = []

        string_loc = []

        tree = ImgTree()

        temp = rd

        end_depth = eval(gl.get_value('end_depth'))

        block_size_iter = eval(gl.get_value('block_size_iter'))

        RD4 = tree.next_depth([rd],  end_depth)

        Whole_Region1 = temp[0][1].reshape(temp[0][0][1], temp[0][0][0])

        #cv2.imshow('', np.array(Whole_Region1))

        #cv2.waitKey()

        with concurrent.futures.ThreadPoolExecutor() as executor:

            for i in range(block_size_iter[0], block_size_iter[1]):

                future = executor.submit(self.ocr_concurrent_wrap, i, rd, RD4, Whole_Region1)

                futures.append(future)

            for future in concurrent.futures.as_completed(futures):

                string_loc.append(future.result())

        strings_all = []

        for i in range(len(RD4)):

            string_cl = []

            for j in range(len(string_loc)):

                string_cl.append(string_loc[j][i])

            strings_all.append([string_cl, [RD4[i][0][0][0]/2+RD4[i][0][0][2], RD4[i][0][0][1]/2+RD4[i][0][0][3]]])

        return strings_all

    def string_levenstein(self, string, strings_all):

        positive_index = []

        for i in range(len(strings_all)):

            l_ratio = []

            for j in range(len(strings_all[i][0])):

                l_ratio.append(Levenshtein.distance(string.lower(), strings_all[i][0][j].lower())/(len(string)+len(strings_all[i][0][j])))

            positive_index.append(min(l_ratio))

        positive_index = np.array(positive_index)

        if min(positive_index) <= 0.15:

            index = np.where(positive_index == min(positive_index))[0]

        else:

            index = []

        return index

    def string_process(self, string_input):

        string_temp = string_input.split(' ')

        string_combination = [string_temp]

        for i in range(1, len(string_temp)):

            for j in range(len(string_temp) - i):

                string_sort = ''.join(string_temp[j:j + i + 1])

                string_combination.append([''.join(string_temp[0:j]), string_sort, ''.join(string_temp[j + i + 1:])])

        for i in range(len(string_combination)):

            while '' in string_combination[i]:

                string_combination[i].remove('')

        return string_combination

    def string_index(self, string_combination, strings_all):

        string_index = []

        for i in range(len(string_combination)):

            string_index1 = []

            for j in range(len(string_combination[i])):

                string = string_combination[i][j]

                index = list(self.string_levenstein(string, strings_all))

                if len(index) == 0:

                    string_index1 = []

                    break

                else:

                    string_index1.append(index)

            if len(string_index1) > 0:

                string_index.append(string_index1)

        string_index_all = string_index

        target_index_all = []

        for k in range(len(string_index_all)):

            string_index = string_index_all[k]

            if len(string_index) == 1:

                target_index = []

                for i in range(len(string_index[0])):

                    target_index.append([string_index[0][i]])

            elif len(string_index) == 2:

                target_index = []

                target_index_temp = list(itertools.product(string_index[0], string_index[1]))

                for i in range(len(target_index_temp)):

                    target_index.append(list(target_index_temp[i]))

            else:

                target_index = list(itertools.product(string_index[0], string_index[1]))

                for i in range(2, len(string_index)):

                    target_index = list(itertools.product(target_index, string_index[i]))

                    for j in range(len(target_index)):

                        temp = target_index[j]

                        index_temp = []

                        for k in range(len(temp[0])):

                            index_temp.append(temp[0][k])

                        index_temp.append(temp[1])

                        target_index[j] = index_temp

            target_index_filter = []

            for p in range(len(target_index)):

                flag = 1

                for o in range(len(target_index[p])-1):

                    c1 = strings_all[target_index[p][o]][1][0] <= strings_all[target_index[p][o+1]][1][0]

                    c2 = strings_all[target_index[p][o]][1][1] <= strings_all[target_index[p][o+1]][1][1]

                    flag = flag and (c1 or c2)

                if flag:

                    target_index_filter.append(target_index[p])

            target_index_all.append(target_index_filter)

        return target_index_all

    def action_coordinates(self, target_index, strings_all):

        coordinates_all = []

        for n in range(len(target_index)):

            if len(target_index[n][0]) == 1:

                points = []

                for i in range(len(target_index[n][0])):

                    points.append(strings_all[target_index[n][0][i]][1])

                coordinates = np.mean(points, axis=0)

                coordinates_all.append([0, coordinates[0], coordinates[1]])

            else:

                radius = []

                mid_point = []

                for i in range(len(target_index[n])):

                    points = []

                    for j in range(len(target_index[n][i])):

                        points.append(strings_all[target_index[n][i][j]][1])

                    mid_point.append(np.mean(points, axis=0))

                    circle = circumcircle.make_circle(points)

                    radius.append(circle[2])

                radius = np.array(radius)

                index = np.where(radius == min(radius))[0]

                coordinates = mid_point[index[0]]

                coordinates_all.append([min(radius), coordinates[0], coordinates[1]])

        coordinates_all.sort()

        coordinates_act = np.array(coordinates_all[0][1:])

        return coordinates_act

    def target_anchor_achieve(self, target_index, strings_all):

        target_index_anchor = []

        for n in range(len(target_index)):

            radius = []

            mid_point = []

            for i in range(len(target_index[n])):

                points = []

                for j in range(len(target_index[n][i])):

                    points.append(strings_all[target_index[n][i][j]][1])

                mid_point.append(np.mean(points, axis=0))

                circle = circumcircle.make_circle(points)

                radius.append(circle[2])

            radius = np.array(radius)

            index = np.where(radius == min(radius))[0]

            target_index_anchor.append(target_index[n][index[0]])

        return target_index_anchor


class MKAction(object):

    def MB1(self, coordinates):

        pyautogui.click(coordinates[0], coordinates[1], button='left')

    def DMB1(self, coordinates):

        pyautogui.click(coordinates[0], coordinates[1], clicks=2)

    def MB2(self):

        pyautogui.click(button='middle')

    def MB3(self, coordinates):

        pyautogui.click(coordinates[0], coordinates[1], button='right')

    def RINPUT(self, coordinates, act_input):

        coordinates_rinput = [coordinates[0] + eval(gl.get_value('input_xgap')), coordinates[1]]

        pyautogui.click(coordinates_rinput, clicks=2)

        pyautogui.press('backspace')

        pyautogui.typewrite(act_input)

    def BINPUT(self, coordinates, act_input):

        coordinates_binput = [coordinates[0], coordinates[1]+eval(gl.get_value('input_ygap'))]

        pyautogui.click(coordinates_binput, clicks=2)

        pyautogui.press('backspace')

        pyautogui.typewrite(act_input)

    def RMB1(self, coordinates):

        coordinates_rinput = [coordinates[0] + eval(gl.get_value('list_xgap')), coordinates[1]]

        pyautogui.click(coordinates_rinput)

    def BMB1(self, coordinates):

        coordinates_binput = [coordinates[0], coordinates[1]+eval(gl.get_value('lsit_ygap'))]

        pyautogui.click(coordinates_binput)

    def PRESS(self, key):

        pyautogui.press(key)

    def CURSOR_REST(self):

        window_position = eval(gl.get_value('main_window_position'))

        move_coordinates = np.array([window_position[0], window_position[1]]) + np.array([20, 20])

        pyautogui.moveTo(move_coordinates[0], move_coordinates[1])

    def BUS_STOP(self):

        pyautogui.moveTo(0, 0)


def search_area(script_input):

    mother_canvas, child_canvas, act_key, target, act_input, act_type, anchor = script_input

    search_area = []

    capture_diff = ''

    if act_type in ['RESET', 'REMAIN']:

        if act_type == 'RESET':

            capture_diff = False

        elif act_type == 'REMAIN':

            capture_diff = True

        if mother_canvas == 'main':

            if capture_diff:

                regions = AreaDivide().region_sep_main(capture_diff)

                search_area = regions[0]

            else:

                test_window = ImageCapture(gl.get_value('version').lower())

                test_window.icapture('main', False)

                img = cv2.imread(gl.get_value('scratch_path') + 'previous.png')

                regions = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                if child_canvas == 'top_border':

                    frame_area = gl.get_value('top_border')

                    search_area = []

                    for i in range(frame_area[3], frame_area[3] + frame_area[1]):

                        search_area.append(regions[i][frame_area[2] : frame_area[2] + frame_area[0]])

                    search_area = [[frame_area, np.array(search_area).ravel()]]

                elif child_canvas == 'ribbon_up':

                    frame_area = gl.get_value('ribbon_up')

                    search_area = []

                    for i in range(frame_area[3], frame_area[3] + frame_area[1]):

                        search_area.append(regions[i][frame_area[2]:frame_area[2] + frame_area[0]])

                    search_area = [[frame_area, np.array(search_area).ravel()]]

                elif child_canvas == 'ribbon_down':

                    frame_area = gl.get_value('ribbon_down')

                    search_area = []

                    for i in range(frame_area[3], frame_area[3] + frame_area[1]):

                        search_area.append(regions[i][frame_area[2]:frame_area[2] + frame_area[0]])

                    search_area = [[frame_area, np.array(search_area).ravel()]]

                elif child_canvas == 'quick_access':

                    frame_area = gl.get_value('quick_access')

                    search_area = []

                    for i in range(frame_area[3], frame_area[3] + frame_area[1]):

                        search_area.append(regions[i][frame_area[2]:frame_area[2] + frame_area[0]])

                    search_area = [[frame_area, np.array(search_area).ravel()]]

                elif child_canvas == 'left_border':

                    frame_area = gl.get_value('left_border')

                    search_area = []

                    for i in range(frame_area[3], frame_area[3] + frame_area[1]):

                        search_area.append(regions[i][frame_area[2]:frame_area[2] + frame_area[0]])

                    search_area = [[frame_area, np.array(search_area).ravel()]]

                elif child_canvas == 'navigator':

                    frame_area = gl.get_value('navigator')

                    search_area = []

                    for i in range(frame_area[3], frame_area[3] + frame_area[1]):

                        search_area.append(regions[i][frame_area[2]:frame_area[2] + frame_area[0]])

                    search_area = [[frame_area, np.array(search_area).ravel()]]

                elif child_canvas == 'main_window':

                    frame_area = gl.get_value('main_window')

                    search_area = []

                    for i in range(frame_area[3], frame_area[3] + frame_area[1]):

                        search_area.append(regions[i][frame_area[2]:frame_area[2] + frame_area[0]])

                    search_area = [[frame_area, np.array(search_area).ravel()]]

        else:

            regions = AreaDivide().region_sep_dialog(child_canvas, capture_diff)

            search_area = regions[0]

        '''

        search_img = search_area[0][1].reshape(search_area[0][0][1], search_area[0][0][0])

        thresh = cv2.adaptiveThreshold(search_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 3, 3)

        Whole_Region_blur = cv2.GaussianBlur(thresh, (7, 7), 1)

        edges = cv2.Canny(Whole_Region_blur, 100, 200)

        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 20, minLineLength=20, maxLineGap=0)

        if lines is not None:

            for i in range(len(lines)):

                line = lines[i]

                cv2.line(thresh, (line[0][0], line[0][1]), (line[0][2], line[0][3]), (255, 255, 255), 5)

        search_area = [[search_area[0][0], thresh.ravel()]]
        
    '''

    return search_area


def coordinates_act(script_input, search_area):

    mother_canvas, child_canvas, act_key, target, act_input, act_type, anchor = script_input

    scale = eval(gl.get_value('scale'))

    margin = eval(gl.get_value('margin'))

    search_iteration = eval(gl.get_value('thresh_hold_iteration'))

    if mother_canvas in ['main', 'dialog']:

        if mother_canvas == 'main':

            window_position = eval(gl.get_value('main_window_position'))

            window_size = eval(gl.get_value('main_window_size'))

        else:

            window_position = eval(gl.get_value('dialog_position'))

            window_size = eval(gl.get_value('dialog_size'))

        if act_key in ['mb1', 'mb3', 'dmb1', 'rinput', 'binput', 'rmb1', 'bmb1']:

            coordinates = itm.icon_match(target, gl.get_value('icon_path'))

            if len(coordinates) == 0:

                for m in range(search_iteration):

                    strings_all = StringLoc().string_ocr(search_area)

                    print(strings_all)

                    if anchor == 'invalid':

                        string_combination = StringLoc().string_process(target)

                        target_index = StringLoc().string_index(string_combination, strings_all)

                        if len(target_index) == 0:

                            th = eval(gl.get_value('cutting_threshold'))

                            print('Cutting Threshold = ', th)

                            gl.set_value('cutting_threshold', str(th+eval(gl.get_value('thresh_hold_step'))))

                            continue

                        else:

                            coordinates = StringLoc().action_coordinates(target_index, strings_all)

                            break

                    else:

                        target_index_final = []

                        string_combination = StringLoc().string_process(target)

                        target_index = StringLoc().string_index(string_combination, strings_all)

                        if len(target_index) == 0:

                            th = eval(gl.get_value('cutting_threshold'))

                            print('Cutting Threshold = ', th)

                            gl.set_value('cutting_threshold', str(th + eval(gl.get_value('thresh_hold_step'))))

                            continue

                        else:

                            string_combination_anchor = StringLoc().string_process(anchor + ' ' + target)

                            target_index_anchor = StringLoc().string_index(string_combination_anchor, strings_all)

                            minimal_target = StringLoc().target_anchor_achieve(target_index_anchor, strings_all)

                            for k in range(len(target_index[0])):

                                if target_index[0][k][0] in minimal_target[0]:

                                    target_index_final = [[target_index[0][k]]]

                                    break

                            coordinates = StringLoc().action_coordinates(target_index_final, strings_all)

                            break

                if len(coordinates) == 0:

                    print('Act Location Not Found !!!')

                coordinates = np.array(coordinates)/scale + np.array([margin, margin])\
                              + np.array([window_position[0], window_position[1]])/scale

            else:

                coordinates = np.array(coordinates) / scale + np.array([margin, margin]) \
                              + np.array([window_position[0], window_position[1]])

            if act_key == 'mb1':

                MKAction().MB1(coordinates)

            elif act_key == 'mb3':

                MKAction().MB3(coordinates)

            elif act_key == 'dmb1':

                MKAction().DMB1(coordinates)

            elif act_key == 'rinput':

                MKAction().RINPUT(coordinates, act_input)

            elif act_key == 'binput':

                MKAction().BINPUT(coordinates, act_input)

            elif act_key == 'rmb1':

                MKAction().RMB1(coordinates)

            elif act_key == 'bmb1':

                MKAction().BMB1(coordinates)

            else:

                print('Wrong Mouse Action Keyword !!!')

        elif act_key == 'press':

            MKAction().MB1(np.array([window_position[0]+window_size[0]/2, window_position[1]]) + np.array([0, 10]))

            time.sleep(0.5)

            MKAction().PRESS(act_input)

        elif act_key == 'mb2':

            MKAction().MB1(np.array([window_position[0] + window_size[0] / 2, window_position[1]]) + np.array([0, 10]))

            time.sleep(0.5)

            MKAction().MB2()

        else:

            print('Wrong Keyboard Action Keyword !!!')

        MKAction().CURSOR_REST()

    else:

        if act_key == 'wait':

            time.sleep(act_input)

        elif act_key == 'finish':

            MKAction().BUS_STOP()

        else:

            print('Wrong System Command!!!')


def coordinates_act_img(script_input, search_area):

    mother_canvas, child_canvas, act_key, target, act_input, act_type, anchor = script_input

    scale = eval(gl.get_value('scale'))

    margin = eval(gl.get_value('margin'))

    if mother_canvas in ['main', 'dialog']:

        if mother_canvas == 'main':

            window_position = eval(gl.get_value('main_window_position'))

        else:

            window_position = eval(gl.get_value('dialog_position'))

        if act_key in ['mb1', 'mb3', 'dmb1', 'rinput', 'binput', 'rmb1', 'bmb1']:

            coordinates = itm.icon_match(target, gl.get_value('icon_path'))

            if len(coordinates) == 0:

                strings_all = StringLoc().string_ocr_img(search_area)

                #print(strings_all)

                if anchor == 'invalid':

                    string_combination = StringLoc().string_process(target)

                    target_index = StringLoc().string_index(string_combination, strings_all)

                    coordinates = StringLoc().action_coordinates(target_index, strings_all)

                else:

                    target_index_final = []

                    string_combination = StringLoc().string_process(target)

                    target_index = StringLoc().string_index(string_combination, strings_all)

                    string_combination_anchor = StringLoc().string_process(anchor + ' ' + target)

                    target_index_anchor = StringLoc().string_index(string_combination_anchor, strings_all)

                    minimal_target = StringLoc().target_anchor_achieve(target_index_anchor, strings_all)

                    for k in range(len(target_index[0])):

                        if target_index[0][k][0] in minimal_target[0]:

                            target_index_final = [[target_index[0][k]]]

                            break

                    coordinates = StringLoc().action_coordinates(target_index_final, strings_all)

                if len(coordinates) == 0:

                    print('Act Location Not Found !!!')

            coordinates = np.array(coordinates)/scale + np.array([margin, margin])\
                          + np.array([window_position[0], window_position[1]])/scale

            if act_key == 'mb1':

                MKAction().MB1(coordinates)

            elif act_key == 'mb3':

                MKAction().MB3(coordinates)

            elif act_key == 'dmb1':

                MKAction().DMB1(coordinates)

            elif act_key == 'rinput':

                MKAction().RINPUT(coordinates, act_input)

            elif act_key == 'binput':

                MKAction().BINPUT(coordinates, act_input)

            elif act_key == 'rmb1':

                MKAction().RMB1(coordinates)

            elif act_key == 'bmb1':

                MKAction().BMB1(coordinates)

            else:

                print('Wrong Mouse Action Keyword !!!')

        elif act_key == 'press':

            MKAction().MB1(np.array(window_position) + np.array([15, 15]))

            time.sleep(0.5)

            MKAction().PRESS(act_input)

        elif act_key == 'mb2':

            MKAction().MB1(np.array(window_position) + np.array([15, 15]))

            time.sleep(0.5)

            MKAction().MB2()

        else:

            print('Wrong Keyboard Action Keyword !!!')

        MKAction().CURSOR_REST()

    else:

        if act_key == 'wait':

            time.sleep(act_input)

        elif act_key == 'finish':

            MKAction().BUS_STOP()

        else:

            print('Wrong System Command!!!')


def smrtr_engine(script_input):

    search = search_area(script_input)

    #coordinates_act_img(script_input, search)

    coordinates_act(script_input, search)

