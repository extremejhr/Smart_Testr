from smrtrpy import *

para_init("config.ini")


class Workbench_Capture(object):

    def __init__(self, title):

        self.title = title

        self.handle = None

    def handle_catch(self, hwnd, lParam):

        if win32gui.IsWindowVisible(hwnd):

            if self.title in win32gui.GetWindowText(hwnd):

                self.handle = hwnd

    def icapture(self, position, size, margin, scale):

        x_axis = 1*(size[0] - 2 * margin)

        y_axis = 1*(size[1] - 2 * margin)

        screen_shot = [[np.array([int(scale*x_axis), int(scale*y_axis), position[0], position[1], 0])], ]

        win32gui.EnumWindows(self.handle_catch, None)

        if self.handle is None:

            print('No Main Window Found!')

        else:

            win32gui.SetForegroundWindow(self.handle)

            win32gui.SetWindowPos(self.handle, win32con.HWND_TOPMOST, position[0], position[1], size[0], size[1],
                                 win32con.SWP_SHOWWINDOW)

            time.sleep(0.5)

            img = np.array(ImageGrab.grab((position[0]+margin, position[1]+margin, position[0]+size[0]-margin,
                                           position[1]+size[1]-margin)))

            img = cv2.resize(img, (int(scale*(size[0]-2*margin)), int(scale*(size[1]-2*margin))))

            gray_mod = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            screen_shot[0].append(np.array(gray_mod).ravel())

            cv2.imwrite('running_temp_file\\' + 'workbench.png', img)

        return [screen_shot]


def icon_create(title, window_position, window_size, margin, scale, gaussian_blur, depth):

    test_window = Workbench_Capture(title)

    region = test_window.icapture(window_position, window_size, margin, scale)

    tree = ImgTree(gaussian_blur)

    temp = region[0]

    RD4 = tree.next_depth([temp], depth)

    tree.img_save(RD4, 'tmp_icon_data\\', 'ALL')


def icon_update(title, window_position, window_size, margin, scale, gaussian_blur, depth):

    test_window = Workbench_Capture(title)

    region = test_window.icapture(window_position, window_size, margin, scale)

    tree = ImgTree(gaussian_blur)

    temp = region[0]

    RD4 = tree.next_depth([temp], depth)

    tree.img_save(RD4, 'icon_data1\\', 'ALL')


if __name__ == '__main__':

    title = 'Simcenter 3D 1899.900'

    window_position = np.array([0, 0])

    window_size = np.array([1900, 1000])

    margin = 10

    scale = 2

    gaussian_blur = (3, 3)

    depth = 9

    icon_create(title, window_position, window_size, margin, scale, gaussian_blur, depth)

