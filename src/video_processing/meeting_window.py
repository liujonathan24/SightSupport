import win32gui
import win32ui
from ctypes import windll
from PIL import Image
import logging
from src.helpers import logging_utils

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s | %(levelname)s | %(name)s | %(funcName)s:%(lineno)d | %(message)s",
)

log = logging.getLogger("zoom_app")

def find_window_title_contains(substring: str) -> int | None:
    substring_lower = substring.lower()
    result = {'hwnd': None}

    def enum_handler(hwnd, _):
        if not win32gui.IsWindowVisible(hwnd):
            return
        title = win32gui.GetWindowText(hwnd)
        if title and substring_lower in title.lower():
            result['hwnd'] = hwnd
    win32gui.EnumWindows(enum_handler, None)
    return result['hwnd']

@trace_class # How do I fix this?
class Window:
    def __init__(self, args):
        window_name = {
            "Zoom": "Zoom Meeting",
            "Google Meet": "Google Meet" # Fix.
        }
        hwnd = find_window_title_contains("Zoom Meeting")

        left, top, right, bot = win32gui.GetClientRect(hwnd)
        screen_width = (right - left ) * 2
        screen_height = (bot - top) * 2

        hwnd = find_window_title_contains("Zoom Meeting")
        if hwnd:
            title = win32gui.GetWindowText(hwnd)
            rect = win32gui.GetWindowRect(hwnd)  # (left, top, right, bottom)
            width = rect[2] - rect[0]
            height = rect[3] - rect[1]
            print(f"Found HWND={hwnd}, title='{title}', size=({width}x{height})")
        else:
            print("No window found containing 'Zoom Meeting'")

        print(f"Screen width x height:{screen_width}x{screen_height}")
        hwndDC = win32gui.GetWindowDC(hwnd)
        mfcDC  = win32ui.CreateDCFromHandle(hwndDC)
        saveDC = mfcDC.CreateCompatibleDC()

        saveBitMap = win32ui.CreateBitmap()
        saveBitMap.CreateCompatibleBitmap(mfcDC, screen_width, screen_height)

        saveDC.SelectObject(saveBitMap)

        result = windll.user32.PrintWindow(hwnd, saveDC.GetSafeHdc(), 3)

        bmpinfo = saveBitMap.GetInfo()
        bmpstr = saveBitMap.GetBitmapBits(True)

        im = Image.frombuffer(
            'RGB',
            (bmpinfo['bmWidth'], bmpinfo['bmHeight']),
            bmpstr, 'raw', 'BGRX', 0, 1)

        win32gui.DeleteObject(saveBitMap.GetHandle())
        saveDC.DeleteDC()
        mfcDC.DeleteDC()
        win32gui.ReleaseDC(hwnd, hwndDC)
        

        return im