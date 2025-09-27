import win32gui
import win32ui
from ctypes import windll
from PIL import Image
import argparse
from src.helpers import logging_utils

# Setup logging for the app once (optionally add a file)
logging_utils.setup_logging(name="zoom_app", level=10)  # DEBUG
log = logging_utils.get_logger("zoom_app")

@logging_utils.trace_calls(logger=log)
def find_window_title_contains(substring: str) -> int | None:
    substring_lower = substring.lower()
    result = {"hwnd": None}

    def enum_handler(hwnd, _):
        if not win32gui.IsWindowVisible(hwnd):
            return
        title = win32gui.GetWindowText(hwnd)
        if title and substring_lower in title.lower():
            if result["hwnd"] is None:
                result["hwnd"] = hwnd

    win32gui.EnumWindows(enum_handler, None)
    if result["hwnd"] is None:
        log.debug("No window title matched substring=%r", substring)
    else:
        log.debug("Matched hwnd=%s title=%r", result["hwnd"], win32gui.GetWindowText(result["hwnd"]))
    return result["hwnd"]

@logging_utils.trace_class(logger=log)
class Window:
    def __init__(self, args):
        window_name = {
            "Zoom": "Zoom Meeting",
            "Google Meet": "Google Meet",
        }

        # Gets the meeting title of the desired app, defaults to Zoom otherwise.
        target_title = window_name.get(args.app, "Zoom Meeting")
        

        self.hwnd = find_window_title_contains(target_title)
        if not self.hwnd:
            log.warning("No window found containing %r", target_title)
            self.image = None
            # Ensure handles are defined for safe cleanup
            self.hwndDC = None
            self.mfcDC = None
            self.saveDC = None
            self.saveBitMap = None
            self._oldBmp = None
            return

        title = win32gui.GetWindowText(self.hwnd)
        left, top, right, bot = win32gui.GetClientRect(self.hwnd)
        screen_width = (right - left) * 2
        screen_height = (bot - top) * 2

        rect = win32gui.GetWindowRect(self.hwnd)
        width = rect[2] - rect[0]
        height = rect[3] - rect[1]
        log.info("Found HWND=%s title=%r size=%dx%d", self.hwnd, title, width, height)
        log.debug("Client area scaled size=%dx%d", screen_width, screen_height)

        # Create DCs and reusable bitmap buffer
        self.hwndDC = win32gui.GetWindowDC(self.hwnd)
        self.mfcDC = win32ui.CreateDCFromHandle(self.hwndDC)
        self.saveDC = self.mfcDC.CreateCompatibleDC()

        self.saveBitMap = win32ui.CreateBitmap()
        self.saveBitMap.CreateCompatibleBitmap(self.mfcDC, screen_width, screen_height)

        # Select the bitmap into the memory DC; keep old for later restore
        self._oldBmp = self.saveDC.SelectObject(self.saveBitMap)

        self.size = (screen_width, screen_height)

    def get_image(self):
        # Render latest frame into the offscreen bitmap
        result = windll.user32.PrintWindow(self.hwnd, self.saveDC.GetSafeHdc(), 3)
        log.debug("PrintWindow result=%s", result)

        bmpinfo = self.saveBitMap.GetInfo()
        bmpstr = self.saveBitMap.GetBitmapBits(True)
        log.debug("Bitmap info: %s", bmpinfo)

        im = Image.frombuffer(
            "RGB",
            (bmpinfo["bmWidth"], bmpinfo["bmHeight"]),
            bmpstr,
            "raw",
            "BGRX",
            0,
            1,
        )

        self.image = im
        return self.image

    def close(self):
        # Proper cleanup: restore selection and delete GDI resources
        try:
            if self.saveDC and self._oldBmp:
                self.saveDC.SelectObject(self._oldBmp)
        except Exception:
            pass
        try:
            if self.saveBitMap:
                win32gui.DeleteObject(self.saveBitMap.GetHandle())
        except Exception:
            pass
        try:
            if self.saveDC:
                self.saveDC.DeleteDC()
        except Exception:
            pass
        try:
            if self.mfcDC:
                self.mfcDC.DeleteDC()
        except Exception:
            pass
        try:
            if self.hwndDC:
                win32gui.ReleaseDC(self.hwnd, self.hwndDC)
        except Exception:
            pass
        log.debug("Released GDI resources")

def main(args):
    meeting = Window(args)
    i = 0
    while True:
        i += 1
        image = meeting.get_image()
        image.save(f"tmp/frame_{i}.png")
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
                prog='Window Capture',
                description='Tests the Window class and its ability to capture images')
    parser.add_argument("--app", "-a", default="Zoom", help="Meeting application")
    args = parser.parse_args() 

    main(args)
