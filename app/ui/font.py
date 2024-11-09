import dearpygui.dearpygui as dpg

import sys

from app.constants import *


def get_windows_font_path():
    import os
    import ctypes
    import ctypes.wintypes

    # Buffer for storing font folder path
    buffer = ctypes.create_unicode_buffer(ctypes.wintypes.MAX_PATH)

    # Load the system directory path using Windows API
    if ctypes.windll.shlwapi.SHGetFolderPathW(None, 0x0014, None, 0, buffer) == 0:
        font_folder_path = os.path.join(buffer.value, "Fonts")
        default_font_path = os.path.join(font_folder_path, "segoeui.ttf")

        # Check if Segoe UI font exists, fallback to another common font if needed
        if os.path.isfile(default_font_path):
            return default_font_path
        else:
            # Alternative fallback font (Arial is commonly available)
            arial_font_path = os.path.join(font_folder_path, "arial.ttf")
            if os.path.isfile(arial_font_path):
                return arial_font_path
            raise RuntimeError
    else:
        raise RuntimeError


def get_font(state):
    scale = state["ui_scale"]
    try:
        with dpg.font_registry():
            # if 'linux' in sys.platform:
            font_path = FONT_PATH
            with dpg.font(font_path, FONT_SIZE * 2, default_font=True) as default_font:
                dpg.add_font_range_hint(dpg.mvFontRangeHint_Cyrillic)

        dpg.set_global_font_scale(scale / 2)
    except:
        dummy_text = dpg.add_window()
        default_font = dpg.get_item_font(dummy_text)
        dpg.set_global_font_scale(4)
        dpg.delete_item(dummy_text)
        state['locale'] = 'en'
        dpg.set_global_font_scale(scale * 2)

    return default_font
