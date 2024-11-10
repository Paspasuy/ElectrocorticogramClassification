import dearpygui.dearpygui as dpg
import sys

from app.constants import *
from app import texts
from app.markup import ECGMarkup
from app.ui.display import display_markup
from app.ui.font import get_font
from app.ui.menu import load_menu
import numpy as np


def get_file_picker_callback(state):
    def file_picker_callback(sender, app_data, user_data):
        state["filename"] = app_data["file_path_name"]
        dpg.set_value(FILENAME_TAG, os.path.basename(state["filename"]))

    return file_picker_callback


def load_file_dialog(state):
    scale = state["ui_scale"]
    with dpg.file_dialog(
            directory_selector=False,
            show=False,
            callback=get_file_picker_callback(state),
            tag=FILE_DIALOG_TAG,
            width=FILE_DIALOG_WIDTH * scale,
            height=FILE_DIALOG_HEIGHT * scale,
            default_path='..',
    ) as item_id:
        state["items"].append(item_id)
        dpg.add_file_extension(".edf", color=PURPLE, custom_text="[EDF]")


def load_folder_dialog(state):
    scale = state["ui_scale"]
    with dpg.file_dialog(
            directory_selector=True,
            show=False,
            callback=get_file_picker_callback(state),
            tag=FOLDER_DIALOG_TAG,
            width=FILE_DIALOG_WIDTH * scale,
            height=FILE_DIALOG_HEIGHT * scale,
            default_path='..',
    ) as item_id:
        state["items"].append(item_id)
        dpg.add_file_extension("", color=ORANGE)  # Directories color
