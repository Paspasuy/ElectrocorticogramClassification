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
        filename = app_data["file_path_name"]
        markup = []
        if filename.endswith('.edf'):
            markup.append(ECGMarkup(filename))
        else:
            for name in os.listdir(app_data["file_path_name"]):
                if name.endswith('.edf'):
                    markup.append(ECGMarkup(os.path.join(filename, name)))

        for mp in markup:
            mp.load_data_and_markup()

        state["markup"] = markup

        display_markup(state)

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
