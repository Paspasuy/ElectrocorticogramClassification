import dearpygui.dearpygui as dpg

import sys

from app.constants import *
from app import texts
from app.ui.display import load_display_window
from app.ui.file_dialog import load_file_dialog, load_folder_dialog
from app.ui.font import get_font
from app.ui.menu import load_menu
from app.ui.window import load_window


def delete_interface(state):
    for item in state["items"]:
        dpg.delete_item(item)
    state["items"] = []


def draw_interface(state):
    scale = state['ui_scale']

    default_font = get_font(state)

    load_menu(state, delete_interface=delete_interface, draw_interface=draw_interface)
    load_file_dialog(state)
    load_folder_dialog(state)
    load_display_window(state)

    window = load_window(state, default_font)

    dpg.set_primary_window("Primary Window", True)
