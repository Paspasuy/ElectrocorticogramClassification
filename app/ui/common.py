import dearpygui.dearpygui as dpg

import sys

from app.constants import *
from app import texts
from app.ui.font import get_font
from app.ui.menu import load_menu


def file_picker_callback(sender, app_data, user_data):
    global latest_filename
    latest_filename = app_data["file_name"]
    print(list(app_data["selections"].values())[0])


def delete_interface(state):
    for item in state["items"]:
        dpg.delete_item(item)
    state["items"] = []


def draw_interface(state):
    scale = state['ui_scale']

    default_font = get_font(state)

    load_menu(state, delete_interface=delete_interface, draw_interface=draw_interface)

    with dpg.file_dialog(
            directory_selector=True,
            show=False,
            callback=file_picker_callback,
            tag=FILE_DIALOG_TAG,
            width=FILE_DIALOG_WIDTH * scale,
            height=FILE_DIALOG_HEIGHT * scale,
    ) as item_id:
        state["items"].append(item_id)
        dpg.add_file_extension("", color=ORANGE)  # Directories color
        dpg.add_file_extension(".edf", color=PURPLE, custom_text="[EDF]")

    # TODO: plot

    with dpg.window(tag="Primary Window") as item_id:
        state["items"].append(item_id)
        text_tag = dpg.add_text("hi", wrap=TEXT_WRAP * scale, pos=resize_tuple(TEXT_POS, scale))

        dpg.bind_font(default_font)

    dpg.set_primary_window("Primary Window", True)
