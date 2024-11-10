#!/usr/bin/python

import dearpygui.dearpygui as dpg

from app.constants import *
from app.ui import common


if __name__ == "__main__":
    dpg.create_context()
    dpg.setup_dearpygui()

    scale = 1
    dpg.create_viewport(title=APP_TITLE, width=VIEWPORT_WIDTH * scale, height=VIEWPORT_HEIGHT * scale)

    state = {
        "ui_scale": scale,
        "ui_scale_text": "100%",
        "locale": "ru",
        "items": [],
        "annotation": [],
        "format": OUT_FORMATS[0]
    }

    common.draw_interface(state)

    dpg.show_viewport()
    dpg.start_dearpygui()
    dpg.destroy_context()
