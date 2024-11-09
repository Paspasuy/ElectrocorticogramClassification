import dearpygui.dearpygui as dpg
import sys

from app.constants import *
from app import texts
from app.ui.font import get_font
from app.ui.menu import load_menu
import numpy as np


def get_data(markup):
    print(f"Get data for markup {markup.filename}")
    return (
        np.random.rand(400*60),
        np.random.rand(400*60),
        np.random.rand(400*60)
    )


def display_markup(state):
    data = get_data(state["markup"][-1])
    dpg.set_value(FRL_TAG, [np.arange(len(data[0])) * 10, data[0]])
    dpg.configure_item(DISPLAY_WINDOW_TAG, show=True)
    pass


def load_display_window(state):
    scale = state["ui_scale"]
    with dpg.window(tag=DISPLAY_WINDOW_TAG, label="Окошечко", show=False) as window_id:
        state["items"].append(window_id)
        with dpg.plot(tag=DISPLAY_TAG, label=DISPLAY_TEXT, height=DISPLAY_HEIGHT * scale, width=DISPLAY_WIDTH * scale):
            dpg.add_plot_legend()

            x_axis_tag = dpg.add_plot_axis(dpg.mvXAxis, no_gridlines=True)
            y_axis_tag = dpg.add_plot_axis(dpg.mvYAxis, label="mV")

            dpg.set_axis_zoom_constraints(x_axis_tag, vmin=400, vmax=10000)
            dpg.set_axis_zoom_constraints(y_axis_tag, vmin=0.01, vmax=5)

            dpg.set_axis_limits_constraints(x_axis_tag, vmin=0, vmax=400*3600*6)


            dpg.add_line_series(
                np.arange(400*60) * 10,
                -1 / (1 + np.arange(400*60)),
                tag=FRL_TAG,
                parent=y_axis_tag,
            )
            # dpg.add_line_series(
            #     np.arange(400*60),
            #     1 / np.arange(400*60),
            #     tag=FRR_TAG,
            #     parent=y_axis_tag,
            # )
            #
            # dpg.add_line_series(
            #     np.arange(400*60),
            #     1 / np.arange(400*60),
            #     tag=OCR_TAG,
            #     parent=y_axis_tag,
            # )
            #
