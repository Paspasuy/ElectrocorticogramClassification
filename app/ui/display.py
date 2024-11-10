import dearpygui.dearpygui as dpg

from app import texts
from app.constants import *

import numpy as np


def get_data(markup):
    return markup.data


def jump_to_markup_segment(state, index):
    markup = state["markup"][-1]
    data = get_data(markup)

    tags = [FRL_TAG, FRR_TAG, OCR_TAG]
    shifts = [0.001, 0, -0.001]

    segment = markup.pretty_markup[index]
    sl, sr = segment[0][0] / 400, segment[1][0] / 400

    true_sr = sr

    if sr - sl > 10:
        sr = sl + 7

    offset = (sr - sl) / 3

    slo = max(0, sl - offset)
    sro = sr + offset

    for item in state["annotation"]:
        dpg.delete_item(item)
    state["annotation"] = []

    state["annotation"].append(dpg.draw_line((sl, -0.003), (sl, 0.003), parent=DISPLAY_TAG, thickness=1/200, color=MAGENTA))
    state["annotation"].append(dpg.draw_line((true_sr, -0.003), (true_sr, 0.003), parent=DISPLAY_TAG, thickness=1/200, color=MAGENTA))

    state["annotation"].append(
        dpg.add_plot_annotation(parent=DISPLAY_TAG, label=segment[0][1], default_value=(sl, -0.0012), offset=(-15, 15), color=YELLOW)
    )
    state["annotation"].append(
        dpg.add_plot_annotation(parent=DISPLAY_TAG, label=segment[1][1], default_value=(true_sr, -0.0012), offset=(15, 15), color=YELLOW)
    )

    for i in range(3):
        dpg.set_value(tags[i], [
            np.arange(len(data[i]))[int(slo * 400): int(sro * 400)] / 400,
            data[i][int(slo * 400): int(sro * 400)] + shifts[i]
        ])
    dpg.set_axis_limits(DISPLAY_TAG+'x', slo, sro)


def display_markup(state):
    markup = state["markup"][-1]
    data = get_data(markup)
    tags = [FRL_TAG, FRR_TAG, OCR_TAG]
    shifts = [-0.001, 0, 0.001]

    if len(markup.pretty_markup) > 0:
        jump_to_markup_segment(state, 0)
    else:
        for i in range(3):
            dpg.set_value(tags[i], [np.arange(len(data[i])) / 400, data[i] + shifts[i]])

    if dpg.does_alias_exist(SEGMENT_TABLE_TAG):
        dpg.delete_item(SEGMENT_TABLE_TAG)

    with dpg.table(parent=DISPLAY_WINDOW_TAG, tag=SEGMENT_TABLE_TAG, header_row=True) as selectablerows:
        dpg.add_table_column(label=tr(texts.BEGIN, state["locale"]))
        dpg.add_table_column(label=tr(texts.END, state["locale"]))
        dpg.add_table_column(label=tr(texts.TYPE, state["locale"]))

        for i in range(len(markup.pretty_markup)):
            segment = markup.pretty_markup[i]
            sl, sr = segment[0][0] / 400, segment[1][0] / 400
            with dpg.table_row():
                dpg.add_button(label=f"{sl}", width=-1, callback=clb_selectable,
                                   user_data=(state, i))
                # dpg.add_clicked_handler
                dpg.add_button(label=f"{sr}", width=-1, callback=clb_selectable,
                                   user_data=(state, i))
                dpg.add_button(label=f"{segment[0][1][:-1]}", width=-1, callback=clb_selectable,
                                   user_data=(state, i))

    name = os.path.basename(markup.filename)
    dpg.configure_item(DISPLAY_WINDOW_TAG, show=True, label=DISPLAY_TEXT + name)


def clb_selectable(sender, app_data, user_data):
    jump_to_markup_segment(user_data[0], user_data[1])


def load_display_window(state):
    scale = state["ui_scale"]
    with dpg.window(tag=DISPLAY_WINDOW_TAG, label="Окошечко", show=False) as window_id:
        state["items"].append(window_id)

        tags = [FRL_TAG, FRR_TAG, OCR_TAG]
        # display_tags = [FRL_DISPLAY_TAG, FRR_DISPLAY_TAG, OCR_DISPLAY_TAG]
        colors = [YELLOW, MAGENTA, CYAN]

        shifts = [-0.001, 0, 0.001]

        with dpg.plot(tag=DISPLAY_TAG, height=DISPLAY_HEIGHT * scale, width=DISPLAY_WIDTH * scale):
            dpg.add_plot_legend()

            x_axis_id = dpg.add_plot_axis(dpg.mvXAxis, no_gridlines=True, tag=DISPLAY_TAG + 'x')
            y_axis_id = dpg.add_plot_axis(dpg.mvYAxis, label="mV", tag=DISPLAY_TAG + 'y')

            dpg.set_axis_zoom_constraints(x_axis_id, vmin=0.25, vmax=10)
            dpg.set_axis_zoom_constraints(y_axis_id, vmin=0.003, vmax=0.006)

            dpg.set_axis_limits_constraints(x_axis_id, vmin=0, vmax=3600 * 36)
            dpg.set_axis_limits_constraints(y_axis_id, vmin=-0.003, vmax=0.003)

            for (tag, shift, color) in zip(tags, shifts, colors):
                dpg.add_line_series(
                    [],
                    [],
                    tag=tag,
                    parent=y_axis_id,
                )
