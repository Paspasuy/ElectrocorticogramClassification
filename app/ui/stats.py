from pathlib import Path

import dearpygui.dearpygui as dpg

from app.constants import *
from app import texts
from app.evaluate import evaluate_markup
from app.markup import ECGMarkup
from app.ui.display import display_markup


def norm(val):
    sm = sum(val)
    return [i / sm for i in val]


def calc_stats(state, markups: list[ECGMarkup]):
    scale = state["ui_scale"]
    locale = state["locale"]
    dpg.delete_item(STATS_WINDOW_TAG)
    with dpg.window(tag=STATS_WINDOW_TAG, label=tr(texts.STATS, locale), show=False) as window_id:
        sm = {key: sum(len(mp.markup.labels[key]) for mp in markups) for key in TYPES}
        avg = {key: sum(sum((s[1] - s[0]) / 400 for s in mp.markup.labels[key]) for mp in markups) / sm[key] for key in TYPES}
        # avg = {key: sum(mp.markup.labels for mp in markups) for key in TYPES}
        print(avg)
        dpg.add_text(f"{tr(texts.COUNT, locale)}, {tr(texts.AVG, locale)}")
        with dpg.plot(tag=STATS_TAG, height=DISPLAY_PLOT_SIDE * scale, width=DISPLAY_PLOT_SIDE * 2 * scale, parent=STATS_WINDOW_TAG, no_mouse_pos=True):
            dpg.add_plot_legend()
            x_axis_id = dpg.add_plot_axis(dpg.mvXAxis, no_gridlines=True, tag=STATS_TAG + 'x', no_tick_marks=True, no_tick_labels=True)
            y_axis_id = dpg.add_plot_axis(dpg.mvYAxis, tag=STATS_TAG + 'y', no_gridlines=True, no_tick_marks=True, no_tick_labels=True)


            dpg.add_pie_series(x=1., y=0., radius=1., values=list(sm.values()), labels=list(sm.keys()),
                               tag=STATS_COUNT_TAG, label=tr(texts.COUNT, locale), parent=y_axis_id)

            dpg.add_pie_series(x=3., y=0., radius=1., values=list(avg.values()), labels=list(avg.keys()),
                               tag=STATS_AVG_TAG, label=tr(texts.AVG, locale), parent=y_axis_id)

            dpg.set_axis_limits(x_axis_id, 0., 4.)
            dpg.set_axis_limits(y_axis_id, -1., 1.)

        dpg.configure_item(STATS_WINDOW_TAG, show=True)


def load_stats(state):
    scale = state["ui_scale"]
    locale = state["locale"]
    with dpg.window(tag=STATS_WINDOW_TAG, label=tr(texts.STATS, locale), show=False) as window_id:
        state["items"].append(window_id)
        # dpg.add_pie_series([], [], radius=PIE_RADIUS * scale, tag=STATS_AVG_TAG)
