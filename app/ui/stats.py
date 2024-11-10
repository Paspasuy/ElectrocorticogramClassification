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
        avg = {key: sum(sum((s[1] - s[0]) / 400 for s in mp.markup.labels[key]) for mp in markups) / (0.0000001 + sm[key]) for key in TYPES}

        with dpg.table(tag=STATS_TAG, header_row=True):
            dpg.add_table_column()
            for key in TYPES:
                dpg.add_table_column(label=key)

            for data, text in zip([sm, avg], [texts.COUNT, texts.AVG]):
                with dpg.table_row():
                    dpg.add_text(tr(text, locale))
                    for key in TYPES:
                        dpg.add_text(f"{round(data[key], 3)}")

        dpg.configure_item(STATS_WINDOW_TAG, show=True)


def load_stats(state):
    scale = state["ui_scale"]
    locale = state["locale"]
    with dpg.window(tag=STATS_WINDOW_TAG, label=tr(texts.STATS, locale), show=False) as window_id:
        state["items"].append(window_id)
        # dpg.add_pie_series([], [], radius=PIE_RADIUS * scale, tag=STATS_AVG_TAG)
