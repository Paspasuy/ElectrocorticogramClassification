from pathlib import Path

import dearpygui.dearpygui as dpg

from app.constants import *
from app import texts
from app.evaluate import evaluate_markup
from app.markup import ECGMarkup
from app.ui.display import display_markup
from app.ui.stats import calc_stats


def get_format_callback(state):
    def format_callback(sender, app_data, user_data):
        state["format"] = app_data

    return format_callback


def get_go_callback(state):
    def go_callback(sender, app_data, user_data):
        filename = state["filename"]

        state["output_dir"] = dpg.get_value(OUTPUT_DIR)
        Path(state["output_dir"]).mkdir(parents=True, exist_ok=True)

        markup = []
        if filename.endswith('.edf'):
            markup.append(ECGMarkup(filename))
        else:
            for name in os.listdir(filename):
                if name.endswith('.edf'):
                    markup.append(ECGMarkup(os.path.join(filename, name)))

        for mp in markup:
            dpg.set_value(STATUS_TAG, tr(texts.PROCESSING, state["locale"]) % os.path.basename(mp.filename))
            dpg.configure_item(STATUS_TAG, color=ORANGE)

            # High load part
            mp.load_data()
            result = evaluate_markup(mp, state)
            mp.set_markup(result)
            mp.save(state)

        dpg.set_value(STATUS_TAG, tr(texts.READY, state["locale"]))
        dpg.configure_item(STATUS_TAG, color=GREEN)
        state["markup"] = markup
        display_markup(state)
        calc_stats(state, markup)
    return go_callback


def load_window(state, font):
    scale = state["ui_scale"]
    locale = state["locale"]
    with dpg.window(tag="Primary Window") as item_id:
        state["items"].append(item_id)
        dpg.add_spacer(height=int(20 * scale))
        dpg.add_text(tr(texts.NOT_CHOSEN, locale), tag=FILENAME_TAG, wrap=TEXT_WRAP * scale)
        dpg.add_radio_button(OUT_FORMATS, callback=get_format_callback(state))
        dpg.add_input_text(hint=tr(texts.ENTER_OUTPUT, locale), tag=OUTPUT_DIR)
        dpg.add_button(label=tr(texts.GO, locale), callback=get_go_callback(state))
        dpg.add_text(tr(texts.READY, locale), tag=STATUS_TAG, pos=resize_tuple(STATUS_POS, scale), color=GREEN)

        dpg.bind_font(font)
