import dearpygui.dearpygui as dpg

from app.constants import *
from app import texts
from app.markup import ECGMarkup
from app.ui.display import display_markup


def get_format_callback(state):
    def format_callback(sender, app_data, user_data):
        state["format"] = app_data

    return format_callback


def get_go_callback(state):
    def go_callback(sender, app_data, user_data):
        filename = state["filename"]
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
            mp.load_data_and_markup()

        dpg.set_value(STATUS_TAG, tr(texts.READY, state["locale"]))
        dpg.configure_item(STATUS_TAG, color=GREEN)

        state["markup"] = markup

        display_markup(state)
    return go_callback


def load_window(state, font):
    scale = state["ui_scale"]
    locale = state["locale"]
    with dpg.window(tag="Primary Window") as item_id:
        state["items"].append(item_id)
        dpg.add_spacer(height=int(20 * scale))
        dpg.add_text("Файл не выбран!", tag=FILENAME_TAG, wrap=TEXT_WRAP * scale)
        dpg.add_radio_button(OUT_FORMATS, callback=get_format_callback(state))
        dpg.add_button(label=tr(texts.GO, locale), callback=get_go_callback(state))
        dpg.add_text(tr(texts.READY, locale), tag=STATUS_TAG, pos=resize_tuple(STATUS_POS, scale), color=GREEN)

        dpg.bind_font(font)
