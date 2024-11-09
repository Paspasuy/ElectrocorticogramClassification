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
            mp.load_data_and_markup()

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

        dpg.bind_font(font)
