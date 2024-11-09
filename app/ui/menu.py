import dearpygui.dearpygui as dpg

from app.constants import *
from app import texts


def get_resize_callback(new_text, reload_ui_f, state):
    def resize_callback():
        k = int(new_text[:-1]) / int(state["ui_scale_text"][:-1])
        state["ui_scale_text"] = new_text
        state["ui_scale"] *= k

        dpg.set_viewport_height(VIEWPORT_HEIGHT * state["ui_scale"])
        dpg.set_viewport_width(VIEWPORT_WIDTH * state["ui_scale"])

        reload_ui_f["delete_interface"](state)
        reload_ui_f["draw_interface"](state)

    return resize_callback


def get_ru_lang_callback(reload_ui_f, state):
    def ru_lang_callback():
        state["locale"] = 'ru'

        reload_ui_f["delete_interface"](state)
        reload_ui_f["draw_interface"](state)
    return ru_lang_callback


def get_eng_lang_callback(reload_ui_f, state):
    def eng_lang_callback():
        state["locale"] = 'en'

        reload_ui_f["delete_interface"](state)
        reload_ui_f["draw_interface"](state)
    return eng_lang_callback


def load_menu(state, **reload_ui_f):
    locale = state["locale"]
    with dpg.viewport_menu_bar() as item_id:
        state["items"].append(item_id)
        with dpg.menu(label=tr(texts.FILE, locale)):
            dpg.add_menu_item(
                label=tr(texts.OPEN, locale), callback=lambda: dpg.show_item(FILE_DIALOG_TAG)
            )
            dpg.add_menu_item(
                label=tr(texts.OPEN_FOLDER, locale), callback=lambda: dpg.show_item(FOLDER_DIALOG_TAG)
            )
            dpg.add_menu_item(label=tr(texts.EXIT, locale), callback=lambda: dpg.stop_dearpygui())
        dpg.add_menu_item(
            label="History", callback=lambda: dpg.show_item(PREVIEW_WINDOW_TAG)
        )
        with dpg.menu(label=tr(texts.VIEW, locale)):
            with dpg.menu(label=tr(texts.SCALE, locale)):
                for scale_text in UI_SCALE_TEXTS:
                    dpg.add_menu_item(
                        label=scale_text, callback=get_resize_callback(scale_text, reload_ui_f, state)
                    )
            with dpg.menu(label=tr(texts.LANG, locale)):
                dpg.add_menu_item(label="Русский", callback=get_ru_lang_callback(reload_ui_f, state))
                dpg.add_menu_item(label="English", callback=get_eng_lang_callback(reload_ui_f, state))
