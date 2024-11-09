import os

APP_TITLE = "Классификатор ЭКоГ"

items = []

UI_SCALE_TEXTS = [
    "50%",
    "75%",
    "100%",
    "150%",
    "200%",
]

GREEN = (50, 255, 50)
WHITE = (255, 255, 255)
RED = (255, 50, 50)
PURPLE = (255, 0, 255)
ORANGE = (255, 150, 0)

FONT_SIZE = 20
VIEWPORT_WIDTH = 720
VIEWPORT_HEIGHT = 480

FILE_DIALOG_WIDTH = 600
FILE_DIALOG_HEIGHT = 360
FILE_DIALOG_TAG = "file_dialog_tag"
FOLDER_DIALOG_TAG = "folder_dialog_tag"

DISPLAY_WINDOW_TAG = "DISPLAY_WINDOW_TAG"
DISPLAY_TAG = "DISPLAY_TAG"
DISPLAY_WIDTH = 600
DISPLAY_HEIGHT = 360
DISPLAY_TEXT = "Разметка для файла %s"

FRL_TAG = "FRL_TAG"
FRR_TAG = "FRR_TAG"
OCR_TAG = "OCR_TAG"

TEXT_POS = (10, 30)
TEXT_WRAP = 400


PREVIEW_WINDOW_TAG = "preview_window_tag"

FONT_PATH = "./app/font/OpenSans-Regular.ttf"
# FONT_PATH = "/usr/share/fonts/TTF/OpenSans-Regular.ttf"
DATA_DIR = os.getcwd()


def resize_tuple(t, k):
    return tuple(num * k for num in t)


def tr(text, locale):
    idx = {
        "ru": 0,
        "en": 1,
    }[locale]
    return text[idx]
