import os
import csv
import mne
import numpy as np

from app.constants import COLUMNS, OUT_FORMATS, TYPES
from load_data import get_data_and_annotations
from test_model import TestResult


def change_markup_format(markup):
    result = []
    for tp in TYPES:
        result += [((pair[0], tp+'1'), (pair[1], tp+'2')) for pair in markup[tp]]
    result.sort()
    return result


class ECGMarkup:
    def __init__(self, filename):
        self.markup = None
        self.pretty_markup = None
        self.filename = filename
        self.data = None
        self.raw = None

    def load_data(self):
        self.raw = mne.io.read_raw_edf(
            self.filename, preload=True)
        self.data, _ = get_data_and_annotations(self.raw)

    def load_data_and_markup(self):
        self.raw = mne.io.read_raw_edf(
            self.filename, preload=True)
        self.data, self.markup = get_data_and_annotations(self.raw)
        self.pretty_markup = change_markup_format(self.markup)

    def set_markup(self, markup: TestResult):
        self.markup = markup
        self.pretty_markup = change_markup_format(markup.labels)

    def _save_to_csv(self, state):
        name = os.path.basename(self.filename).replace('.edf', '.csv')
        dir = state["output_dir"]
        path = os.path.join(dir, name)

        with open(path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=' ', quotechar='"', quoting=csv.QUOTE_ALL)
            writer.writerow(COLUMNS)
            for event in self.pretty_markup:
                writer.writerow([event[0][0], event[1][0], event[0][1][:-1]])

    def _save_to_edf(self, state):
        name = os.path.basename(self.filename)
        dir = state["output_dir"]
        path = os.path.join(dir, name)
        _save_to_edf(self.markup, self.raw, path)

    def _save_inplace(self, state):
        _save_to_edf(self.markup, self.raw, self.filename)
        pass

    def save(self, state):
        idx = OUT_FORMATS.index(state["format"])
        if idx == 0:
            self._save_to_edf(state)
        elif idx == 1:
            self._save_inplace(state)
        elif idx == 2:
            self._save_to_csv(state)


def _save_to_edf(markup, raw, output_path):
    """
    Преобразует предсказанные отрезки в edf файл с аннотациями.
    """

    # Подготавливаем аннотации
    onsets = []
    durations = []
    descriptions = []

    for label_type, segments in markup.labels.items():
        for start, end in segments:
            # Конвертируем отсчеты в секунды (частота дискретизации 400 Гц)
            onset1 = start / 400.0
            onset2 = end / 400.0

            onsets.extend([onset1, onset2])
            durations.extend([0.0, 0.0])
            descriptions.extend([label_type + '1', label_type + '2'])

    # Сортируем аннотации по времени начала
    sorted_indices = np.argsort(onsets)
    onsets = np.array(onsets)[sorted_indices]
    durations = np.array(durations)[sorted_indices]
    descriptions = np.array(descriptions)[sorted_indices]

    # Создаем объект аннотаций
    annotations = mne.Annotations(onset=onsets,
                                  duration=durations,
                                  description=descriptions)

    # Добавляем аннотации к сырым данным
    raw.set_annotations(annotations)

    # Сохраняем файл
    raw.export(output_path, overwrite=True)
