import mne

from load_data import get_data_and_annotations


def change_markup_format(markup):
    result = []
    for tp in ['is', 'ds', 'swd']:
        result += [((pair[0], tp+'1'), (pair[1], tp+'2')) for pair in markup[tp]]
    result.sort()
    return result


class ECGMarkup:
    def __init__(self, filename):
        self.markup = None
        self.filename = filename
        self.data = None

    def load_data(self):
        raw = mne.io.read_raw_edf(
            self.filename, preload=True)
        self.data, _ = get_data_and_annotations(raw)

    def load_data_and_markup(self):
        raw = mne.io.read_raw_edf(
            self.filename, preload=True)
        self.data, markup = get_data_and_annotations(raw)
        self.markup = change_markup_format(markup)

    def set_markup(self, markup):
        self.markup = change_markup_format(markup)
