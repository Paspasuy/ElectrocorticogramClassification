from test_model import test_model


def evaluate_markup(mp, state):
    config = {
        'swd': {
            'model_path': 'models/swd_new.pth',
            'segment_length': 400,
            'step': 100,
            'partitions': 1
        },
        'is': {
            'model_path': 'models/is_new.pth',
            'segment_length': 400 * 5,
            'step': 100 * 5,
            'partitions': 1
        },
        'ds': {
            'model_path': 'models/ds_new.pth',
            'segment_length': 400 * 10,
            'step': 100 * 10,
            'partitions': 1
        },
    }

    result = test_model([mp.filename], config)[0]

    return result
