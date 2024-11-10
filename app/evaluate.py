from test_model import test_model


def evaluate_markup(mp, state):
    config = {
        'swd': {
            'model_path': 'models/simple_model_swd.pth',
            'segment_length': 400,
            'step': 200,
            'partitions': 4
        },
        'ds': {
            'model_path': 'models/simple_model_ds.pth',
            'segment_length': 400 * 10,
            'step': 200 * 10,
            'partitions': 20
        },
        'is': {
            'model_path': 'models/simple_model_is.pth',
            'segment_length': 400 * 5,
            'step': 200 * 5,
            'partitions': 10
        }
    }

    result = test_model([mp.filename], config)[0]

    return result
