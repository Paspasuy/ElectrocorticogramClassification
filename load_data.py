import mne
import numpy as np
from typing import Tuple

def get_data_and_annotations(raw: mne.io.Raw) -> Tuple[np.ndarray, dict]:
    """Get annotations from raw data.

    Args:
        raw (mne.io.Raw): Raw data.

    Returns:
        dict: contains swd, ds, is event lists
        Each event is described by a tuple (onset, end) (indices, not times)
    """
    
    data = raw.get_data()
    annotations = raw.annotations
        
    description_map = {
        'sdw1': 'swd1', 'sdw2': 'swd2',
        'dd1': 'ds1', 'dd2': 'ds2',
        'dds1': 'ds1', 'dds2': 'ds2',
        'swd1': 'swd1', 'swd2': 'swd2',
        'is1': 'is1', 'is2': 'is2', 
        'ds1': 'ds1', 'ds2': 'ds2'
    }
    file_segments = {
        'swd1': [], 'swd2': [], 
        'is1': [], 'is2': [],
        'ds1': [], 'ds2': []
    }

    annotations_fixed = []
    current_event = None
    for annotation in annotations:
        onset = annotation['onset']
        description = annotation['description']
        if description not in description_map:
            raise ValueError(f"Неизвестный тип события: {description}")
        
        mapped_description = description_map[description]
        event_base = mapped_description[:-1]  # убираем цифру в конце
        event_num = int(mapped_description[-1])  # получаем цифру
        annotations_fixed.append((mapped_description, onset))
        
        # Проверяем правильность последовательности (1->2)
        if event_num == 1:
            if current_event == event_base:
                mapped_description = mapped_description[:-1] + '2'
                event_num = 2
            else:
                if current_event is not None:
                    file_segments[current_event + '1'].pop()
                current_event = event_base
        if event_num == 2:
            if current_event is None and annotations_fixed == 'swd':
                file_segments[event_base + '2'].pop()
            elif current_event != event_base:
                continue
            current_event = None
            
        file_segments[mapped_description].append(onset)
    
    # Проверяем, что последнее событие закрыто
    if current_event is not None:
        raise ValueError(f"Ошибка: событие {current_event} не закрыто в конце файла")
    
    event_dict = {'swd': [], 'is': [], 'ds': []}

    # Для каждого типа события в текущем файле
    for event_type in ['swd', 'is', 'ds']:
        # Находим все последовательные сегменты
        for i in range(len(file_segments[event_type + '1'])):
            onset = file_segments[event_type + '1'][i]
            end = file_segments[event_type + '2'][i]
            if end < onset:
                raise ValueError(f"Ошибка: событие {event_type} заканчивается раньше, чем начинается")
                        
            event_dict[event_type].append((int(onset * raw.info['sfreq']), int(end * raw.info['sfreq'])))
            
    return raw.get_data(), event_dict


if __name__ == '__main__':
    raw = mne.io.read_raw_edf('data/ECoG_fully_marked_(4+2 files, 6 h each)/Ati4x3_9m_Xyl01(Pharm!)_6h_fully_marked.edf', preload=True)
    data, annotations = get_data_and_annotations(raw)
    print(annotations)
