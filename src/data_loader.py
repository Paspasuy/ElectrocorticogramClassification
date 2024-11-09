import os
import numpy as np
import mne
from typing import List, Tuple, Optional
from tqdm import tqdm


class ECoGDataLoader:
    def __init__(self, data_dir: str, segment_length: int, step: int, label_type: str):
        self.data_dir = data_dir
        self.segment_length = segment_length
        self.step = step
        self.label_type = label_type
        self.description_map = {
            'sdw1': 'swd1', 'sdw2': 'swd2',
            'dd1': 'ds1', 'dd2': 'ds2', 
            'dds1': 'ds1', 'dds2': 'ds2',
            'swd1': 'swd1', 'swd2': 'swd2',
            'is1': 'is1', 'is2': 'is2',
            'ds1': 'ds1', 'ds2': 'ds2'
        }

    def load_files(self) -> List[str]:
        files = [os.path.join(self.data_dir, f) for f in os.listdir(self.data_dir) if 'fully_marked' in f]
        return files

    def extract_segments(self, data: np.ndarray, annotations: List[dict]) -> List[Tuple[np.ndarray, dict]]:
        segments = []
        n = self.segment_length
        k = self.step
        num_points = data.shape[1]
        
        # Собираем временные сегменты для каждого типа события
        file_segments = {
            'swd1': [], 'swd2': [],
            'is1': [], 'is2': [],
            'ds1': [], 'ds2': []
        }
        
        # Обработка аннотаций
        current_event = None
        for annotation in annotations:
            onset = int(annotation['onset'] * annotation['sfreq'])
            description = annotation['description']
            
            if description not in self.description_map:
                raise ValueError(f"Description {description} not in description_map")
                
            mapped_description = self.description_map[description]
            event_base = mapped_description[:-1]
            event_num = int(mapped_description[-1])
            
            if event_num == 1:
                if current_event == event_base:
                    mapped_description = mapped_description[:-1] + '2'
                    event_num = 2
                else:
                    if current_event is not None:
                        file_segments[current_event + '1'].pop()
                    current_event = event_base
            if event_num == 2:
                if current_event is None and event_base == 'swd':
                    continue
                elif current_event != event_base:
                    continue
                current_event = None
                
            file_segments[mapped_description].append(onset)
        
        # Создание сегментов с метками
        # Собираем все сегменты в разные списки
        positive_segments = []
        negative_starts = []
        
        for start in tqdm(range(0, num_points - self.segment_length + 1, self.step)):
            end = start + self.segment_length
            segment = data[:, start:end]
            label = self.get_label(start + self.segment_length // 4, end - self.segment_length // 4, file_segments)
            if label == 1:
                positive_segments.append((segment, label))
            elif label == 0:
                negative_starts.append(start)
                
        # Выбираем случайно в 2 раза больше отрицательных сегментов
        num_negative = min(len(negative_starts), 2 * len(positive_segments))
        selected_negative_starts = np.random.choice(negative_starts, size=num_negative, replace=False)
        
        # Формируем итоговый список сегментов
        segments = [(data[:, start:start + self.segment_length], 0) for start in selected_negative_starts]
        segments.extend(positive_segments)
                
        return segments

    def get_label(self, start: int, end: int, file_segments: dict) -> Optional[dict]:
        label = 0
        # Проверяем пары начало-конец для каждого типа события
        starts = file_segments[f'{self.label_type}1']
        ends = file_segments[f'{self.label_type}2']
        
        for s, e in zip(starts, ends):
            # Если есть пересечение сегмента с событием
            if not (end <= s or start >= e):
                label = 1 if (min(e, end) - max(s, start)) / (self.segment_length // 2) > 0.5 else 0
                break
                    
        return label

    def load_data(self) -> Tuple[np.ndarray, np.ndarray]:
        all_segments = []
        files = self.load_files()
        for file_path in files:
            raw = mne.io.read_raw_edf(file_path, preload=True)
            data = raw.get_data()
            annotations = [
                {
                    'description': a['description'],
                    'onset': a['onset'],
                    'sfreq': raw.info['sfreq']
                }
                for a in raw.annotations
            ]
            segments = self.extract_segments(data, annotations)
            all_segments.extend(segments)
            
        X = np.array([seg[0] for seg in all_segments])
        y = np.array([seg[1] for seg in all_segments])
        return X, y