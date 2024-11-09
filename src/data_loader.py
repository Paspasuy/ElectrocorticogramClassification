import os
import numpy as np
import mne
from typing import List, Tuple, Optional
from tqdm import tqdm
from load_data import get_data_and_annotations


class ECoGDataLoader:
    """Класс для загрузки и обработки данных электрокортикограммы (ECoG).
    
    Attributes:
        path (str): Путь к директории с данными
        segment_length (int): Длина сегмента в отсчетах
        step (int): Шаг между сегментами в отсчетах
        label_type (str): Тип меток для классификации ('swd', 'is' или 'ds')
        mode (str): Режим работы ('train', 'val', 'val_full' или 'test')
    """

    def __init__(self, path: str, segment_length: int, step: int, label_type: str, mode: str = 'train'):
        """Инициализация загрузчика данных.

        Args:
            path (str): Путь к директории с данными
            segment_length (int): Длина сегмента в отсчетах
            step (int): Шаг между сегментами в отсчетах
            label_type (str): Тип меток для классификации
            mode (str): Режим работы ('train', 'val', 'val_full' или 'test')
        """
        self.path = path
        self.segment_length = segment_length
        self.step = step
        self.label_type = label_type
        self.mode = mode

    def load_files(self) -> List[str]:
        """Загружает список файлов из директории для трейна и валидации или один файл для теста.

        Returns:
            List[str]: Список путей к файлам с пометкой 'fully_marked'
        """
        if self.mode == 'test':
            return [self.path]
        
        all_files = [os.path.join(self.path, f) for f in os.listdir(self.path) if 'fully_marked' in f]
        val_file = [f for f in all_files if 'Ati4x3_12m_BL_6h_fully_marked.edf' in f]
        train_files = [f for f in all_files if 'Ati4x3_12m_BL_6h_fully_marked.edf' not in f]
        
        if self.mode == 'train':
            return train_files
        else:
            return val_file

    def extract_segments(self, data: np.ndarray, annotations: Optional[dict] = None) -> List[Tuple[np.ndarray, float]]:
        """Извлекает сегменты данных и их метки из сигнала.

        Args:
            data (np.ndarray): Массив данных ECoG
            annotations (dict, optional): Словарь с аннотациями

        Returns:
            List[Tuple[np.ndarray, float]]: Список кортежей (сегмент, метка)
        """
        if self.mode == 'test':
            segments = []
            num_points = data.shape[1]
            for start in tqdm(range(0, num_points - self.segment_length + 1, self.step)):
                end = start + self.segment_length
                segment = data[:, start:end]
                segments.append((segment, -1))  # -1 означает отсутствие метки
            return segments
            
        segments = []
        num_points = data.shape[1]
        
        positive_segments = []
        negative_starts = []
        
        for start in tqdm(range(0, num_points - self.segment_length + 1, self.step)):
            end = start + self.segment_length
            segment = data[:, start:end]
            middle = start + self.segment_length // 2
            label = self.get_label(middle - self.step // 2, middle + self.step // 2, annotations)
            if self.mode != 'val_full':
                if label > 0:
                    positive_segments.append((segment, label))
                elif label == 0:
                    negative_starts.append(start)
            else:
                segments.append((segment, label))
                
        if self.mode != 'val_full':
            num_negative = min(len(negative_starts), 2 * len(positive_segments))
            np.random.seed(42)
            selected_negative_starts = np.random.choice(negative_starts, size=num_negative, replace=False)
        
            segments = [(data[:, start:start + self.segment_length], 0) for start in selected_negative_starts]
            segments.extend(positive_segments)
                
        return segments

    def get_label(self, start: int, end: int, annotations: dict) -> float:
        """Определяет метку для сегмента на основе его пересечения с событиями.

        Args:
            start (int): Начало сегмента
            end (int): Конец сегмента
            annotations (dict): Словарь с временными метками событий

        Returns:
            float: Метка сегмента (0 или значение > 0)
        """
        label = 0        
        for s, e in annotations[self.label_type]:
            if not (end <= s or start >= e):
                label = (min(e, end) - max(s, start)) / self.step
                break
                    
        return label

    def normalize_data_and_annotations(self, data: np.ndarray, annotations: Optional[dict] = None) -> Tuple[np.ndarray, Optional[dict]]:
        """Нормализует данные по каналам и удаляет выбросы.

        Args:
            data (np.ndarray): Исходные данные
            annotations (dict, optional): Исходные аннотации
        Returns:
            Tuple[np.ndarray, Optional[dict]]: Нормализованные данные и аннотации
        """
        for i in range(data.shape[0]):
            channel = data[i, :]
            channel_min = np.min(channel)
            channel_mean = np.mean(channel)
            data[i, :] = (channel - channel_min) / (2 * channel_mean)
        
        if self.mode == 'test' or annotations is None:
            return data, None
            
        bounders = {'swd': (2 * 400, 20 * 400), 'is': (0, 50 * 400), 'ds': (0, 1000 * 400)}
        
        new_annotations = {}
        for key, value in annotations.items():
            new_value = []
            for s, e in value:
                duration = e - s
                if duration > bounders[key][0] and duration < bounders[key][1]:
                    new_value.append((s, e))
            new_annotations[key] = new_value

        return data, new_annotations

    def load_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Загружает и обрабатывает все данные.
        В случае теста вместо меток возвращает -1

        Returns:
            Tuple[np.ndarray, np.ndarray]: Кортеж (X, y), где X - данные, y - метки
        """
        all_segments = []
        files = self.load_files()
        for file_path in files:
            raw = mne.io.read_raw_edf(file_path, preload=True)
            if self.mode == 'test':
                data = raw.get_data()
                data, _ = self.normalize_data_and_annotations(data)
                segments = self.extract_segments(data)
            else:
                data, annotations = get_data_and_annotations(raw)
                data, annotations = self.normalize_data_and_annotations(data, annotations)
                segments = self.extract_segments(data, annotations)
            all_segments.extend(segments)
            
        X = np.array([seg[0] for seg in all_segments])
        y = np.array([seg[1] for seg in all_segments])
        return X, y
