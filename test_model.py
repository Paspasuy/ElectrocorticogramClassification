import os
import numpy as np
from src.data_loader import ECoGDataLoader
from src.feature_extractor import FeatureExtractor
from src.models.simple_model import SimpleNN
import torch
from src.visualize import plot_segment
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader
import mne
from collections import OrderedDict
from tqdm import tqdm


class TestResult:
    """
    Класс для хранения результатов тестирования.

    Attributes:
        labels (dict): Массив меток.
        file_path (str): Путь к размеченному файлу.
    """
    def __init__(self, labels: dict, file_path: str):
        self.labels = labels
        self.file_path = file_path
    
    def convert_to_edf(self, directory: str):
        """
        Преобразует предсказанные отрезки в edf файл с аннотациями.
        """
        # Читаем исходный файл
        raw = mne.io.read_raw_edf(self.file_path, preload=True)
        
        # Создаем новый файл
        output_path = os.path.join(directory, os.path.basename(self.file_path).replace('.edf', '_predicted.edf'))
        
        # Подготавливаем аннотации
        onsets = []
        durations = []
        descriptions = []
        
        for label_type, segments in self.labels.items():
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
        
        return output_path


def test_model(file_list: list,
               config: dict,
               device: str = 'cuda' if torch.cuda.is_available() else 'cpu') -> list:
    """
    Функция для тестирования модели на списке файлов.

    Args:
        file_list (list): Список путей к файлам для тестирования.
        config (dict): Конфигурация для загрузки данных и запуска моделей для каждого класса.
        device (str, optional): Устройство для вычислений. По умолчанию 'cuda' если доступно, иначе 'cpu'.

    Returns:
        list: Список объектов TestResult с метками и путями к файлам.
    """
    result = []

    bounders = {'swd': (2 * 400, 20 * 400), 'is': (0, 50 * 400), 'ds': (0, 1000 * 400)}
    
    # Загрузка и извлечение сегментов
    for file_path in file_list:
        file_result = {}
        for label_type in config.keys():
            # Инициализация загрузчика данных
            loader = ECoGDataLoader(file_path, config[label_type]['segment_length'], config[label_type]['step'], label_type, mode='test')
            
            segments, _ = loader.load_data()
            
            # Извлечение признаков
            extractor = FeatureExtractor()
            segments_features = extractor.transform(segments, partitions=config[label_type]['partitions'])
            
            # Преобразование в тензоры
            segments_tensor = torch.tensor(segments_features).to(device).float()
            test_loader = DataLoader(segments_tensor, batch_size=1, shuffle=False)
            
            # Загрузка модели
            model = SimpleNN(input_dim=segments_features.shape[1], hidden_dim=64, output_dim=1)
            model.load_state_dict(torch.load(config[label_type]['model_path'], map_location=device))
            model.to(device)
            model.eval()

            all_outputs = []
            with torch.no_grad():
                for features in tqdm(test_loader):
                    features = features.to(device).float()
                    outputs = model(features)
                    all_outputs.append(outputs.cpu().item())
            
            def moving_average(data, window_size):
                return np.convolve(data, np.ones(window_size)/window_size, mode='same')
            
            def postprocess(outputs):
                # Сначала применяем скользящее среднее
                outputs = moving_average(outputs, window_size=3)
                # Бинаризуем выход
                binary = (outputs > 0.5).astype(int)
                
                # Находим отрезки единиц
                segments = []
                start = None
                for i in range(len(binary)):
                    if binary[i] == 1 and start is None:
                        start = i
                    elif binary[i] == 0 and start is not None:
                        segments.append((start, i-1))
                        start = None
                
                if start is not None:
                    segments.append((start, len(binary)-1))
                
                segments = [(seg[0] * config[label_type]['step'] + (config[label_type]['segment_length'] - config[label_type]['step']) // 2,
                             (seg[1] + 1) * config[label_type]['step'] + (config[label_type]['segment_length'] - config[label_type]['step']) // 2) for seg in segments]
                
                segments = [seg for seg in segments if bounders[label_type][0] < seg[1] - seg[0] < bounders[label_type][1]]

                return segments
            
            all_outputs = postprocess(all_outputs)
            
            # Сохранение результатов
            file_result[label_type] = all_outputs
        
        result.append(TestResult(labels=file_result, file_path=file_path))
    
    return result

# Пример использования
if __name__ == '__main__':
    import argparse
    
    files = ['data/ECoG_fully_marked_(4+2 files, 6 h each)/Ati4x3_12m_BL_6h_fully_marked.edf']

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
    
    test_results = test_model(files, config)
    
    for result in test_results:
        print(f'Файл: {result.file_path}')
        print(f'Метки: {result.labels}') 
        result.convert_to_edf('results')
