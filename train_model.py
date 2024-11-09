import os
import numpy as np
from src.data_loader import ECoGDataLoader
from src.feature_extractor import FeatureExtractor, DummyFeatureExtractor, WaveletFeatureExtractor
from src.models.simple_model import SimpleNN
from src.models.simple_cnn_model import SimpleCNN
from src.models.wavelet_cnn_model import WaveletCNN
from src.train import train_model, get_data_loaders
import torch
from src.visualize import plot_segment
from sklearn.metrics import classification_report
import argparse
    

parser = argparse.ArgumentParser(description='Обучение и тестирование модели')
parser.add_argument('--not-train', action='store_true', help='Не запускать обучение модели', default=False)
parser.add_argument('--not-validate', action='store_true', help='Не запускать валидацию модели', default=False)

args = parser.parse_args()


# Параметры
data_dir = 'data/ECoG_fully_marked_(4+2 files, 6 h each)'
segment_length = 400 * 10
step = 200 * 10
partitions = 20
batch_size = 32
num_epochs = 40
hidden_dim = 64
device = 'cuda' if torch.cuda.is_available() else 'cpu'
save_path = 'models/simple_model_ds.pth'
label_type = 'ds'

# Загрузка данных
if not args.not_train:
    loader_train = ECoGDataLoader(data_dir, segment_length, step, label_type, mode='train')
    loader_val = ECoGDataLoader(data_dir, segment_length, step, label_type, mode='val')
if not args.not_validate:
    loader_val_full = ECoGDataLoader(data_dir, segment_length, step, label_type, mode='val_full')

if not args.not_train:
    X_train, y_train = loader_train.load_data()
    X_val, y_val = loader_val.load_data()
    print(f'Данные загружены: {X_train.shape}, {y_train.shape}, {X_val.shape}, {y_val.shape}')
if not args.not_validate:
    X_val_full, y_val_full = loader_val_full.load_data()
    print(f'Данные full загружены: {X_val_full.shape}, {y_val_full.shape}')

# Извлечение фич
extractor = DummyFeatureExtractor()
if not args.not_train:
    X_train_features = extractor.transform(X_train, partitions=partitions)
    X_val_features = extractor.transform(X_val, partitions=partitions)
    y_train = np.expand_dims(y_train, axis=1)
    y_val = np.expand_dims(y_val, axis=1)
    print(f'Фичи извлечены: X_train: {X_train_features.shape}, y_train: {y_train.shape}, X_val: {X_val_features.shape}, y_val: {y_val.shape}')
if not args.not_validate:
    X_val_full_features = extractor.transform(X_val_full, partitions=segment_length // step * 2)
    y_val_full = np.expand_dims(y_val_full, axis=1)
    print(f'Фичи извлечены: X_val_full: {X_val_full_features.shape}, y_val_full: {y_val_full.shape}')

if not args.not_train:
    # Разделение на тренир и валидацию
    train_loader, val_loader = get_data_loaders(X_train_features, y_train, X_val_features, y_val, batch_size)

# Инициализация модели
shape = X_train_features.shape[1] if not args.not_train else X_val_full_features.shape[1]
input_dim = shape
output_dim = 1
model = SimpleCNN(input_dim=input_dim, output_dim=output_dim)

def validate():
    # Загрузка лучшей модели
    model.load_state_dict(torch.load(save_path))
    model.to(device)

    # Тестирование на валидации
    model.eval()
    all_preds = []
    all_labels = []
    all_outputs = []
    with torch.no_grad():
        for features, labels in val_loader:
            features = features.to(device).float()
            labels = labels.float()
            outputs = model(features)
            preds = (outputs > 0.5).int().cpu().numpy()
            all_outputs.append(outputs.cpu().numpy())
            all_preds.append(preds)
            all_labels.append(labels.numpy())

    all_preds = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)
    all_outputs = np.vstack(all_outputs)
    
    # Получаем индексы, где метки равны 1
    positive_indices = np.where((all_labels == 1) | (all_labels == 0))[0]
    one_preds = all_preds[positive_indices]
    one_labels = all_labels[positive_indices]

    # Метрики
    report = classification_report(one_labels, one_preds)
    print(f'\nРезультаты классификации для {label_type.upper()}:')
    print(report)

    # Визуализация примера
    for idx in [-1, -2, -3, 100]:
        assert one_labels[idx] == y_val[positive_indices[idx]], f'{one_labels[idx]} != {y_val[positive_indices[idx]]}'
        plot_segment(np.linspace(0, segment_length, segment_length),
                    X_val[positive_indices[idx]],
                    one_labels[idx],
                    one_preds[idx],
                    'swd')


def train():
    # Критерий и оптимизатор
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Обучение модели
    train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device, save_path)
    validate()


def validate_full():
    pass


if __name__ == '__main__':
    if not args.not_train:
        train()
    elif not args.not_validate:
        validate_full()
