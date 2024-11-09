import os
import numpy as np
from src.data_loader import ECoGDataLoader
from src.feature_extractor import FeatureExtractor
from src.model import SimpleNN
from src.train import train_model, get_data_loaders
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score
from src.visualize import plot_segment


def main():
    # Параметры
    data_dir = 'data/ECoG_fully_marked_(4+2 files, 6 h each)'
    segment_length = 400 * 3  # пример длины сегмента
    step = 400  # пример шага
    batch_size = 32
    num_epochs = 20
    hidden_dim = 64
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    save_path = 'model.pth'
    label_type = 'swd'

    # Загрузка данных
    loader = ECoGDataLoader(data_dir, segment_length, step, label_type)
    X, y = loader.load_data()
    print(f'Данные загружены: {X.shape}, {y.shape}')

    # Извлечение фич
    extractor = FeatureExtractor()
    X_features = extractor.transform(X)
    y = np.expand_dims(y, axis=1)
    print(f'Фичи извлечены: X: {X_features.shape}, y: {y.shape}')

    # Разделение на тренир и валидацию
    train_loader, val_loader = get_data_loaders(X_features, y, batch_size)

    # Инициализация модели
    input_dim = X_features.shape[1]
    output_dim = 1
    model = SimpleNN(input_dim, hidden_dim, output_dim)

    # Критерий и оптимизатор
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Обучение модели
    train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device, save_path)

    # Загрузка лучшей модели
    model.load_state_dict(torch.load(save_path))
    model.to(device)

    # Тестирование на валидации
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for features, labels in val_loader:
            features = features.to(device).float()
            labels = labels.to(device).float()
            outputs = model(features)
            preds = (outputs > 0.5).int().cpu().numpy()
            all_preds.append(preds)
            all_labels.append(labels.cpu().numpy())

    all_preds = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)

    # Метрики
    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds)
    rec = recall_score(all_labels, all_preds)
    print(f'{label_type.upper()} - accuracy: {acc:.2f}, precision: {prec:.2f}, recall: {rec:.2f}')

    # Визуализация примера
    window_size = 100
    # plot_segment(np.linspace(0, segment_length, segment_length),
    #             X[0],
    #             np.var(X[0, :window_size]),
    #             y[0],
    #             'swd')


if __name__ == '__main__':
    main()