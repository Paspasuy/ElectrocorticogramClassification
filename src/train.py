import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import os

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs: int, device: str, save_path: str):
    model.to(device)
    best_val_loss = float('inf')
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for features, labels in train_loader:
            features = features.to(device).float()
            labels = labels.to(device).float()
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * features.size(0)
            
        epoch_loss = running_loss / len(train_loader.dataset)
        val_loss = evaluate(model, val_loader, criterion, device)
        print(f'Epoch {epoch+1}/{num_epochs}, Training Loss: {epoch_loss:.4f}, Validation Loss: {val_loss:.4f}')
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path)
            print(f'Mодель сохранена с val_loss: {best_val_loss:.4f}')
    
    print('Обучение завершено')

def evaluate(model, val_loader, criterion, device: str):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for features, labels in val_loader:
            features = features.to(device).float()
            labels = labels.to(device).float()
            outputs = model(features)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * features.size(0)
    return running_loss / len(val_loader.dataset)


class FeatureDataset(torch.utils.data.Dataset):
    def __init__(self, X, y, extract_feachures=None):
        self.X = X
        self.y = y
        self.extract_feachures = extract_feachures

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        segment = self.X[idx]
        # Извлекаем фичи
        features = self.extract_feachures(segment) if self.extract_feachures is not None else segment
        return torch.tensor(features, dtype=torch.float32), torch.tensor(self.y[None, idx], dtype=torch.float32)

def get_data_loaders(X_train, y_train, X_val, y_val, batch_size: int, extract_feachures=None):
    # Создаем датасеты с извлечением фич
    train_dataset = FeatureDataset(X_train, y_train, extract_feachures)
    val_dataset = FeatureDataset(X_val, y_val, extract_feachures)

    # Создаем загрузчики данных
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader