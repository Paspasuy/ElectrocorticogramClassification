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

def get_data_loaders(X_train, y_train, X_val, y_val, batch_size: int, split: float = 0.8):
    # Создаем тренировочный и валидационный наборы данных
    train_dataset = TensorDataset(
        torch.tensor(X_train), 
        torch.tensor(y_train)
    )
    val_dataset = TensorDataset(
        torch.tensor(X_val),
        torch.tensor(y_val)
    )
    # Создаем загрузчики данных
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader