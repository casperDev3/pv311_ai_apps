import torch
import torch.nn as nn
import torch.optim as optim
from  torch.utils.data import DataLoader, TensorDataset
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image # імпорт бібліотеки PIL для роботи з зображеннями
import requests
from io import BytesIO
import os

# Перевірка наявності GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


# Визначення трансформацій для даних
transform_train = transforms.Compose([
    transforms.Resize((224, 224)), # зміна розміру зображень
    transforms.RandomHorizontalFlip(p=0.5), # випадкове горизонтальне відображення на 50%
    transforms.RandomRotation(10), # випадкова ротація на ±10 градусів
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1), # випадкові зміни яскравості, контрасту, насиченості та відтінку
    transforms.ToTensor(), # перетворення зображень у тензори
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)) # нормалізація
]) # трансформації для тренувальних даних

transform_test = transforms.Compose([
    transforms.Resize((224, 224)), # зміна розміру зображень
    transforms.ToTensor(), # перетворення зображень у тензори
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
]) # трансформації для тестових даних

# Створення датасету котиків з CIFAR-10
class CatDataset(torch.utils.data.Dataset):
    def __init__(self, cifar_dataset):
        self.data = [] # список для збереження зображень
        self.labels = [] # список для збереження міток
        for i, (img, label) in enumerate(cifar_dataset):
            if label == 3: # якщо мітка відповідає коту (3 - це мітка кота в CIFAR-10)
                self.data.append(img) # додавання зображення кота
                self.labels.append(1) # додавання мітки 1 для кота
            elif i % 10 == 0: # додавання інших зображень з інтервалом 1 з 10
                self.data.append(img) # додавання інших зображень (1 з 10)
                self.labels.append(0) # додавання мітки 0 для інших зображень

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

# Архітектура моделі (проста CNN)
class CatDetectionCNN(nn.Module):
    def __init__(self):
        super(CatDetectionCNN, self).__init__() # виклик конструктора батьківського класу

        # Конволюційні шари - це звичайні шари згортки (зворотний зв'язок) для витягання ознак із зображень
        # Згортка - зворотнє поширення інформації
        self.conv_layers = nn.Sequential(
            # Перший блок
            nn.Conv2d(3, 32, kernel_size=3, padding=1), # 3 вхідні канали (RGB), 32 вихідні канали, розмір ядра 3x3, padding=1 для збереження розміру
            nn.BatchNorm2d(32), # нормалізація пакетів для стабілізації навчання
            nn.ReLU(inplace=True), # функція активації ReLU
            nn.Conv2d(32, 32, kernel_size=3, padding=1), # другий шар згортки
            nn.BatchNorm2d(32), # нормалізація пакетів
            nn.ReLU(inplace=True), # функція активації ReLU
            nn.MaxPool2d(kernel_size=2, stride=2), # максимальне підвибіркування для зменшення розміру
            nn.Dropout2d(0.25), # випадкове відключення нейронів для запобігання перенавчанню

            # Другий блок
            nn.Conv2d(32, 64, kernel_size=3, padding=1), # 32 вхідні канали, 64 вихідні канали
            nn.BatchNorm2d(64), # нормалізація пакетів
            nn.ReLU(inplace=True), # функція активації ReLU
            nn.Conv2d(64, 64, kernel_size=3, padding=1), # другий шар згортки
            nn.BatchNorm2d(64), # нормалізація пакетів
            nn.ReLU(inplace=True), # функція активації ReLU
            nn.MaxPool2d(kernel_size=2, stride=2), # максимальне підвибіркування
            nn.Dropout2d(0.25), # випадкове відключення нейронів для запобігання перенавчанню

            # Третій блок
            nn.Conv2d(64, 128, kernel_size=3, padding=1), # 64 вхідні канали, 128 вихідних каналів
            nn.BatchNorm2d(128), # нормалізація пакетів
            nn.ReLU(inplace=True), # функція активації ReLU
            nn.Conv2d(128, 128, kernel_size=3, padding=1), # другий шар згортки
            nn.BatchNorm2d(128), # нормалізація пакетів
            nn.ReLU(inplace=True), # функція активації ReLU
            nn.MaxPool2d(kernel_size=2, stride=2), # максимальне підвибіркування
        ) # послідовність шарів

        # Повнозв'язні шари - це шари, де кожен нейрон пов'язаний з кожним нейроном попереднього шару
        self.classifier = nn.Sequential(
            nn.Flatten(), # розгортання багатовимірного тензора в одномірний вектор
            nn.Linear(128 * 28 * 28, 512), # повнозв'язний шар з 128*28*28 входами та 512 виходами
            nn.ReLU(inplace=True), # функція активації ReLU
            nn.Dropout(0.5), # випадкове відключення нейронів для запобігання перенавчанню
            nn.Linear(512, 1), # повнозв'язний шар з 512 входами та 1 виходом (бінарна класифікація)
            nn.ReLU(inplace=True), # функція активації ReLU
            nn.Dropout(0.5),
            nn.Linear(128, 2) # повнозв'язний шар з 128 входами та 2 виходами (для двох класів: кіт і не кіт)

        ) # послідовність шарів

    def forward(self, x):
        x = self.conv_layers(x) # проходження через конволюційні шари
        x = self.classifier(x) # проходження через повнозв'язні шари
        return x

# Завантаження  CIFAR-10 датасету
cifar_train = CIFAR10(root='./data', train=True, download=True, transform=transform_train)
cifar_test = CIFAR10(root='./data', train=False, download=True, transform=transform_test)


# Створення датасетів котиків
train_dataset = CatDataset(cifar_train)
test_dataset = CatDataset(cifar_test)

# Створення DataLoader для тренувального та тестового наборів
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2) # batch_size - це кількість зразків, що обробляються одночасно, shuffle=True для випадкового перемішування даних
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2)

print("test")


# ініціалізація моделі
model = CatDetectionCNN().to(device) # перенесення моделі на GPU або CPU
print(f"Модель створена з {sum(p.numel() for p in model.parameters() if p.requires_grad)} параметрами.")

# Визначення функції втрат та оптимізатора
criterion = nn.CrossEntropyLoss() # функція втрат для багатокласової класифікації
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4) # оптимізатор Adam з початковою швидкістю навчання 0.001
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1) # зменшення швидкості навчання кожні 10 епох в 10 разів

# Функція для тренування моделі
def train_model(model, train_loader, test_loader, num_epoch=20):
    train_loses = [] # список для збереження втрат на тренувальному наборі
    train_accuracies = [] # список для збереження точності на тренувальному наборі
    test_accuracies = [] # список для збереження точності на тестовому наборі

    for epoch in range(num_epoch): # цикл по епохах
        model.train() # встановлення моделі в режим тренування
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for batch_idx, (data, labels) in enumerate(train_loader):
            data, target = data.to(device), labels.to(device) # перенесення даних на GPU або CPU

            optimizer.zero_grad() # обнулення градієнтів
            outputs = model(data) # проходження вперед
            loss = criterion(outputs, target) # обчислення втрат
            loss.backward() # зворотнє поширення
            optimizer.step() # оновлення ваг

            running_loss += loss.item() # накопичення втрат
            _, predicted = torch.max(outputs.data, 1) # отримання передбачених міток
            total_train += target.size(0) # загальна кількість зразків
            correct_train += (predicted == target).sum().item() # кількість правильних передбачень

            if batch_idx % 50 == 0:
                print(f"Epoch [{epoch+1}/{num_epoch}], Batch [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}")

            # Обчислення точності на тренувальному наборі
        train_accuracy = 100 * correct_train / total_train
        train_accuracies.append(train_accuracy)

        # Тестування моделі на тестовому наборі
        model.eval() # встановлення моделі в режим оцінки
        correct_test = 0
        total_test = 0

        with torch.no_grad(): # відключення обчислення градієнтів
            for data, target in test_loader:
                data, target = data.to(device), target.to(device) # перенесення даних на GPU або CPU
                outputs = model(data) # проходження вперед
                _, predicted = torch.max(outputs.data, 1) # отримання передбачених міток
                total_test += target.size(0)
                correct_test += (predicted == target).sum().item()

        test_accuracy = 100 * correct_test / total_test

        train_loses.append(running_loss / len(train_loader)) # середня втрата за епоху
        test_accuracies.append(test_accuracy)
        train_accuracies.append(train_accuracy)

        print(f'Epoch [{epoch+1}/{num_epoch}]'
              f', Loss: {running_loss / len(train_loader):.4f}'
              f', Train Accuracy: {train_accuracy:.2f}%'
              f', Test Accuracy: {test_accuracy:.2f}%')
        print("-" * 50)

        scheduler.step() # оновлення швидкості навчання

    return  train_loses, train_accuracies, test_accuracies

# Тренування моделі
train_losses, train_accuracies, test_accuracies = train_model(model, train_loader, test_loader, num_epoch=20)

# Збереження моделі
model_path = "saved_model/cat_detection_cnn.pth"
torch.save(model.state_dict(), model_path)
print(f"Модель збережена за шляхом: {model_path}")

