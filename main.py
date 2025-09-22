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
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
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

# Завантаження  CIFAR-10 датасету
cifar_train = CIFAR10(root='./data', train=True, download=True, transform=transform_train)
cifar_test = CIFAR10(root='./data', train=False, download=True, transform=transform_test)

# Створення датасетів котиків
train_dataset = CatDataset(cifar_train)
test_dataset = CatDataset(cifar_test)

# Створення DataLoader для тренувального та тестового наборів
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2) # batch_size - це кількість зразків, що обробляються одночасно, shuffle=True для випадкового перемішування даних
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2)

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