
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import requests
from io import BytesIO
import os

# Перевірка доступності GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Використовується пристрій: {device}')

# Визначення трансформацій для даних
transform_train = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

transform_test = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# Клас для створення датасету котиків з CIFAR-10
class CatDataset(Dataset):
    def __init__(self, cifar_dataset):
        # Фільтруємо лише зображення котиків (клас 3 в CIFAR-10)
        self.data = []
        self.labels = []

        for i, (img, label) in enumerate(cifar_dataset):
            # Створюємо бінарну класифікацію: 1 - котик, 0 - не котик
            if label == 3:  # Котик
                self.data.append(img)
                self.labels.append(1)
            elif i % 10 == 0:  # Беремо кожне 10-те не-котикове зображення для балансу
                self.data.append(img)
                self.labels.append(0)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


# Завантаження CIFAR-10 датасету
print("Завантаження CIFAR-10 датасету...")
cifar_train = CIFAR10(root='./data', train=True, transform=transform_train, download=True)
cifar_test = CIFAR10(root='./data', train=False, transform=transform_test, download=True)

# Створення датасетів котиків
train_dataset = CatDataset(cifar_train)
test_dataset = CatDataset(cifar_test)

print(f"Розмір тренувального датасету: {len(train_dataset)}")
print(f"Розмір тестового датасету: {len(test_dataset)}")

# Створення DataLoaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


# Архітектура нейронної мережі для розпізнавання котиків
class CatDetectionCNN(nn.Module):
    def __init__(self):
        super(CatDetectionCNN, self).__init__()

        # Конволюційні шари
        self.conv_layers = nn.Sequential(
            # Перший блок
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25),

            # Другий блок
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25),

            # Третій блок
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25),
        )

        # Повнозв'язані шари
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 28 * 28, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, 2)  # 2 класи: котик/не котик
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.classifier(x)
        return x


# Ініціалізація моделі
model = CatDetectionCNN().to(device)
print(f"Модель створена з {sum(p.numel() for p in model.parameters())} параметрами")

# Функція втрат та оптимізатор
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)


# Функція тренування
def train_model(model, train_loader, test_loader, num_epochs=20):
    train_losses = []
    train_accuracies = []
    test_accuracies = []

    for epoch in range(num_epochs):
        # Тренування
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total_train += target.size(0)
            correct_train += (predicted == target).sum().item()

            if batch_idx % 50 == 0:
                print(
                    f'Epoch [{epoch + 1}/{num_epochs}], Batch [{batch_idx}/{len(train_loader)}], Loss: {loss.item():.4f}')

        # Обчислення точності на тренувальних даних
        train_accuracy = 100 * correct_train / total_train
        avg_loss = running_loss / len(train_loader)

        # Тестування
        model.eval()
        correct_test = 0
        total_test = 0

        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                _, predicted = torch.max(output.data, 1)
                total_test += target.size(0)
                correct_test += (predicted == target).sum().item()

        test_accuracy = 100 * correct_test / total_test

        train_losses.append(avg_loss)
        train_accuracies.append(train_accuracy)
        test_accuracies.append(test_accuracy)

        print(f'Epoch [{epoch + 1}/{num_epochs}]:')
        print(f'  Тренувальна точність: {train_accuracy:.2f}%')
        print(f'  Тестова точність: {test_accuracy:.2f}%')
        print(f'  Середня втрата: {avg_loss:.4f}')
        print('-' * 50)

        scheduler.step()

    return train_losses, train_accuracies, test_accuracies


# Тренування моделі
print("Початок тренування...")
train_losses, train_accs, test_accs = train_model(model, train_loader, test_loader, num_epochs=15)

# Збереження моделі
model_path = 'cat_detection_model.pth'
torch.save({
    'model_state_dict': model.state_dict(),
    'model_class': CatDetectionCNN,
}, model_path)
print(f"Модель збережена як {model_path}")


# Функція для завантаження зображення з URL
def load_image_from_url(url):
    try:
        response = requests.get(url)
        img = Image.open(BytesIO(response.content)).convert('RGB')
        return img
    except Exception as e:
        print(f"Помилка завантаження зображення: {e}")
        return None


# Функція для предикції
def predict_cat(model, image_path_or_url, transform):
    model.eval()

    # Завантаження зображення
    if image_path_or_url.startswith('http'):
        image = load_image_from_url(image_path_or_url)
    else:
        image = Image.open(image_path_or_url).convert('RGB')

    if image is None:
        return None, None

    # Попередня обробка
    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.softmax(output, dim=1)
        _, predicted = torch.max(output, 1)

    is_cat = predicted.item() == 1
    confidence = probabilities[0][predicted].item() * 100

    return is_cat, confidence, image


# Тестування на зображеннях з інтернету
test_images = [
    "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Cat03.jpg/1200px-Cat03.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/thumb/1/15/Cat_August_2010-4.jpg/1200px-Cat_August_2010-4.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/thumb/5/5e/Sleeping_cat_on_her_back.jpg/1200px-Sleeping_cat_on_her_back.jpg"
]

print("\nТестування на зображеннях:")
for i, url in enumerate(test_images):
    result = predict_cat(model, url, transform_test)
    if result[0] is not None:
        is_cat, confidence, _ = result
        print(f"Зображення {i + 1}: {'КОТИК' if is_cat else 'НЕ КОТИК'} (впевненість: {confidence:.1f}%)")

# Візуалізація результатів тренування
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.plot(train_losses)
plt.title('Втрата під час тренування')
plt.xlabel('Епоха')
plt.ylabel('Втрата')

plt.subplot(1, 3, 2)
plt.plot(train_accs, label='Тренувальна точність')
plt.plot(test_accs, label='Тестова точність')
plt.title('Точність моделі')
plt.xlabel('Епоха')
plt.ylabel('Точність (%)')
plt.legend()

plt.subplot(1, 3, 3)
# Показати приклад передбачення
if len(test_images) > 0:
    result = predict_cat(model, test_images[0], transform_test)
    if result[0] is not None:
        is_cat, confidence, image = result
        plt.imshow(image)
        plt.title(f'Предикція: {"КОТИК" if is_cat else "НЕ КОТИК"}\nВпевненість: {confidence:.1f}%')
        plt.axis('off')

plt.tight_layout()
plt.show()

print("\nМодель успішно натренована!")
print(f"Фінальна тестова точність: {test_accs[-1]:.2f}%")
