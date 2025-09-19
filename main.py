import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime


# Генерація даних для навчання
def generate_data(num_samples=1000):
    # Випадкові числа від -10 до 10
    x = np.random.uniform(-10, 10, num_samples)
    # Правильна відповідь: x * 2
    y = x * 2
    return x, y


# Нейронна мережа з прихованим шаром та ReLU
class MultiplyByTwoModel(nn.Module):
    def __init__(self, hidden_size=10):
        super(MultiplyByTwoModel, self).__init__()
        # Вхідний шар -> прихований шар
        self.hidden = nn.Linear(1, hidden_size)
        # ReLU активація
        self.relu = nn.ReLU()
        # Прихований шар -> вихідний шар
        self.output = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # Пряма передача через приховний шар з ReLU
        x = self.hidden(x)
        x = self.relu(x)
        x = self.output(x)
        return x


# Підготовка даних
X_train, y_train = generate_data(1000)
X_test, y_test = generate_data(200)

# Перетворення в тензори PyTorch
X_train_tensor = torch.FloatTensor(X_train).reshape(-1, 1)
y_train_tensor = torch.FloatTensor(y_train).reshape(-1, 1)
X_test_tensor = torch.FloatTensor(X_test).reshape(-1, 1)
y_test_tensor = torch.FloatTensor(y_test).reshape(-1, 1)

# Створення моделі з 10 нейронами в прихованому шарі
model = MultiplyByTwoModel(hidden_size=10)

# Функція втрат і оптимізатор
criterion = nn.MSELoss()  # Середньоквадратична похибка
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Навчання моделі
num_epochs = 2000  # Збільшуємо кількість епох для складнішої моделі
losses = []

print("Початок навчання...")
for epoch in range(num_epochs):
    # Пряма передача
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)

    # Зворотна передача і оптимізація
    optimizer.zero_grad()  # Обнулення градієнтів
    loss.backward()  # Обчислення градієнтів
    optimizer.step()  # Оновлення параметрів

    losses.append(loss.item())

    # Виведення прогресу кожні 10 епох
    if (epoch + 1) % 10 == 0:
        print(f'Епоха [{epoch + 1}/{num_epochs}], Втрати: {loss.item():.6f}')

print("Навчання завершено!")

# Тестування моделі
model.eval()  # Режим оцінювання
with torch.no_grad():
    test_outputs = model(X_test_tensor)
    test_loss = criterion(test_outputs, y_test_tensor)
    print(f"Втрати на тестових даних: {test_loss.item():.6f}")

# Інформація про архітектуру
print(f"\nАрхітектура моделі:")
print(f"Вхідний шар: 1 нейрон")
print(f"Прихований шар: 10 нейронів з ReLU активацією")
print(f"Вихідний шар: 1 нейрон")
print(f"Загальна кількість параметрів: {sum(p.numel() for p in model.parameters())}")

# Тестування на конкретних прикладах
test_values = [1, 2, 5, -3, 10]
print(f"\nТестування на конкретних значеннях:")
for val in test_values:
    input_tensor = torch.FloatTensor([[val]])
    predicted = model(input_tensor).item()
    expected = val * 2
    print(f"Вхід: {val}, Передбачення: {predicted:.2f}, Очікується: {expected}")

#  Збереження моделі
models_dir = "saved_models"
# os.makedirs(models_dir, exist_ok=True)
#
# Метод #1: Збереження всієї моделі
model_path_full = os.path.join(models_dir, f"model_full_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pt")
torch.save(model, model_path_full)
#
# # Метод #2: Збереження лише параметрів моделі
# model_path_state = os.path.join(models_dir, f"model_state_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pth")
# torch.save(model.state_dict(), model_path_state)


# # Візуалізація втрат
# plt.figure(figsize=(10, 4))
#
# plt.subplot(1, 2, 1)
# plt.plot(losses)
# plt.title('Втрати під час навчання')
# plt.xlabel('Епоха')
# plt.ylabel('Втрати')
# plt.grid(True)
#
# # Візуалізація результатів
# plt.subplot(1, 2, 2)
# with torch.no_grad():
#     # Візьмемо підмножину тестових даних для графіка
#     sample_x = X_test[:50]
#     sample_y_true = y_test[:50]
#     sample_x_tensor = torch.FloatTensor(sample_x).reshape(-1, 1)
#     sample_y_pred = model(sample_x_tensor).numpy().flatten()
#
# plt.scatter(sample_x, sample_y_true, alpha=0.5, label='Справжні значення')
# plt.scatter(sample_x, sample_y_pred, alpha=0.5, label='Передбачення')
# plt.plot([-10, 10], [-20, 20], 'r--', label='y = 2x (ідеальна лінія)')
# plt.xlabel('Вхід')
# plt.ylabel('Вихід')
# plt.title('Результати моделі')
# plt.legend()
# plt.grid(True)
#
# plt.tight_layout()
# plt.show()