import numpy as np
import matplotlib.pyplot as plt


# Гребінчаста регресія (англ. Ridge Regression) / Ridge Regression
from sklearn.linear_model import Ridge

# Задаємо параметри навчання учнів у класі: кількість годин, вид спорту, кишенькові на тиждень та їхні оцінки
X = np.array([[5, 1, 20],
              [6, 0, 15],
              [7, 1, 30],
              [8, 0, 25],
              [9, 1, 40],
              [10, 0, 35],
              [11, 1, 50]]) # [години навчання, вид спорту (1 - активний, 0 - пасивний), кишенькові на тиждень]

y = np.array([70, 65, 80, 75, 90, 85, 95]) # оцінки учнів

model = Ridge(alpha=1.0).fit(X, y) # навчання моделі гребінчастої регресії
# alpha - параметр регуляризації (чим більше значення, тим сильніше регуляризація)
# Регуляризація допомагає уникнути перенавчання моделі, особливо коли є багато вхідних змінних

# Тестові дані: нові учні з різними параметрами
X_test = np.array([[6, 1, 18],
                   [9, 0, 22],
                   [10, 1, 28],
                   [7, 0, 20],
                   [8, 1, 30]]) # нові учні
y_pred = model.predict(X_test) # передбачені оцінки для нових учнів
print("Передбачені оцінки для нових учнів:", y_pred)
# Коофіцієнти моделі
print("Коефіцієнти моделі:", model.coef_)
# Візуалізація результатів
# plt.scatter(range(len(y)), y, color='blue', label='Вихідні дані (оцінки)') # вихідні дані
# plt.scatter(range(len(y), len(y) + len(y_pred)), y_pred, color='red', marker='x', label='Передбачені оцінки') # передбачені оцінки
# plt.xlabel('Учні') # підпис осі X
# plt.ylabel('Оцінки') # підпис осі Y
# plt.title('Гребінчаста регресія: Параметри учнів vs Оцінки') # заголовок графіка
# plt.legend() # оточення поля
# plt.show() # показ графіка

# Bar plot - стовпчиковий графік
labels = [f'Учень {i+1}' for i in range(len(y))] + [f'Новий учень {i+1}' for i in range(len(y_pred))]
scores = np.concatenate([y, y_pred])
x = np.arange(len(labels))
plt.bar(x, scores, color=['blue']*len(y) + ['red']*len(y_pred))
plt.xticks(x, labels, rotation=45)
plt.xlabel('Учні') # підпис осі X
plt.ylabel('Оцінки') # підпис осі Y
plt.title('Гребінчаста регресія: Параметри учнів vs Оцінки') # заголовок графіка
plt.tight_layout() # щоб не обрізались підписи
plt.show() # показ графіка