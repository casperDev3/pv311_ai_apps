import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

X = np.array([30, 35, 40, 45, 50, 55, 60]).reshape(-1, 1) # площа в квадратних метрах
y = np.array([30000, 35000, 40000, 45000, 50000, 55000, 60000]) # ціна в доларах

model = LinearRegression().fit(X, y) # навчання моделі

X_test = np.array([32, 38, 42, 48, 52, 58]).reshape(-1, 1) # тестові дані (площа в квадратних метрах)
y_pred = model.predict(X_test) # передбачені ціни

plt.scatter(X, y, color='blue', label='Вихідні дані') # вихідні дані
plt.plot(X_test, y_pred, color='red', linestyle='--', label='Лінія регресії') # лінія регресії
plt.xlabel('Площа (кв.м)') # підпис осі X
plt.ylabel('Ціна (долари)') # підпис осі Y
plt.title('Лінійна регресія: Площа vs Ціна') # заголовок графіка
plt.legend() # оточення поля
plt.show() # показ графіка