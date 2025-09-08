import numpy as np
from sklearn.linear_model import Lasso

# інфдяція (у %), безробіття (у %), процентні ставки (у %), експорт (у млрд $), імпорт (у млрд $)
economic_indicators = np.array([
    [2.1, 5.0, 1.5, 300, 250],
    [2.3, 3.8, 1.7, 320, 260],
    [1.9, 5.2, 1.9, 310, 255],
    [2.5, 2.5, 1.8, 330, 270],
    [2.0, 5.1, 2.2, 305, 252],
    [2.4, 1.7, 1.9, 325, 265]
])

# ВВП (у млрд $)
gdp = np.array([3.2, 2.1, 5.1, 4, 1.5, 2.9])

# Створення та навчання моделі Lasso регресії
lasso = Lasso(alpha=0.3)
lasso.fit(economic_indicators, gdp)

# Виведення коефіцієнтів моделі
print("Коефіцієнти моделі Lasso регресії:")
print(lasso.coef_)

# Прогнозування ВВП на основі нових економічних показників
new_indicators = np.array([[2.2, 4.9, 1.6, 315, 258]])
predicted_gdp = lasso.predict(new_indicators)
print(predicted_gdp)