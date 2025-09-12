import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import seaborn as sns # confusion matrix
import xgboost as xgb
from pandas.core.common import random_state

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
import warnings

from xgboost.testing import eval_error_metric

print("Прогнозування діабету за допомогою XGBoost для Бінарної Класифікації")

print("1. Завантаження даних...")

# Створюємо синтетичний набір даних для демонстрації на Pima Indians Diabetes Database
np.random.seed(42) # для відтворюваності
num_samples = 768 # кількість зразків у наборі даних

# Генеруємо ознаки
data = {
    'Pregnancies': np.random.randint(0, 17, num_samples), # кількість вагітностей
    'Glucose': np.random.randint(50, 200, num_samples), # рівень глюкози
    'BloodPressure': np.random.randint(30, 122, num_samples), # артеріальний тиск
    'SkinThickness': np.random.randint(7, 99, num_samples), # товщина шкіри
    'Insulin': np.random.randint(15, 846, num_samples), # рівень інсуліну
    'BMI': np.random.uniform(18.0, 67.1, num_samples), # індекс маси тіла
    'DiabetesPedigreeFunction': np.random.uniform(0.078, 2.42, num_samples), # функція родинної схильності до діабету
    'Age': np.random.randint(21, 81, num_samples), # вік
}

df = pd.DataFrame(data)

# Генеруємо цільову змінну (1 - діабет, 0 - не діабет)
outcome = (
    (df['Glucose'] > 140) * 0.4 +
    (df['BMI'] > 30) * 0.3 +
    (df['Age'] > 45) * 0.2 +
    (df['DiabetesPedigreeFunction'] > 0.5) * 0.1 +
    np.random.normal(0, 0.1, num_samples)
)
df['Outcome'] = (outcome > 0.5).astype(int)

# Забезпечуємо позитивні значення для ознак, де це необхідно
df = df.abs() # робимо всі значення позитивними
df['Outcome'] = ((df['Glucose'] > 140) | (df['BMI'] > 30)).astype(int) # цільова змінна залежить від глюкози та ІМТ

# 2. Підготовка даних до навчання моделі
print("2. Підготовка даних до навчання моделі...")

X = df.drop('Outcome', axis=1) # ознаки
y = df['Outcome'] # цільова змінна

# Розділяємо дані на тренувальні та тестові набори (80% тренувальні, 20% тестові)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Тренувальна вибірка:", X_train.shape, "Тестова вибірка:", X_test.shape)

# Навчання моделі XGBoost
print("3. Навчання моделі XGBoost...")

base_model = xgb.XGBClassifier(
    random_state = 42, # для відтворюваності
    eval_metric = 'logloss', # метрика для оцінки якості
)

#  навчання моделі на тренувальних даних
base_model.fit(X_train, y_train)

# Робимо передбачення на тестових даних
print("4. Робимо передбачення на тестових даних...")
y_pred_base = base_model.predict(X_test)

# Обчислення точності моделі
accuracy_base = accuracy_score(y_test, y_pred_base)
print(f"Точність базової моделі: {accuracy_base:.2f}")


# Прогнозування даних з реальними значеннями
print("5. Прогнозування даних з реальними значеннями...")
real_predictions_one = base_model.predict([[6, 148, 72, 35, 0, 33.6, 0.627, 50]]) # приклад з діабетом
real_predictions_two = base_model.predict([[1, 85, 66, 29, 0, 26.6, 0.351, 31]]) # приклад без діабету

print(f"Прогноз для [6, 148, 72, 35, 0, 33.6, 0.627, 50]: {'Діабет' if real_predictions_one[0] == 1 else 'Не діабет'}")
print(f"Прогноз для [1, 85, 66, 29, 0, 26.6, 0.351, 31]: {'Діабет' if real_predictions_two[0] == 1 else 'Не діабет'}")
