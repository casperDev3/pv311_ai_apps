from pyexpat import features

import numpy as np
import matplotlib.pyplot as plt
from seaborn.algorithms import bootstrap
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
import seaborn as sns # confusion matrix


# Генерація синтетичних даних для 200 ресторанів
np.random.seed(42) # для відтворюваності
num_samples = 200

# Генеруємо випадкові ознаки для ресторанів
price = np.random.randint(1, 5, num_samples) # ціновий діапазон від 1 до 5
food_quality = np.random.randint(1, 6, num_samples) # якість їжі від 1 до 5
service = np.random.randint(1, 6, num_samples) # якість обслуговування від 1 до 5
location = np.random.randint(1, 6, num_samples) # розташування від 1 до 5

# Обраховуємо середіній рейтинг та генеруємо цільову змінну (1 - хороший ресторан, 0 - поганий)
average_rating = (food_quality + service + location) / 3
target = (average_rating > 3.5).astype(int) # хороший ресторан, якщо середній рейтинг > 3.5

# Розділяємо дані на тренувальні та тестові набори (80% тренувальні, 20% тестові)
X_train, X_test, y_train, y_test = train_test_split(
    np.column_stack((price, food_quality, service, location)),
    target,
    test_size=0.2,
    random_state=42
)




rf_model = RandomForestClassifier(
    n_estimators=100, # кількість дерев у лісі
    random_state=42, # для відтворюваності
    bootstrap=True, # використання бутстрепу для вибірок
    max_features='sqrt' # кількість ознак для розбиття кожного вузла
)

# Навчання моделі на тренувальних даних
rf_model.fit(X_train, y_train)

# Робимо передбачення на тестових даних
y_pred = rf_model.predict(X_test)

# Обчислення точності моделі
accuracy = accuracy_score(y_test, y_pred)
print(f"Точність моделі: {accuracy:.2f}")

# Візуалізація
features_names = ['Price', 'Food Quality', 'Service', 'Location']
importances = rf_model.feature_importances_

# Створюємо бар-чарт для важливості ознак
# plt.figure(figsize=(10, 6)) # розмір фігури
# plt.barh (features_names, importances, color='skyblue') # горизонтальний бар-чарт
# plt.xlabel('Importance') # підпис осі X
# plt.title('Feature Importances in Random Forest Classifier') # заголовок
# for i, v in enumerate(importances):
#     plt.text(v + 0.01, i, f"{v:.2f}", color='blue', va='center') # додавання тексту з важливістю
# plt.tight_layout()
# plt.show() # показуємо графік

new_restaurant = np.array([[3, 5, 4, 4]]) # приклад нового ресторану
prediction = rf_model.predict(new_restaurant)

print(
    f"Новий ресторан з характеристиками {new_restaurant[0]} "
    f"прогнозується як {'хороший' if prediction[0] == 1 else 'поганий'} ресторан."

)

# cm = confusion_matrix(y_test, y_pred)
# plt.figure(figsize=(8, 6))
# sns.heatmap(
#     cm, # дані
#     annot=True, # інверсія
#     fmt='d', #  формат
#     cmap='Blues',
#     xticklabels=['Поганий', 'Хороший'],
#     yticklabels=['Хороший', 'Поганий']
# )
#
# plt.xlabel("Передбачено")
# plt.ylabel("Фактичне")
# plt.title("CF matrix")
# plt.tight_layout()
# plt.show()