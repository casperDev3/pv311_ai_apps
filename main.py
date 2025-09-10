# Машинне навчання: Дерева рішень, Випадковий ліс, Градієнтний бустинг та Логістична регресія
# Практичні приклади для занять

# %%
# Імпорт необхідних бібліотек
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score
import xgboost as xgb
from sklearn.datasets import load_boston, load_breast_cancer, load_iris
import warnings

warnings.filterwarnings('ignore')

# Налаштування візуалізації
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

print("✅ Всі бібліотеки успішно імпортовані!")

# %%
# =============================================================================
# ПАРА 1: ДЕРЕВА РІШЕНЬ І ВИПАДКОВИЙ ЛІС
# =============================================================================

# Частина 1: Демонстрація концепції ентропії
print("🌳 ДЕРЕВА РІШЕНЬ: Розуміння ентропії")
print("=" * 50)


def calculate_entropy(y):
    """Розрахунок ентропії для масиву міток"""
    _, counts = np.unique(y, return_counts=True)
    probabilities = counts / len(y)
    entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10))
    return entropy


# Приклад розрахунку ентропії
example_labels = ['sunny', 'sunny', 'sunny', 'sunny', 'sunny', 'sunny', 'sunny', 'sunny', 'rainy', 'rainy']
entropy_value = calculate_entropy(example_labels)
print(f"Приклад: {example_labels}")
print(f"Ентропія = {entropy_value:.3f}")


# Візуалізація ентропії
def plot_entropy():
    p = np.linspace(0.01, 0.99, 100)
    entropy = -p * np.log2(p) - (1 - p) * np.log2(1 - p)

    plt.figure(figsize=(8, 5))
    plt.plot(p, entropy, 'b-', linewidth=2)
    plt.xlabel('Ймовірність класу 1')
    plt.ylabel('Ентропія')
    plt.title('Функція ентропії для бінарної класифікації')
    plt.grid(True, alpha=0.3)
    plt.axvline(x=0.5, color='r', linestyle='--', alpha=0.7, label='Максимум невизначеності')
    plt.legend()
    plt.show()


plot_entropy()

# %%
# Завантаження даних про нерухомість (Housing prices)
print("\n🏠 ЗАВАНТАЖЕННЯ ДАНИХ ПРО НЕРУХОМІСТЬ")
print("=" * 40)

# Створимо синтетичний датасет, схожий на Boston Housing
np.random.seed(42)
n_samples = 500

# Генерація ознак
data = {
    'rooms': np.random.normal(6, 2, n_samples),  # Кількість кімнат
    'age': np.random.uniform(0, 100, n_samples),  # Вік будинку
    'distance_to_center': np.random.exponential(5, n_samples),  # Відстань до центру
    'crime_rate': np.random.exponential(0.5, n_samples),  # Рівень злочинності
    'pollution': np.random.normal(0.5, 0.2, n_samples),  # Рівень забруднення
}

# Створення цільової змінної (ціна) на основі ознак
price = (
        data['rooms'] * 15000 +  # Більше кімнат = вища ціна
        (100 - data['age']) * 500 +  # Новіші будинки дорожче
        -data['distance_to_center'] * 2000 +  # Ближче до центру = дорожче
        -data['crime_rate'] * 10000 +  # Менше злочинності = дорожче
        -data['pollution'] * 20000 +  # Менше забруднення = дорожче
        np.random.normal(0, 10000, n_samples)  # Випадковий шум
)

data['price'] = np.maximum(price, 50000)  # Мінімальна ціна 50к

# Створення DataFrame
df = pd.DataFrame(data)

print("Перші 5 рядків датасету:")
print(df.head())
print(f"\nРозмір датасету: {df.shape}")
print(f"Статистика:")
print(df.describe())

# %%
# Візуалізація даних
print("\n📊 ВІЗУАЛІЗАЦІЯ ДАНИХ")
print("=" * 25)

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

for i, column in enumerate(df.columns):
    if i < len(axes):
        if column == 'price':
            axes[i].hist(df[column], bins=30, alpha=0.7, color='red')
            axes[i].set_title(f'Розподіл: {column} (Цільова змінна)')
        else:
            axes[i].scatter(df[column], df['price'], alpha=0.6)
            axes[i].set_xlabel(column)
            axes[i].set_ylabel('price')
            axes[i].set_title(f'{column} vs price')

# Видалення зайвого графіка
if len(df.columns) < len(axes):
    fig.delaxes(axes[-1])

plt.tight_layout()
plt.show()

# %%
# Реалізація дерева рішень
print("\n🌳 ДЕРЕВО РІШЕНЬ ДЛЯ ПРОГНОЗУВАННЯ ЦІН")
print("=" * 45)

# Підготовка даних
X = df.drop('price', axis=1)
y = df['price']

# Розділення на тренувальну та тестову вибірки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Створення та навчання дерева рішень
dt_regressor = DecisionTreeRegressor(
    max_depth=5,  # Обмежуємо глибину для запобігання перенавчанню
    min_samples_split=10,  # Мінімум зразків для розділення
    min_samples_leaf=5,  # Мінімум зразків в листі
    random_state=42
)

dt_regressor.fit(X_train, y_train)

# Прогнозування
y_pred_dt = dt_regressor.predict(X_test)

# Оцінка якості
mse_dt = mean_squared_error(y_test, y_pred_dt)
rmse_dt = np.sqrt(mse_dt)

print(f"Дерево рішень:")
print(f"RMSE: {rmse_dt:.2f}")
print(f"Середня абсолютна помилка: {np.mean(np.abs(y_test - y_pred_dt)):.2f}")

# Важливість ознак
feature_importance_dt = pd.DataFrame({
    'feature': X.columns,
    'importance': dt_regressor.feature_importances_
}).sort_values('importance', ascending=False)

print(f"\nВажливість ознак (Дерево рішень):")
print(feature_importance_dt)

# %%
# Візуалізація дерева рішень (спрощена версія)
print("\n🎨 ВІЗУАЛІЗАЦІЯ ДЕРЕВА РІШЕНЬ")
print("=" * 35)

plt.figure(figsize=(20, 10))
plot_tree(dt_regressor,
          feature_names=X.columns,
          max_depth=3,  # Показуємо тільки перші 3 рівні
          filled=True,
          fontsize=10)
plt.title("Дерево рішень для прогнозування цін на нерухомість")
plt.show()

# %%
# Випадковий ліс
print("\n🌲 ВИПАДКОВИЙ ЛІС")
print("=" * 20)

# Створення та навчання випадкового лісу
rf_regressor = RandomForestRegressor(
    n_estimators=100,  # Кількість дерев
    max_depth=10,  # Максимальна глибина
    min_samples_split=5,  # Мінімум зразків для розділення
    min_samples_leaf=2,  # Мінімум зразків в листі
    random_state=42,
    n_jobs=-1  # Використання всіх ядер процесора
)

rf_regressor.fit(X_train, y_train)

# Прогнозування
y_pred_rf = rf_regressor.predict(X_test)

# Оцінка якості
mse_rf = mean_squared_error(y_test, y_pred_rf)
rmse_rf = np.sqrt(mse_rf)

print(f"Випадковий ліс:")
print(f"RMSE: {rmse_rf:.2f}")
print(f"Середня абсолютна помилка: {np.mean(np.abs(y_test - y_pred_rf)):.2f}")

# Важливість ознак
feature_importance_rf = pd.DataFrame({
    'feature': X.columns,
    'importance': rf_regressor.feature_importances_
}).sort_values('importance', ascending=False)

print(f"\nВажливість ознак (Випадковий ліс):")
print(feature_importance_rf)

# %%
# Порівняння результатів дерева рішень та випадкового лісу
print("\n📊 ПОРІВНЯННЯ ДЕРЕВА РІШЕНЬ ТА ВИПАДКОВОГО ЛІСУ")
print("=" * 55)

# Графік порівняння прогнозів
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Дерево рішень
axes[0].scatter(y_test, y_pred_dt, alpha=0.6)
axes[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
axes[0].set_xlabel('Справжні значення')
axes[0].set_ylabel('Прогнозовані значення')
axes[0].set_title(f'Дерево рішень (RMSE: {rmse_dt:.0f})')

# Випадковий ліс
axes[1].scatter(y_test, y_pred_rf, alpha=0.6)
axes[1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
axes[1].set_xlabel('Справжні значення')
axes[1].set_ylabel('Прогнозовані значення')
axes[1].set_title(f'Випадковий ліс (RMSE: {rmse_rf:.0f})')

plt.tight_layout()
plt.show()

# Порівняння важливості ознак
plt.figure(figsize=(12, 6))
x = np.arange(len(X.columns))
width = 0.35

plt.bar(x - width / 2, feature_importance_dt['importance'], width, label='Дерево рішень', alpha=0.8)
plt.bar(x + width / 2, feature_importance_rf['importance'], width, label='Випадковий ліс', alpha=0.8)

plt.xlabel('Ознаки')
plt.ylabel('Важливість')
plt.title('Порівняння важливості ознак')
plt.xticks(x, X.columns, rotation=45)
plt.legend()
plt.tight_layout()
plt.show()

# %%
# =============================================================================
# ПАРА 2: ГРАДІЄНТНИЙ БУСТИНГ ТА ЛОГІСТИЧНА РЕГРЕСІЯ
# =============================================================================

# Градієнтний бустинг з XGBoost
print("\n🚀 ГРАДІЄНТНИЙ БУСТИНГ (XGBoost)")
print("=" * 35)

# Створення та навчання XGBoost
xgb_regressor = xgb.XGBRegressor(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

xgb_regressor.fit(X_train, y_train)

# Прогнозування
y_pred_xgb = xgb_regressor.predict(X_test)

# Оцінка якості
mse_xgb = mean_squared_error(y_test, y_pred_xgb)
rmse_xgb = np.sqrt(mse_xgb)

print(f"XGBoost:")
print(f"RMSE: {rmse_xgb:.2f}")
print(f"Середня абсолютна помилка: {np.mean(np.abs(y_test - y_pred_xgb)):.2f}")

# Важливість ознак для XGBoost
feature_importance_xgb = pd.DataFrame({
    'feature': X.columns,
    'importance': xgb_regressor.feature_importances_
}).sort_values('importance', ascending=False)

print(f"\nВажливість ознак (XGBoost):")
print(feature_importance_xgb)

# %%
# Фінальне порівняння всіх алгоритмів регресії
print("\n🏆 ФІНАЛЬНЕ ПОРІВНЯННЯ АЛГОРИТМІВ РЕГРЕСІЇ")
print("=" * 45)

results_regression = pd.DataFrame({
    'Алгоритм': ['Дерево рішень', 'Випадковий ліс', 'XGBoost'],
    'RMSE': [rmse_dt, rmse_rf, rmse_xgb],
    'MAE': [
        np.mean(np.abs(y_test - y_pred_dt)),
        np.mean(np.abs(y_test - y_pred_rf)),
        np.mean(np.abs(y_test - y_pred_xgb))
    ]
})

print(results_regression)

# Візуалізація порівняння
plt.figure(figsize=(10, 6))
plt.bar(results_regression['Алгоритм'], results_regression['RMSE'], alpha=0.7)
plt.xlabel('Алгоритми')
plt.ylabel('RMSE')
plt.title('Порівняння точності алгоритмів регресії')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# %%
# =============================================================================
# ЛОГІСТИЧНА РЕГРЕСІЯ ТА КЛАСИФІКАЦІЯ
# =============================================================================

print("\n🎯 ЛОГІСТИЧНА РЕГРЕСІЯ: КЛАСИФІКАЦІЯ")
print("=" * 40)

# Завантаження даних для класифікації (рак грудей)
cancer_data = load_breast_cancer()
X_cancer = pd.DataFrame(cancer_data.data, columns=cancer_data.feature_names)
y_cancer = cancer_data.target

print("Датасет про рак грудей:")
print(f"Кількість зразків: {X_cancer.shape[0]}")
print(f"Кількість ознак: {X_cancer.shape[1]}")
print(f"Класи: {cancer_data.target_names}")
print(f"Розподіл класів: {np.bincount(y_cancer)}")

# %%
# Демонстрація сигмоїдної функції
print("\n📈 СИГМОЇДНА ФУНКЦІЯ")
print("=" * 22)


def sigmoid(z):
    """Сигмоїдна функція"""
    return 1 / (1 + np.exp(-z))


# Візуалізація сигмоїди
z = np.linspace(-10, 10, 100)
y_sigmoid = sigmoid(z)

plt.figure(figsize=(10, 6))
plt.plot(z, y_sigmoid, 'b-', linewidth=3, label='σ(z) = 1/(1+e^(-z))')
plt.axhline(y=0.5, color='r', linestyle='--', alpha=0.7, label='Поріг класифікації')
plt.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
plt.xlabel('z (лінійна комбінація ознак)')
plt.ylabel('σ(z) (ймовірність)')
plt.title('Сигмоїдна функція')
plt.grid(True, alpha=0.3)
plt.legend()
plt.ylim(-0.1, 1.1)
plt.show()

print("Ключові точки сигмоїди:")
print(f"σ(-∞) ≈ {sigmoid(-100):.6f}")
print(f"σ(0) = {sigmoid(0):.6f}")
print(f"σ(+∞) ≈ {sigmoid(100):.6f}")

# %%
# Підготовка даних для логістичної регресії
print("\n🔧 ПІДГОТОВКА ДАНИХ ДЛЯ ЛОГІСТИЧНОЇ РЕГРЕСІЇ")
print("=" * 48)

# Для спрощення візуалізації використаємо тільки 2 найважливіші ознаки
from sklearn.feature_selection import SelectKBest, f_classif

# Вибираємо 2 найкращі ознаки
selector = SelectKBest(score_func=f_classif, k=2)
X_cancer_selected = selector.fit_transform(X_cancer, y_cancer)

# Отримуємо назви вибраних ознак
selected_features = X_cancer.columns[selector.get_support()].tolist()
print(f"Вибрані ознаки для візуалізації: {selected_features}")

# Розділення на тренувальну та тестову вибірки
X_train_cancer, X_test_cancer, y_train_cancer, y_test_cancer = train_test_split(
    X_cancer_selected, y_cancer, test_size=0.2, random_state=42, stratify=y_cancer
)

# Масштабування ознак (важливо для логістичної регресії!)
scaler = StandardScaler()
X_train_cancer_scaled = scaler.fit_transform(X_train_cancer)
X_test_cancer_scaled = scaler.transform(X_test_cancer)

print(f"Форма тренувальних даних: {X_train_cancer_scaled.shape}")
print(f"Форма тестових даних: {X_test_cancer_scaled.shape}")

# %%
# Логістична регресія
print("\n🎯 ЛОГІСТИЧНА РЕГРЕСІЯ")
print("=" * 25)

# Створення та навчання моделі
logistic_model = LogisticRegression(random_state=42)
logistic_model.fit(X_train_cancer_scaled, y_train_cancer)

# Прогнозування
y_pred_cancer = logistic_model.predict(X_test_cancer_scaled)
y_pred_proba_cancer = logistic_model.predict_proba(X_test_cancer_scaled)[:, 1]

# Основні метрики
accuracy = accuracy_score(y_test_cancer, y_pred_cancer)
precision = precision_score(y_test_cancer, y_pred_cancer)
recall = recall_score(y_test_cancer, y_pred_cancer)
f1 = f1_score(y_test_cancer, y_pred_cancer)

print(f"Результати логістичної регресії:")
print(f"Точність (Accuracy): {accuracy:.3f}")
print(f"Прецизійність (Precision): {precision:.3f}")
print(f"Повнота (Recall): {recall:.3f}")
print(f"F1-мера: {f1:.3f}")

print(f"\nКоефіцієнти моделі:")
for i, coef in enumerate(logistic_model.coef_[0]):
    print(f"{selected_features[i]}: {coef:.4f}")
print(f"Вільний член: {logistic_model.intercept_[0]:.4f}")

# %%
# Confusion Matrix
print("\n📊 МАТРИЦЯ НЕВІДПОВІДНОСТЕЙ (CONFUSION MATRIX)")
print("=" * 50)

# Обчислення матриці невідповідностей
cm = confusion_matrix(y_test_cancer, y_pred_cancer)

# Візуалізація матриці невідповідностей
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Злоякісний', 'Доброякісний'],
            yticklabels=['Злоякісний', 'Доброякісний'])
plt.xlabel('Передбачені мітки')
plt.ylabel('Справжні мітки')
plt.title('Матриця невідповідностей')
plt.show()

# Детальний розбір матриці
tn, fp, fn, tp = cm.ravel()
print("Детальний розбір матриці невідповідностей:")
print(f"True Negatives (TN): {tn} - правильно передбачені злоякісні")
print(f"False Positives (FP): {fp} - помилково передбачені як доброякісні")
print(f"False Negatives (FN): {fn} - помилково передбачені як злоякісні")
print(f"True Positives (TP): {tp} - правильно передбачені доброякісні")

print(f"\nЗначення метрик:")
print(f"Precision = TP/(TP+FP) = {tp}/({tp}+{fp}) = {precision:.3f}")
print(f"Recall = TP/(TP+FN) = {tp}/({tp}+{fn}) = {recall:.3f}")
print(f"F1-Score = 2*(Precision*Recall)/(Precision+Recall) = {f1:.3f}")

# %%
# Демонстрація Cross-Entropy Loss
print("\n💰 ФУНКЦІЯ ВТРАТ: CROSS-ENTROPY")
print("=" * 35)


def cross_entropy_loss(y_true, y_pred_proba):
    """Обчислення cross-entropy loss"""
    # Додаємо маленьке число для уникнення log(0)
    epsilon = 1e-15
    y_pred_proba = np.clip(y_pred_proba, epsilon, 1 - epsilon)

    return -np.mean(y_true * np.log(y_pred_proba) + (1 - y_true) * np.log(1 - y_pred_proba))


# Обчислення loss для наших прогнозів
loss = cross_entropy_loss(y_test_cancer, y_pred_proba_cancer)
print(f"Cross-entropy loss: {loss:.4f}")

# Візуалізація функції втрат
p_range = np.linspace(0.01, 0.99, 100)
loss_y1 = -np.log(p_range)  # Коли справжня мітка = 1
loss_y0 = -np.log(1 - p_range)  # Коли справжня мітка = 0

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(p_range, loss_y1, 'b-', linewidth=2, label='y=1 (доброякісний)')
plt.plot(p_range, loss_y0, 'r-', linewidth=2, label='y=0 (злоякісний)')
plt.xlabel('Передбачена ймовірність')
plt.ylabel('Cross-entropy loss')
plt.title('Функція втрат Cross-entropy')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.hist(y_pred_proba_cancer[y_test_cancer == 0], bins=20, alpha=0.7, label='Злоякісні', color='red')
plt.hist(y_pred_proba_cancer[y_test_cancer == 1], bins=20, alpha=0.7, label='Доброякісні', color='blue')
plt.xlabel('Передбачена ймовірність')
plt.ylabel('Частота')
plt.title('Розподіл передбачених ймовірностей')
plt.legend()
plt.axvline(x=0.5, color='black', linestyle='--', alpha=0.7, label='Поріг')

plt.tight_layout()
plt.show()

# %%
# Візуалізація границі прийняття рішень
print("\n🎨 ВІЗУАЛІЗАЦІЯ ГРАНИЦІ ПРИЙНЯТТЯ РІШЕНЬ")
print("=" * 45)


def plot_decision_boundary(X, y, model, scaler, title):
    """Візуалізація границі прийняття рішень"""
    plt.figure(figsize=(10, 8))

    # Створення сітки точок
    h = 0.01
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    # Прогнозування для всієї сітки
    mesh_points = np.c_[xx.ravel(), yy.ravel()]
    mesh_points_scaled = scaler.transform(mesh_points)
    Z = model.predict_proba(mesh_points_scaled)[:, 1]
    Z = Z.reshape(xx.shape)

    # Візуалізація
    plt.contourf(xx, yy, Z, levels=50, alpha=0.6, cmap='RdYlBu')
    plt.colorbar(label='Ймовірність доброякісності')

    # Нанесення точок
    scatter = plt.scatter(X[:, 0], X[:, 1], c=y, cmap='RdYlBu', edgecolors='black')
    plt.xlabel(selected_features[0])
    plt.ylabel(selected_features[1])
    plt.title(title)

    # Додавання легенди
    handles, labels = scatter.legend_elements()
    plt.legend(handles, ['Злоякісний', 'Доброякісний'])

    plt.show()


plot_decision_boundary(X_train_cancer_scaled, y_train_cancer, logistic_model, scaler,
                       'Логістична регресія: Границя прийняття рішень')

# %%
# Порівняння різних алгоритмів класифікації
print("\n🏆 ПОРІВНЯННЯ АЛГОРИТМІВ КЛАСИФІКАЦІЇ")
print("=" * 40)

# Підготовка повного датасету (всі ознаки)
X_train_full, X_test_full, y_train_full, y_test_full = train_test_split(
    X_cancer, y_cancer, test_size=0.2, random_state=42, stratify=y_cancer
)

# Масштабування для логістичної регресії
scaler_full = StandardScaler()
X_train_full_scaled = scaler_full.fit_transform(X_train_full)
X_test_full_scaled = scaler_full.transform(X_test_full)

# Словник з алгоритмами
classifiers = {
    'Логістична регресія': LogisticRegression(random_state=42),
    'Дерево рішень': DecisionTreeClassifier(max_depth=5, random_state=42),
    'Випадковий ліс': RandomForestClassifier(n_estimators=100, random_state=42),
    'XGBoost': xgb.XGBClassifier(random_state=42)
}

# Результати
results_classification = []

for name, classifier in classifiers.items():
    # Вибір даних (масштабовані для логістичної регресії, оригінальні для інших)
    if name == 'Логістична регресія':
        X_train_used = X_train_full_scaled
        X_test_used = X_test_full_scaled
    else:
        X_train_used = X_train_full
        X_test_used = X_test_full

    # Навчання та прогнозування
    classifier.fit(X_train_used, y_train_full)
    y_pred = classifier.predict(X_test_used)

    # Метрики
    accuracy = accuracy_score(y_test_full, y_pred)
    precision = precision_score(y_test_full, y_pred)
    recall = recall_score(y_test_full, y_pred)
    f1 = f1_score(y_test_full, y_pred)

    results_classification.append({
        'Алгоритм': name,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1
    })

# Перетворення в DataFrame
results_df = pd.DataFrame(results_classification)
print(results_df.round(3))

# %%
# Візуалізація порівняння алгоритмів
print("\n📊 ВІЗУАЛІЗАЦІЯ ПОРІВНЯННЯ АЛГОРИТМІВ")
print("=" * 40)

# Графік порівняння метрик
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']

for i, metric in enumerate(metrics):
    row, col = i // 2, i % 2
    bars = axes[row, col].bar(results_df['Алгоритм'], results_df[metric], alpha=0.8)
    axes[row, col].set_title(f'{metric}')
    axes[row, col].set_ylabel('Значення')
    axes[row, col].set_ylim(0, 1)
    axes[row, col].tick_params(axis='x', rotation=45)

    # Додавання значень на стовпчики
    for bar, value in zip(bars, results_df[metric]):
        height = bar.get_height()
        axes[row, col].text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                            f'{value:.3f}', ha='center', va='bottom')

plt.tight_layout()
plt.show()

# %%
# Детальний звіт по класифікації для найкращого алгоритму
print("\n📋 ДЕТАЛЬНИЙ ЗВІТ КЛАСИФІКАЦІЇ")
print("=" * 35)

# Знаходимо найкращий алгоритм по F1-Score
best_algorithm = results_df.loc[results_df['F1-Score'].idxmax(), 'Алгоритм']
print(f"Найкращий алгоритм по F1-Score: {best_algorithm}")

# Створюємо та навчаємо найкращу модель
best_classifier = classifiers[best_algorithm]
if best_algorithm == 'Логістична регресія':
    best_classifier.fit(X_train_full_scaled, y_train_full)
    y_pred_best = best_classifier.predict(X_test_full_scaled)
else:
    best_classifier.fit(X_train_full, y_train_full)
    y_pred_best = best_classifier.predict(X_test_full)

# Детальний звіт
print(f"\nЗвіт класифікації для {best_algorithm}:")
print(classification_report(y_test_full, y_pred_best,
                            target_names=['Злоякісний', 'Доброякісний']))

# %%
# Демонстрація важливості вибору порогу для класифікації
print("\n⚖️ ВПЛИВ ПОРОГУ КЛАСИФІКАЦІЇ НА МЕТРИКИ")
print("=" * 45)

# Використовуємо логістичну регресію для демонстрації
log_reg = LogisticRegression(random_state=42)
log_reg.fit(X_train_full_scaled, y_train_full)
y_proba = log_reg.predict_proba(X_test_full_scaled)[:, 1]

# Різні пороги
thresholds = np.arange(0.1, 0.9, 0.1)
threshold_results = []

for threshold in thresholds:
    y_pred_threshold = (y_proba >= threshold).astype(int)

    precision = precision_score(y_test_full, y_pred_threshold)
    recall = recall_score(y_test_full, y_pred_threshold)
    f1 = f1_score(y_test_full, y_pred_threshold)

    threshold_results.append({
        'Поріг': threshold,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1
    })

threshold_df = pd.DataFrame(threshold_results)
print(threshold_df.round(3))

# Візуалізація впливу порогу
plt.figure(figsize=(12, 8))
plt.plot(threshold_df['Поріг'], threshold_df['Precision'], 'o-', label='Precision', linewidth=2)
plt.plot(threshold_df['Поріг'], threshold_df['Recall'], 's-', label='Recall', linewidth=2)
plt.plot(threshold_df['Поріг'], threshold_df['F1-Score'], '^-', label='F1-Score', linewidth=2)
plt.axvline(x=0.5, color='red', linestyle='--', alpha=0.7, label='Стандартний поріг (0.5)')
plt.xlabel('Поріг класифікації')
plt.ylabel('Значення метрики')
plt.title('Вплив порогу класифікації на якість моделі')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# %%
# Підсумок та рекомендації
print("\n🎓 ПІДСУМОК ЗАНЯТТЯ")
print("=" * 22)

print("📚 Що ми вивчили:")
print("1. Дерева рішень - прості для інтерпретації, але схильні до перенавчання")
print("2. Випадковий ліс - покращує дерева через ансамблювання")
print("3. Градієнтний бустинг (XGBoost) - послідовне покращення помилок")
print("4. Логістична регресія - лінейний метод для класифікації з ймовірностями")
print("5. Метрики якості: Precision, Recall, F1-Score, Confusion Matrix")

print("\n🔍 Ключові висновки:")
print("• Для регресії (прогнозування цін):")
print(f"  - Найкращий алгоритм: XGBoost (RMSE: {rmse_xgb:.0f})")
print("• Для класифікації (діагностика раку):")
print(f"  - Найкращий алгоритм: {best_algorithm}")

print("\n💡 Практичні рекомендації:")
print("• Завжди масштабуйте дані для логістичної регресії")
print("• Для медичних задач високий Recall часто важливіший за Precision")
print("• Використовуйте крос-валідацію для оцінки стабільності моделі")
print("• Експериментуйте з різними порогами класифікації")

print("\n🏠 Домашнє завдання:")
print("1. Завантажте власний датасет")
print("2. Застосуйте всі вивчені алгоритми")
print("3. Порівняйте результати та зробіть висновки")
print("4. Поекспериментуйте з гіперпараметрами")

print("\n✅ Заняття завершено! Успіхів у вивченні машинного навчання! 🚀")