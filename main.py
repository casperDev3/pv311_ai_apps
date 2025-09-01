# Імпорт необхідних бібліотек
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from matplotlib.patches import Circle
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Налаштування стилю
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


# 1. Лінійний графік з градієнтом
def create_gradient_line_plot():
    x = np.linspace(0, 10, 100)
    y1 = np.sin(x) * np.exp(-x / 10)
    y2 = np.cos(x) * np.exp(-x / 8)

    fig, ax = plt.subplots(figsize=(12, 6))

    # Створення градієнтного фону
    gradient = np.linspace(0, 1, 256).reshape(1, -1)
    ax.imshow(gradient, extent=[0, 10, -1, 1], aspect='auto', alpha=0.3, cmap='viridis')

    ax.plot(x, y1, linewidth=3, label='sin(x)×e^(-x/10)', color='#FF6B6B')
    ax.plot(x, y2, linewidth=3, label='cos(x)×e^(-x/8)', color='#4ECDC4')

    ax.set_xlabel('X значення', fontsize=14, fontweight='bold')
    ax.set_ylabel('Y значення', fontsize=14, fontweight='bold')
    ax.set_title('Затухаючі коливання з градієнтним фоном', fontsize=16, fontweight='bold')
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


# 2. Красивий scatter plot
def create_beautiful_scatter():
    # Генерація даних
    n = 200
    x = np.random.randn(n)
    y = 2 * x + np.random.randn(n) * 0.5
    colors = np.random.rand(n)
    sizes = 1000 * np.random.rand(n)

    fig, ax = plt.subplots(figsize=(10, 8))

    scatter = ax.scatter(x, y, c=colors, s=sizes, alpha=0.6, cmap='plasma', edgecolors='black', linewidth=0.5)

    # Додавання лінії тренду
    z = np.polyfit(x, y, 1)
    p = np.poly1d(z)
    ax.plot(x, p(x), "r--", alpha=0.8, linewidth=2, label=f'Тренд: y = {z[0]:.2f}x + {z[1]:.2f}')

    ax.set_xlabel('X координата', fontsize=14, fontweight='bold')
    ax.set_ylabel('Y координата', fontsize=14, fontweight='bold')
    ax.set_title('Scatter Plot з кольоровою палітрою', fontsize=16, fontweight='bold')
    ax.legend(fontsize=12)

    # Додавання кольорової шкали
    cbar = plt.colorbar(scatter)
    cbar.set_label('Інтенсивність кольору', fontsize=12)

    plt.tight_layout()
    plt.show()


# 3. Кругова діаграма з ефектами
def create_stylish_pie_chart():
    sizes = [25, 30, 15, 20, 10]
    labels = ['Продукт A', 'Продукт B', 'Продукт C', 'Продукт D', 'Продукт E']
    colors = ['#FF9999', '#66B2FF', '#99FF99', '#FFCC99', '#FF99CC']
    explode = (0.1, 0, 0, 0.1, 0)  # виділення секцій

    fig, ax = plt.subplots(figsize=(10, 8))

    wedges, texts, autotexts = ax.pie(sizes, explode=explode, labels=labels, colors=colors,
                                      autopct='%1.1f%%', shadow=True, startangle=90,
                                      textprops={'fontsize': 12, 'fontweight': 'bold'})

    # Додавання тіні та ефектів
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')

    ax.set_title('Розподіл продажів за продуктами', fontsize=16, fontweight='bold', pad=20)

    plt.tight_layout()
    plt.show()


# 4. Heatmap з seaborn
def create_correlation_heatmap():
    # Створення випадкових даних для демонстрації
    data = np.random.randn(10, 12)
    months = ['Січ', 'Лют', 'Бер', 'Кві', 'Тра', 'Чер',
              'Лип', 'Сер', 'Вер', 'Жов', 'Лис', 'Гру']
    years = [f'Рік {i}' for i in range(2014, 2024)]

    df = pd.DataFrame(data, columns=months, index=years)

    plt.figure(figsize=(12, 8))

    # Створення heatmap
    sns.heatmap(df, annot=True, cmap='RdYlBu_r', center=0,
                square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})

    plt.title('Теплова карта даних за роками та місяцями', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Місяці', fontsize=14, fontweight='bold')
    plt.ylabel('Роки', fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.show()


# 5. Інтерактивний графік з Plotly
def create_interactive_plot():
    # Генерація даних
    x = np.linspace(-5, 5, 100)
    y1 = np.sin(x)
    y2 = np.cos(x)
    y3 = np.tan(x / 2)

    fig = go.Figure()

    # Додавання ліній
    fig.add_trace(go.Scatter(x=x, y=y1, mode='lines', name='sin(x)',
                             line=dict(color='red', width=3)))
    fig.add_trace(go.Scatter(x=x, y=y2, mode='lines', name='cos(x)',
                             line=dict(color='blue', width=3)))
    fig.add_trace(go.Scatter(x=x, y=y3, mode='lines', name='tan(x/2)',
                             line=dict(color='green', width=3)))

    # Налаштування макета
    fig.update_layout(
        title='Інтерактивні тригонометричні функції',
        title_font_size=20,
        xaxis_title='X значення',
        yaxis_title='Y значення',
        font=dict(size=14),
        hovermode='x unified',
        template='plotly_dark'
    )

    fig.show()


# 6. 3D поверхня
def create_3d_surface():
    x = np.linspace(-5, 5, 50)
    y = np.linspace(-5, 5, 50)
    X, Y = np.meshgrid(x, y)
    Z = np.sin(np.sqrt(X ** 2 + Y ** 2))

    fig = go.Figure(data=[go.Surface(z=Z, x=X, y=Y, colorscale='Viridis')])

    fig.update_layout(
        title='3D поверхня функції sin(√(x² + y²))',
        title_font_size=20,
        scene=dict(
            xaxis_title='X вісь',
            yaxis_title='Y вісь',
            zaxis_title='Z вісь'
        ),
        font=dict(size=14)
    )

    fig.show()


# 7. Комбінований графік з підграфіками
def create_subplots_dashboard():
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Лінійний графік', 'Гістограма', 'Box plot', 'Scatter plot'),
        specs=[[{"secondary_y": True}, {}],
               [{}, {}]]
    )

    # Дані
    x = np.linspace(0, 10, 50)
    y1 = np.sin(x)
    y2 = np.cos(x)
    data = np.random.randn(100)

    # 1. Лінійний графік
    fig.add_trace(go.Scatter(x=x, y=y1, name='sin(x)'), row=1, col=1)
    fig.add_trace(go.Scatter(x=x, y=y2, name='cos(x)'), row=1, col=1, secondary_y=True)

    # 2. Гістограма
    fig.add_trace(go.Histogram(x=data, name='Розподіл'), row=1, col=2)

    # 3. Box plot
    fig.add_trace(go.Box(y=data, name='Box plot'), row=2, col=1)

    # 4. Scatter plot
    fig.add_trace(go.Scatter(x=np.random.randn(50), y=np.random.randn(50),
                             mode='markers', name='Точки'), row=2, col=2)

    fig.update_layout(height=600, title_text="Дашборд з різними типами графіків")
    fig.show()


# Виконання всіх функцій
if __name__ == "__main__":
    print("🎨 Створення красивих графіків...")

    print("\n1. Лінійний графік з градієнтом:")
    create_gradient_line_plot()

    print("\n2. Scatter plot:")
    create_beautiful_scatter()

    print("\n3. Кругова діаграма:")
    create_stylish_pie_chart()

    print("\n4. Теплова карта:")
    create_correlation_heatmap()

    print("\n5. Інтерактивний графік (Plotly):")
    create_interactive_plot()

    print("\n6. 3D поверхня:")
    create_3d_surface()

    print("\n7. Комбінований дашборд:")
    create_subplots_dashboard()

    print("\n✅ Всі графіки створено успішно!")


# Додаткова функція для швидкого створення простого красивого графіку
def quick_beautiful_plot(x_data, y_data, title="Красивий графік", x_label="X", y_label="Y"):
    """Швидке створення красивого графіку"""
    plt.figure(figsize=(10, 6))
    plt.plot(x_data, y_data, linewidth=3, color='#2E86C1', marker='o', markersize=6,
             markerfacecolor='#F39C12', markeredgecolor='black', markeredgewidth=1)

    plt.title(title, fontsize=16, fontweight='bold', pad=20)
    plt.xlabel(x_label, fontsize=14, fontweight='bold')
    plt.ylabel(y_label, fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3, linestyle='--')

    # Додавання градієнтного фону
    ax = plt.gca()
    ax.set_facecolor('#F8F9FA')

    plt.tight_layout()
    plt.show()

# Приклад використання швидкої функції:
# x = np.linspace(0, 10, 20)
# y = x**2
# quick_beautiful_plot(x, y, "Квадратична функція", "X значення", "Y = X²")