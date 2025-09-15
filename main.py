import numpy as np
from sklearn.datasets import load_digits
import matplotlib.pyplot as mpl
from  sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns

class Perceptron:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.01):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate

        # Ініціалізація ваг з випадковими значеннями
        # Використаємо метод Xavier для ініціалізації ваг
        self.W1 = np.random.randn(self.input_size, self.hidden_size) * np.sqrt(self.input_size)
        self.b1 = np.zeros((1, self.hidden_size))

        self.W2 = np.random.randn(self.hidden_size, self.output_size) * np.sqrt(self.hidden_size)
        self.b2 = np.zeros((1, self.output_size))

        # Список для збереження втрат під час навчання
        self.loss_history = []
        self.accuracy_history = []

    def train(self, X_train, y_train, X_val, y_val, epochs = 1000, verbose= True):
        for epoch in range(epochs):
            # Прямий прохід
            y_pred_train = self.forward_propagation(X_train)
            
            # Обчислення втрат
            loss = self.compute_loss(y_train, y_pred_train)
            self.loss_history.append(loss)

            # Зворотний прохід (оновлення ваг)
            self.backward_propagation(X_train, y_train, y_pred_train)

            # Облчислення точності на валідаційному наборі
            if epoch % 50 == 0:
                y_pred_val = self.forward_propagation(X_val)
                val_accuracy = self.evaluate(X_val, y_val)
                self.accuracy_history.append(val_accuracy)
                if verbose and epoch % 100 == 0:
                    print(f"Епоха {epoch+1}/{epochs} - Втрата: {loss:.4f} - Точність на валідації: {val_accuracy:.4f}")



def one_hot_encode(y, num_classes):
    # 3 ->  [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]

    one_hot = np.zeros((y.shape[0], num_classes))
    one_hot[np.arange(y.shape[0]), y] = 1
    return one_hot

def load_and_prepare_data():
    print("Завагтаження даних...")

    # Завантаження набору даних з цифрами
    digits = load_digits()
    X, y = digits.data, digits.target

    # print(f"Розмір даних: {X.shape}")
    # print(f"Кількість класів: {len(np.unique(y))}")
    # print(f"Розмірність кожного зразка: {X.shape[1]}")

    # Розділення даних на навчальні, валідаційні та тестові набори
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42)  # 0.25 x 0.8 = 0.2

    # Нормалізація даних
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    # Перетворення міток у формат one-hot encoding
    y_train_onehot = one_hot_encode(y_train, num_classes=10)
    y_val_onehot = one_hot_encode(y_val, num_classes=10)
    y_test_onehot = one_hot_encode(y_test, num_classes=10)

    return (
        X_train, y_train_onehot,
        X_val, y_val_onehot,
        X_test, y_test_onehot,
        y_test, scaler  # Збереження оригінальних міток для оцінки
    )

def visualize_samples(X, y, scaler, num_samples=10):
    fig, axes = mpl.subplots(2, 5, figsize=(12, 6))
    fig.suptitle("Приклади цифр з набору даних", fontsize=16)

    # Деанормалізація зразків для візуалізації
    X_denorm = scaler.inverse_transform(X)

    for i in range(num_samples):
        row = i // 5
        col = i % 5

        # Перетворення 1D масиву назад у 2D зображення 8x8
        image = X_denorm[i].reshape(8, 8)
        axes[row, col].imshow(image, cmap='gray')
        axes[row, col].set_title(f"Цифра: {y[i]}")
        axes[row, col].axis('off')

    mpl.show()


def main():
    print(" -- ПЕРЦЕПТРОН ДЛЯ РОЗПІЗНАВАННЯ ЦИФР --")

    # Завантаження та підготовка даних
    X_train, y_train, X_val, y_val, X_test, y_test_onehot, y_test_labels, scaler = load_and_prepare_data()

    # Візуалізація деяких зразків даних
    visualize_samples(X_train, y_test_labels, scaler)

    # Створення та налаштування моделі перцептрона
    input_size = X_train.shape[1] # 64 для зображень 8x8
    hidden_size = 100 # Кількість нейронів у прихованому шарі
    output_size = 10  # Кількість класів (цифр від 0 до 9)
    learning_rate = 0.1 # Швидкість навчання

    # print("Параметри моделі:")
    # print(f" - Розмір вхідного шару: {input_size}")
    # print(f" - Розмір прихованого шару: {hidden_size}")
    # print(f" - Розмір вихідного шару: {output_size}")
    # print(f" - Швидкість навчання: {learning_rate}")


if __name__  == "__main__":
    model = main()
