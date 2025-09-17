import numpy as np
from numpy.ma.core import indices
from sklearn.datasets import load_digits
import matplotlib.pyplot as mpl
from  sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

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

        # Додатковий прихований шар
        self.W2 = np.random.randn(self.hidden_size, self.output_size) * np.sqrt(self.hidden_size)
        self.b2 = np.zeros((1, self.hidden_size))

        # Вихідний шар
        self.W3 = np.random.randn(self.hidden_size, self.output_size) * np.sqrt(self.hidden_size)
        self.b3 = np.zeros((1, self.output_size))


        # Список для збереження втрат під час навчання
        self.loss_history = []
        self.accuracy_history = []

    def predict(self,  X):
        y_pred = self.forward_propagation(X)
        return np.argmax(y_pred, axis=1)

    def sigmoid(self, z):
        z = np.clip(z, -500, 500) # Запобігання переповненню
        return 1 / (1 + np.exp(-z))

    def softmax(self, z):
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def forward_propagation(self, X):
        # Прямий прохід
        self.Z1 = np.dot(X, self.W1) + self.b1 # Лінійне перетворення
        self.A1 = self.sigmoid(self.Z1) # Активація

        self.Z2 = np.dot(self.A1, self.W2) + self.b2  # Лінійне перетворення
        self.A2 = self.sigmoid(self.Z2)  # Активація

        self.Z3 = np.dot(self.A2, self.W3) + self.b3 # Лінійне перетворення
        self.A3 = self.softmax(self.Z3)

        return self.A3

    def compute_loss(self, y_true, y_pred):
        m = y_true.shape[0] # Кількість зразків
        espilon = 1e-15 # Маленьке значення для запобігання log(0)
        y_pred = np.clip(y_pred, espilon, 1 - espilon) # Запобігання log(0)
        loss = -np.sum(y_true * np.log(y_pred)) / m
        return loss

    def sigmoid_derivative(self, A):
        return A * (1 - A)

    def backward_propagation(self, X, y_true, y_pred):
        m = X.shape[0]

        # Обраховуємо помилку вихідного шару
        dZ3 = y_pred - y_true
        dW3 = np.dot(self.A2.T, dZ3) / m
        db3 = np.sum(dZ3, axis=0, keepdims=True) / m

        # Обраховуємо помилку другого прихованого шару
        dA2 = np.dot(dZ3, self.W3.T)
        dZ2 = dA2 * self.sigmoid_derivative(self.A2)  # Похідна сигмоїди
        dW2 = np.dot(self.A1.T, dZ2) / m
        db2 = np.sum(dZ2, axis=0, keepdims=True) / m

        # Обраховуємо помилку прихованого шару
        dA1 = np.dot(dZ2, self.W2.T)
        dZ1 = dA1 * self.sigmoid_derivative(self.A1)  # Похідна сигмоїди
        dW1 = np.dot(X.T, dZ1) / m # Похідна сигмоїди
        db1 = np.sum(dZ1, axis=0, keepdims=True) / m

        # Оновлюємо ваги та зсуви
        self.W3 -= self.learning_rate * dW3
        self.b3 -= self.learning_rate * db3
        self.W2 -= self.learning_rate * dW2
        self.b2 -= self.learning_rate * db2
        self.W1 -= self.learning_rate * dW1
        self.b1 -= self.learning_rate * db2

    def caluclate_accuracy(self, y_true, y_pred):
        y_true_labels = np.argmax(y_true, axis=1)
        y_pred_labels = np.argmax(y_pred, axis=1)
        accuracy = np.mean(y_true_labels == y_pred_labels) # вирахування точності
        return accuracy

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
                val_pred = self.forward_propagation(X_val)
                val_accuracy = self.caluclate_accuracy(y_val, val_pred)
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

def evaluate_model(model, X_test, y_test_onehot, y_test_labels):
    y_pred = model.forward_propagation(X_test)
    test_accuracy = model.caluclate_accuracy(y_test_onehot, y_pred)
    print(f"\nТочність на тестовому наборі: {test_accuracy:.4f}")

    y_pred_labels = np.argmax(y_pred, axis=1)

    print("\nЗвіт про класифікацію:")
    print(classification_report(y_test_labels, y_pred_labels))

    cm = confusion_matrix(y_test_labels, y_pred_labels)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=np.arange(10), yticklabels=np.arange(10))
    plt.xlabel('Передбачені мітки')
    plt.ylabel('Справжні мітки')
    plt.title('Матриця плутанини')
    plt.show()

    return test_accuracy


def show_predictions(model, X_test, y_test_labels, scaler):
    print("Відображення передбачень моделі...")
    predictions = model.predict(X_test)
    X_test_denorm = scaler.inverse_transform(X_test)

    indices = np.random.choice(len(X_test), size=10, replace=False) # Випадковий вибір 10 зразків

    fig, axes = plt.subplots(2, 5, figsize=(18, 8))
    fig.suptitle("Передбачення моделі", fontsize=16)

    for i, idx in enumerate(indices):
        row = i // 5
        col = i % 5

        image = X_test_denorm[idx].reshape(8, 8)
        axes[row, col].imshow(image, cmap='gray')

        correct = predictions[idx] == y_test_labels[idx]
        color = 'green' if correct else 'red'

        axes[row, col].set_title(f"Справжня: {y_test_labels[idx]}\nПередбачена: {predictions[idx]}", color=color)
        axes[row, col].axis('off')

    plt.show()

def main():
    print(" -- ПЕРЦЕПТРОН ДЛЯ РОЗПІЗНАВАННЯ ЦИФР --")

    # Завантаження та підготовка даних
    X_train, y_train, X_val, y_val, X_test, y_test_onehot, y_test_labels, scaler = load_and_prepare_data()

    # Візуалізація деяких зразків даних
    # visualize_samples(X_train, y_test_labels, scaler)

    # Створення та налаштування моделі перцептрона
    input_size = X_train.shape[1] # 64 для зображень 8x8
    hidden_size = 100 # Кількість нейронів у прихованому шарі
    output_size = 10  # Кількість класів (цифр від 0 до 9)
    learning_rate = 0.1 # Швидкість навчання

    print("Параметри моделі:")
    print(f" - Розмір вхідного шару: {input_size}")
    print(f" - Розмір прихованого шару: {hidden_size}")
    print(f" - Розмір вихідного шару: {output_size}")
    print(f" - Швидкість навчання: {learning_rate}")

    # Ініціалізація моделі
    perceptron = Perceptron(input_size, hidden_size, output_size, learning_rate)

    # Навчання моделі
    epochs = 1000
    perceptron.train(X_train, y_train, X_val, y_val, epochs=epochs, verbose=True)

    # Оцінка моделі на тестовому наборі
    evaluate_model(perceptron, X_test, y_test_onehot, y_test_labels)

    # Відображення передбачень
    show_predictions(perceptron, X_test, y_test_labels, scaler)

    return perceptron


if __name__  == "__main__":
    model = main()
