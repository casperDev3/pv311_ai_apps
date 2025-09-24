import torch
import torch.nn as nn


# Клас архітектури моделі
class OptimizedCatDetectionCNN(nn.Module):
    def __init__(self, dropout_rate=0.3):
        super(OptimizedCatDetectionCNN, self).__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(dropout_rate / 2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(dropout_rate / 2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(dropout_rate / 2),

            nn.AdaptiveAvgPool2d((4, 4))
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.classifier(x)
        return x


def convert_old_model_to_new(old_model_path, new_model_path):
    """
    Конвертує стару модель у новий безпечний формат
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    try:
        print(f"Завантаження старої моделі: {old_model_path}")

        # Завантажуємо стару модель (небезпечний метод)
        old_checkpoint = torch.load(old_model_path, map_location=device, weights_only=False)

        # Створюємо нову модель
        model = OptimizedCatDetectionCNN()

        # Завантажуємо ваги
        if isinstance(old_checkpoint, dict) and 'model_state_dict' in old_checkpoint:
            model.load_state_dict(old_checkpoint['model_state_dict'])
            metadata = {
                'best_acc': old_checkpoint.get('best_acc', None),
                'epoch': old_checkpoint.get('epoch', None),
                'final_train_accuracy': old_checkpoint.get('final_train_accuracy', None),
                'final_test_accuracy': old_checkpoint.get('final_test_accuracy', None)
            }
        else:
            model.load_state_dict(old_checkpoint)
            metadata = {}

        # Зберігаємо в новому безпечному форматі
        new_save_dict = {
            'model_state_dict': model.state_dict(),
            'model_architecture': 'OptimizedCatDetectionCNN',
            'pytorch_version': torch.__version__,
            **metadata
        }

        # Безпечне збереження
        torch.save(new_save_dict, new_model_path)

        print(f"✅ Модель успішно конвертована!")
        print(f"   Стара модель: {old_model_path}")
        print(f"   Нова модель: {new_model_path}")

        if metadata.get('best_acc'):
            print(f"   Точність: {metadata['best_acc']:.2f}%")

        return True

    except Exception as e:
        print(f"❌ Помилка конвертації: {e}")
        return False


def test_converted_model(model_path):
    """Тестування конвертованої моделі"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    try:
        # Завантаження в новому безпечному форматі
        checkpoint = torch.load(model_path, map_location=device, weights_only=True)

        model = OptimizedCatDetectionCNN()
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()

        print(f"✅ Конвертована модель успішно завантажена!")
        print(f"   PyTorch версія при збереженні: {checkpoint.get('pytorch_version', 'Невідомо')}")

        if 'best_acc' in checkpoint and checkpoint['best_acc']:
            print(f"   Точність: {checkpoint['best_acc']:.2f}%")

        # Простий тест
        test_input = torch.randn(1, 3, 128, 128).to(device)
        with torch.no_grad():
            output = model(test_input)
            probabilities = torch.softmax(output, dim=1)
            print(f"   Тестовий вихід: {probabilities.cpu().numpy()[0]}")

        return True

    except Exception as e:
        print(f"❌ Помилка тестування: {e}")
        return False


if __name__ == '__main__':
    # Список ваших моделей для конвертації
    models_to_convert = [
        ('best_cat_detection_model.pth', 'best_cat_detection_model_safe.pth'),
        ('optimized_cat_detection_model.pth', 'optimized_cat_detection_model_safe.pth'),
        ('checkpoint_epoch_5.pth', 'checkpoint_epoch_5_safe.pth'),
        ('checkpoint_epoch_10.pth', 'checkpoint_epoch_10_safe.pth')
    ]

    print("🔄 Початок конвертації моделей...")

    successful_conversions = []

    for old_path, new_path in models_to_convert:
        print(f"\n{'=' * 60}")

        # Конвертація
        if convert_old_model_to_new(old_path, new_path):
            # Тестування
            if test_converted_model(new_path):
                successful_conversions.append(new_path)

        print('=' * 60)

    print(f"\n🎉 Конвертація завершена!")
    print(f"   Успішно конвертовано: {len(successful_conversions)} моделей")

    if successful_conversions:
        print("\n✅ Готові до використання моделі:")
        for model_path in successful_conversions:
            print(f"   - {model_path}")

        print(f"\nТепер використовуйте будь-яку з цих моделей:")
        print(f"detector = SafeCatDetector('{successful_conversions[0]}')")