import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import requests
from io import BytesIO
import time


# Архітектура моделі
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


class CatDetector:
    def __init__(self, model_path='best_cat_detection_model.pth', device=None):
        """
        Ініціалізація детектора котиків

        Args:
            model_path: шлях до збереженої моделі
            device: пристрій для обчислень (cuda/cpu)
        """
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self._load_model(model_path)
        self.transform = self._get_transform()

        print(f"Детектор котиків ініціалізовано на пристрої: {self.device}")

    def _load_model(self, model_path):
        """Завантаження збереженої моделі"""
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            model = OptimizedCatDetectionCNN().to(self.device)

            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
                print(f"Модель завантажена з точністю: {checkpoint.get('best_acc', 'невідомо')}%")
            else:
                model.load_state_dict(checkpoint)

            model.eval()
            return model

        except FileNotFoundError:
            print("Збережена модель не знайдена. Створюємо нову модель...")
            print("Спочатку потрібно натренувати модель, запустивши основний скрипт")
            return OptimizedCatDetectionCNN().to(self.device)

    def _get_transform(self):
        """Трансформації для предобробки зображень"""
        return transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def load_image_from_url(self, url, timeout=10):
        """Завантаження зображення з URL"""
        try:
            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
            response = requests.get(url, timeout=timeout, headers=headers)
            response.raise_for_status()
            image = Image.open(BytesIO(response.content)).convert('RGB')
            return image
        except Exception as e:
            print(f"Помилка завантаження {url}: {e}")
            return None

    def predict_single_image(self, image_url):
        """
        Розпізнавання одного зображення з URL

        Args:
            image_url: URL зображення

        Returns:
            tuple: (результат, впевненість, зображення)
        """
        try:
            # Завантаження зображення
            image = self.load_image_from_url(image_url)
            if image is None:
                return None, 0, None

            # Предобробка
            input_tensor = self.transform(image).unsqueeze(0).to(self.device)

            # Передбачення
            with torch.no_grad():
                output = self.model(input_tensor)
                probabilities = torch.softmax(output, dim=1)
                confidence_scores = probabilities.cpu().numpy()[0]
                predicted = torch.argmax(probabilities, dim=1).item()

            # Результати
            is_cat = predicted == 1
            confidence = confidence_scores[predicted] * 100
            cat_probability = confidence_scores[1] * 100

            result = {
                'is_cat': is_cat,
                'confidence': confidence,
                'cat_probability': cat_probability,
                'not_cat_probability': confidence_scores[0] * 100,
                'prediction': 'КОТИК' if is_cat else 'НЕ КОТИК'
            }

            return result, image

        except Exception as e:
            print(f"Помилка при розпізнаванні зображення {image_url}: {e}")
            return None, None

    def predict_batch_urls(self, image_urls, show_results=True, delay=1):
        """
        Пакетна обробка зображень з URL

        Args:
            image_urls: список URL зображень
            show_results: чи показувати результати
            delay: затримка між запитами (секунди)

        Returns:
            list: результати для кожного зображення
        """
        results = []
        successful_downloads = []

        print(f"Початок обробки {len(image_urls)} зображень з інтернету...")
        print("-" * 60)

        for i, url in enumerate(image_urls):
            print(f"\nОбробка зображення {i + 1}/{len(image_urls)}")
            print(f"URL: {url}")

            # Додаємо затримку між запитами
            if i > 0 and delay > 0:
                time.sleep(delay)

            result, image = self.predict_single_image(url)

            if result is not None:
                results.append({
                    'url': url,
                    'result': result,
                    'image': image
                })
                successful_downloads.append((result, image, url))

                print(f"Результат: {result['prediction']}")
                print(f"Впевненість: {result['confidence']:.1f}%")
                print(f"Ймовірність котика: {result['cat_probability']:.1f}%")

                if result['confidence'] > 80:
                    print("Висока впевненість")
                elif result['confidence'] > 60:
                    print("Середня впевненість")
                else:
                    print("Низька впевненість")
            else:
                print("Не вдалося обробити зображення")

        # Статистика
        if results:
            cats_found = sum(1 for r in results if r['result']['is_cat'])
            avg_confidence = sum(r['result']['confidence'] for r in results) / len(results)

            print(f"\n" + "=" * 60)
            print("СТАТИСТИКА ОБРОБКИ:")
            print(f"Всього URL: {len(image_urls)}")
            print(f"Успішно завантажено: {len(results)}")
            print(f"Знайдено котиків: {cats_found}")
            print(f"Не котики: {len(results) - cats_found}")
            print(f"Відсоток котиків: {(cats_found / len(results) * 100):.1f}%")
            print(f"Середня впевненість: {avg_confidence:.1f}%")
            print("=" * 60)

        # Відображення результатів
        if show_results and successful_downloads:
            self._show_batch_results(successful_downloads)

        return results

    def _show_batch_results(self, results_data):
        """Відображення результатів пакетної обробки"""
        if not results_data:
            return

        # Обмежуємо кількість зображень для відображення
        n_images = min(len(results_data), 12)
        cols = 4
        rows = (n_images + cols - 1) // cols

        plt.figure(figsize=(16, 4 * rows))

        for i in range(n_images):
            try:
                result, image, url = results_data[i]

                plt.subplot(rows, cols, i + 1)
                plt.imshow(image)
                plt.axis('off')

                # Назва з результатом
                color = 'green' if result['is_cat'] else 'red'
                title = f"{result['prediction']}\n{result['confidence']:.1f}%"

                # Додаємо частину URL для ідентифікації
                url_part = url.split('/')[-1][:15] + "..." if len(url.split('/')[-1]) > 15 else url.split('/')[-1]
                title += f"\n{url_part}"

                plt.title(title, fontsize=9, color=color, weight='bold')

            except Exception as e:
                print(f"Помилка відображення зображення {i}: {e}")

        plt.suptitle('Результати розпізнавання котиків з інтернету', fontsize=16, y=0.98)
        plt.tight_layout()
        plt.show()


def demo_batch_processing():
    """Демонстрація пакетної обробки зображень з інтернету"""

    # Ініціалізація детектора
    detector = CatDetector()

    # Список URL зображень для тестування
    test_urls = [
        # Котики
        "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Cat03.jpg/1200px-Cat03.jpg",
        "https://upload.wikimedia.org/wikipedia/commons/thumb/1/15/Cat_August_2010-4.jpg/1200px-Cat_August_2010-4.jpg",
        "https://upload.wikimedia.org/wikipedia/commons/thumb/5/5e/Sleeping_cat_on_her_back.jpg/1200px-Sleeping_cat_on_her_back.jpg",
        "https://upload.wikimedia.org/wikipedia/commons/thumb/4/4d/Cat_March_2010-1.jpg/1200px-Cat_March_2010-1.jpg",
        "https://upload.wikimedia.org/wikipedia/commons/thumb/6/68/Orange_tabby_cat_sitting_on_fallen_leaves-Hisashi-01A.jpg/1200px-Orange_tabby_cat_sitting_on_fallen_leaves-Hisashi-01A.jpg",

        # Не котики
        "https://upload.wikimedia.org/wikipedia/commons/thumb/d/d9/Collage_of_Nine_Dogs.jpg/1200px-Collage_of_Nine_Dogs.jpg",
        "https://upload.wikimedia.org/wikipedia/commons/thumb/5/50/Vd-Orig.png/256px-Vd-Orig.png",
        "https://upload.wikimedia.org/wikipedia/commons/thumb/f/fb/Welkersdorf_Tanzlinde.jpg/1200px-Welkersdorf_Tanzlinde.jpg",
        "https://upload.wikimedia.org/wikipedia/commons/thumb/b/b6/Image_created_with_a_mobile_phone.png/1200px-Image_created_with_a_mobile_phone.png",

        # Більше котиків
        "https://upload.wikimedia.org/wikipedia/commons/thumb/9/9b/Gustav_chocolate.jpg/1200px-Gustav_chocolate.jpg",
        "https://upload.wikimedia.org/wikipedia/commons/thumb/b/bb/Kittyply_edit1.jpg/1200px-Kittyply_edit1.jpg",
        "https://upload.wikimedia.org/wikipedia/commons/thumb/0/0b/Cat_poster_1.jpg/1200px-Cat_poster_1.jpg"
    ]

    print("ДЕМОНСТРАЦІЯ ПАКЕТНОЇ ОБРОБКИ ЗОБРАЖЕНЬ З ІНТЕРНЕТУ")
    print("=" * 60)
    print(f"Буде оброблено {len(test_urls)} зображень")
    print("Затримка між запитами: 1 секунда")
    print("-" * 60)

    # Запуск пакетної обробки
    results = detector.predict_batch_urls(
        image_urls=test_urls,
        show_results=True,
        delay=1  # 1 секунда затримки між запитами
    )

    # Додаткова статистика
    if results:
        print(f"\nДетальна статистика:")

        cats = [r for r in results if r['result']['is_cat']]
        non_cats = [r for r in results if not r['result']['is_cat']]

        if cats:
            avg_cat_confidence = sum(r['result']['confidence'] for r in cats) / len(cats)
            print(f"Середня впевненість для котиків: {avg_cat_confidence:.1f}%")

        if non_cats:
            avg_non_cat_confidence = sum(r['result']['confidence'] for r in non_cats) / len(non_cats)
            print(f"Середня впевненість для не-котиків: {avg_non_cat_confidence:.1f}%")

        # Показати найвпевненіші результати
        sorted_results = sorted(results, key=lambda x: x['result']['confidence'], reverse=True)

        print(f"\nТоп-3 найвпевненіші результати:")
        for i, item in enumerate(sorted_results[:3]):
            result = item['result']
            url_name = item['url'].split('/')[-1][:30]
            print(f"{i + 1}. {result['prediction']} ({result['confidence']:.1f}%) - {url_name}")


if __name__ == "__main__":
    demo_batch_processing()