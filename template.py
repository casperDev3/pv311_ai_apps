import cv2
import face_recognition
import numpy as np
import time
import os
from pathlib import Path


class FaceRecognitionSystem:
    """Система розпізнавання облич"""

    def __init__(self):
        self.known_face_encodings = []
        self.known_face_names = []
        self.face_locations = []
        self.face_encodings = []
        self.face_names = []

    def load_known_faces(self, photos_folder="my_photos"):
        """
        Завантажує фотографії осіб для розпізнавання

        Args:
            photos_folder: папка з фотографіями (кожна особа в окремій підпапці)

        Структура папок:
        my_photos/
        ├── Ім'я_Особи_1/
        │   ├── photo1.jpg
        │   ├── photo2.jpg
        ├── Ім'я_Особи_2/
        │   ├── photo1.jpg
        """
        print("\n🔄 Завантаження фотографій для навчання...")

        if not os.path.exists(photos_folder):
            print(f"❌ Папка '{photos_folder}' не знайдена!")
            print(f"📁 Створіть папку та додайте підпапки з фото:")
            print(f"   {photos_folder}/Ваше_Імя/photo1.jpg")
            return False

        person_folders = [f for f in Path(photos_folder).iterdir() if f.is_dir()]

        if not person_folders:
            print(f"❌ Не знайдено підпапок з фотографіями в '{photos_folder}'")
            return False

        total_photos = 0

        for person_folder in person_folders:
            person_name = person_folder.name
            photo_files = list(person_folder.glob("*.jpg")) + \
                          list(person_folder.glob("*.jpeg")) + \
                          list(person_folder.glob("*.png"))

            if not photo_files:
                print(f"⚠️  Пропущено '{person_name}': немає фото")
                continue

            print(f"\n👤 Обробка: {person_name}")

            for photo_path in photo_files:
                try:
                    # Завантажуємо зображення
                    image = face_recognition.load_image_file(str(photo_path))

                    # Знаходимо обличчя
                    face_encodings = face_recognition.face_encodings(image)

                    if len(face_encodings) == 0:
                        print(f"   ⚠️  {photo_path.name}: обличчя не знайдено")
                        continue

                    if len(face_encodings) > 1:
                        print(f"   ⚠️  {photo_path.name}: знайдено кілька облич, використано перше")

                    # Додаємо кодування обличчя
                    self.known_face_encodings.append(face_encodings[0])
                    self.known_face_names.append(person_name)
                    total_photos += 1
                    print(f"   ✅ {photo_path.name}")

                except Exception as e:
                    print(f"   ❌ {photo_path.name}: помилка - {e}")

        if total_photos == 0:
            print("\n❌ Не вдалося завантажити жодного обличчя!")
            return False

        print(f"\n✅ Успішно завантажено {total_photos} фото")
        print(f"📊 Розпізнаватиму: {', '.join(set(self.known_face_names))}")
        return True

    def process_frame(self, frame, scale_factor=0.25):
        """
        Обробляє один кадр відео для розпізнавання облич

        Args:
            frame: кадр з відео
            scale_factor: масштаб для прискорення (0.25 = 4x швидше)
        """
        # Зменшуємо розмір для швидшої обробки
        small_frame = cv2.resize(frame, (0, 0), fx=scale_factor, fy=scale_factor)

        # Конвертуємо BGR в RGB
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        # Знаходимо обличчя на кадрі
        self.face_locations = face_recognition.face_locations(rgb_small_frame)
        self.face_encodings = face_recognition.face_encodings(rgb_small_frame, self.face_locations)

        self.face_names = []

        for face_encoding in self.face_encodings:
            # Порівнюємо з відомими обличчями
            matches = face_recognition.compare_faces(
                self.known_face_encodings,
                face_encoding,
                tolerance=0.6  # Чутливість (менше = точніше, але строгіше)
            )
            name = "Невідома особа"
            confidence = 0

            # Використовуємо обличчя з найменшою відстанню
            face_distances = face_recognition.face_distance(
                self.known_face_encodings,
                face_encoding
            )

            if len(face_distances) > 0:
                best_match_index = np.argmin(face_distances)

                if matches[best_match_index]:
                    name = self.known_face_names[best_match_index]
                    # Конвертуємо відстань у впевненість
                    confidence = (1 - face_distances[best_match_index]) * 100

            self.face_names.append((name, confidence))

        # Масштабуємо координати назад
        self.face_locations = [
            (int(top / scale_factor), int(right / scale_factor),
             int(bottom / scale_factor), int(left / scale_factor))
            for (top, right, bottom, left) in self.face_locations
        ]

    def draw_results(self, frame):
        """Малює результати на кадрі"""
        for (top, right, bottom, left), (name, confidence) in zip(
                self.face_locations, self.face_names
        ):
            # Вибираємо колір рамки
            if name == "Невідома особа":
                color = (0, 0, 255)  # Червоний
            else:
                color = (0, 255, 0)  # Зелений

            # Малюємо рамку навколо обличчя
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)

            # Малюємо фон для тексту
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)

            # Малюємо ім'я та впевненість
            font = cv2.FONT_HERSHEY_DUPLEX
            if confidence > 0:
                text = f"{name} ({confidence:.1f}%)"
            else:
                text = name

            cv2.putText(frame, text, (left + 6, bottom - 6),
                        font, 0.6, (255, 255, 255), 1)


def run_face_recognition(camera_id=0, photos_folder="my_photos"):
    """
    Запускає систему розпізнавання облич

    Args:
        camera_id: ID камери
        photos_folder: папка з фотографіями для навчання
    """
    # Ініціалізуємо систему
    system = FaceRecognitionSystem()

    # Завантажуємо відомі обличчя
    if not system.load_known_faces(photos_folder):
        print("\n💡 ІНСТРУКЦІЯ:")
        print("1. Створіть папку 'my_photos'")
        print("2. В ній створіть підпапку зі своїм іменем (наприклад, 'Vasyl')")
        print("3. Додайте туди 3-5 своїх фото (різні ракурси)")
        print("4. Запустіть програму знову")
        return

    # Відкриваємо камеру
    cap = cv2.VideoCapture(camera_id)

    if not cap.isOpened():
        print("❌ Не вдалося відкрити камеру")
        return

    # Налаштування камери
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print("\n" + "=" * 60)
    print("🎥 РОЗПІЗНАВАННЯ ОБЛИЧ - ЗАПУЩЕНО")
    print("=" * 60)
    print("📌 'q' - вихід")
    print("📌 's' - зберегти скріншот")
    print("=" * 60 + "\n")

    fps_start_time = time.time()
    fps_counter = 0
    fps = 0
    screenshot_counter = 0

    # Обробляємо кожен другий кадр для швидкості
    process_this_frame = True

    while True:
        ret, frame = cap.read()

        if not ret:
            print("❌ Помилка читання кадру")
            break

        # Обробляємо кадр
        if process_this_frame:
            system.process_frame(frame, scale_factor=0.25)

        process_this_frame = not process_this_frame

        # Малюємо результати
        system.draw_results(frame)

        # Обчислюємо FPS
        fps_counter += 1
        if time.time() - fps_start_time >= 1.0:
            fps = fps_counter
            fps_counter = 0
            fps_start_time = time.time()

        # Відображаємо FPS
        cv2.rectangle(frame, (5, 5), (150, 35), (0, 0, 0), -1)
        cv2.putText(frame, f"FPS: {fps}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Показуємо кадр
        cv2.imshow('Rozpiznavannya oblich', frame)

        # # Обробка клавіш
        # key = cv2.waitKey(1) & 0xFF
        #
        # if key == ord('q'):
        #     print("\n👋 Завершення роботи...")
        #     break
        # elif key == ord('s'):
        #     screenshot_counter += 1
        #     filename = f"face_screenshot_{screenshot_counter}.jpg"
        #     cv2.imwrite(filename, frame)
        #     print(f"📸 Скріншот збережено: {filename}")

    # Звільняємо ресурси
    cap.release()
    cv2.destroyAllWindows()
    print("✅ Робота завершена")


# ============================================
# ЗАПУСК ПРОГРАМИ
# ============================================

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  СИСТЕМА РОЗПІЗНАВАННЯ ОБЛИЧ")
    print("=" * 60)

    try:
        # Змініть camera_id на потрібний (0, 1, 2, 3...)
        # Змініть photos_folder на шлях до вашої папки з фото
        run_face_recognition(camera_id=0, photos_folder="photos")

    except KeyboardInterrupt:
        print("\n\n👋 Програма перервана користувачем")
    except Exception as e:
        print(f"\n❌ Помилка: {e}")
        import traceback

        traceback.print_exc()