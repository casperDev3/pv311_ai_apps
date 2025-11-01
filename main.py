import cv2 # Малювання і обробка зображень
import numpy as np # Робота з масивами даних
import time # Робота з часом
import os # Робота з операційною системою
from pathlib import Path # Робота з файловими шляхами
import face_recognition # Розпізнавання облич
import traceback # Відстеження помилок

from face_recognition import face_locations
from ultralytics import YOLO # Модель YOLO для обробки зображень
import threading # Багатопоточність

class FrameGrabber(threading.Thread):
    def __init__(self, src=0, width=640, height=480):
        super().__init__()
        self.capture = cv2.VideoCapture(src)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.lock = threading.Lock()
        self.latest_frame = None
        self.running = True
        if not self.capture.isOpened():
            raise ValueError("Unable to open video source", src)

    def run(self):
        while self.running:
            ret, frame = self.capture.read()
            if not ret:
                continue
            with self.lock:
                self.latest_frame = frame

    def read(self):
        with self.lock:
            frame_copy = self.latest_frame.copy() if self.latest_frame is not None else None
        return frame_copy

    def stop(self):
        self.running = False
        self.capture.release()

class FaceRecognitionSystem:
    def __init__(self):
        self.known_faces_encodings = []
        self.known_faces_names = []
        self.face_locations = []
        self.face_encodings = []
        self.face_names = []
        self.people_boxes = []

        print("Завантаження моделі YOLOv8n...")
        self.person_detector = YOLO("yolov8n.pt")
        self.person_detector.to("cuda" if cv2.cuda.getCudaEnabledDeviceCount() > 0 else "cpu")

        self.frame_count = 0
        self.lock = threading.Lock()

    def load_known_faces(self,  directory="known_faces"):
        print("Завантаження відомих облич...")
        if not os.path.exists(directory):
            print("Директорія відомих облич не знайдена!")
            return False
        person_folders = [f.path for f in os.scandir(directory) if f.is_dir()]
        if not person_folders:
            print("Директорія відомих облич порожня!")
            return False

        total_photos = 0
        for person_folder in person_folders:
            person_name = os.path.basename(person_folder)
            photo_files = [f.path for f in os.scandir(person_folder) if f.is_file() and f.name.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))]
            for photo_file in photo_files:
                try:
                    image = face_recognition.load_image_file(photo_file)
                    enc = face_recognition.face_encodings(image)
                    if len(enc) == 0:
                        print(f"Обличчя не знайдено на фото: {photo_file}")
                        continue
                    self.known_faces_encodings.append(enc[0])
                    self.known_faces_names.append(person_name)
                    total_photos += 1
                except Exception as e:
                    print("Помилка при обробці фото:", photo_file)

        print("Завантажено осіб:", len(self.known_faces_names))
        print("Завантажено фото:", total_photos)
        return total_photos > 0

    def process_frame(self, frame, face_interval=3, scale_factor=0.5):
        """"Обробка кадру для виявлення та розпізнавання облич"""
        people_boxes = []

        """"Виявлення людей за допомогою YOLOv8n"""
        small_for_yolo = cv2.resize(frame, (900, 600))
        results = self.person_detector(small_for_yolo, classes=[0], conf=0.5, verbose=False)

        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                h_ratio = frame.shape[0] / 600
                w_ratio = frame.shape[1] / 900
                people_boxes.append((int(x1 * w_ratio), int(y1 * h_ratio), int(x2 * w_ratio), int(y2 * h_ratio)))

        """"Пошук облич"""""
        if self.frame_count % face_interval == 0:
            small_frame = cv2.resize(frame, (0, 0), fx=scale_factor, fy=scale_factor)
            rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

            face_names = []
            for face_encoding in face_encodings:
                matches = face_recognition.compare_faces(self.known_faces_encodings, face_encoding)
                name = "Unknown"
                confidence = 0
                face_distances = face_recognition.face_distance(self.known_faces_encodings, face_encoding)
                if len(face_distances) > 0:
                    best_match_index = np.argmin(face_distances)
                    if matches[best_match_index]:
                        name = self.known_faces_names[best_match_index]
                        confidence = 1 - face_distances[best_match_index]
                face_names.append((name, confidence))

            face_locations = [
                (int(t / scale_factor), int(r / scale_factor), int(b / scale_factor), int(l / scale_factor))
                for (t, r, b, l) in face_locations
            ]

            with self.lock:
                self.face_locations = face_locations
                self.face_names = face_names

        with self.lock:
            self.people_boxes = people_boxes

        self.frame_count += 1

    def draw_results(self, frame):
        with self.lock:
            people_boxes = self.people_boxes.copy()
            face_locations = self.face_locations.copy()
            face_names = self.face_names.copy()

        for (x1, y1, x2, y2)  in people_boxes:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 200, 0), 2)
            cv2.putText(frame, "Person", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 0), 2)

        for (top, right, bottom, left), (name, confidence) in zip(face_locations, face_names):
            color = (0, 255, 0) if name != "Unknown" else (0, 0, 0)
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)

            label = f"{name} ({confidence*100:.1f}%)" if name != "Unknown" else name
            cv2.putText(frame, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

def run_face_recognition(camera_id = 0, photos_folder = "known_faces"):
    system = FaceRecognitionSystem()
    if not system.load_known_faces(photos_folder):
        print("Не вдалося завантажити відомі обличчя. Завершення роботи.")
        return

    grabber = FrameGrabber(camera_id)
    grabber.start()

    print("Початок розпізнавання облич. Натисніть 'q' для виходу.")
    process_thread = None
    fps = 0
    fps_start = time.time()
    fps_counter = 0


    try:
        while True:
            frame = grabber.read()
            if frame is None:
                continue

            """Якщо попередні потік завершився, запускаємо новий"""
            if process_thread is None or not process_thread.is_alive():
                process_thread = threading.Thread(target=system.process_frame, args=(frame,))
                process_thread.start()

            system.draw_results(frame)

            fps_counter += 1
            if time.time() - fps_start >= 1.0:
                fps = fps_counter
                fps_counter = 0
                fps_start = time.time()

            cv2.rectangle(frame, (5, 5), (120, 35), (0, 0, 0), -1)
            cv2.putText(frame, f"FPS: {fps}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.imshow("Face Recognition System", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    except Exception as e:
        print("Виникла помилка:", e)
        traceback.print_exc()
    except KeyboardInterrupt:
        print("Завершення роботи користувачем.")
    finally:
        grabber.stop()
        grabber.join()
        cv2.destroyAllWindows()



if __name__ == "__main__":
    try:
        run_face_recognition(camera_id=4, photos_folder="known_faces")
    except Exception  as e:
        print("Виникла критична помилка:", e)
        traceback.print_exc()
