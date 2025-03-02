"""
Файл: main.py
Описание: Точка входа в программу для детекции людей на видео.
"""

import cv2
from ultralytics import YOLO
import os

def detect_and_draw(input_video_path: str, output_video_path: str, model_path: str = "yolo11x.pt"):
    """
    Функция для детекции людей на видео и сохранения результата в новый видеофайл.

    Args:
        input_video_path (str): Путь к входному видеофайлу.
        output_video_path (str): Путь к выходному видеофайлу.
        model_path (str, optional): Путь к файлу модели YOLO. Defaults to "yolo11x.pt".
    """

    # Загрузка предобученной модели YOLO11x.
    # Если файла модели нет, он будет автоматически загружен.
    try:
        model = YOLO(model_path)
    except Exception as e:
        print(f"Ошибка загрузки модели: {e}")
        return

    # Открытие входного видеофайла.
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print(f"Не удалось открыть видеофайл: {input_video_path}")
        return

    # Получение параметров видео (ширина, высота, FPS).
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Определение кодека и создание объекта VideoWriter для записи выходного видео.
    # Используем 'mp4v' для кросс-платформенности (работает на Linux, macOS, Windows).
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))
    if not out.isOpened():
        print(f"Не удалось создать выходной видеофайл: {output_video_path}")
        cap.release()
        return


    # Обработка видео покадрово.
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break  # Выход из цикла, если кадр не был успешно прочитан (конец видео).

        # Детекция людей на текущем кадре.  conf=0.25 - минимальная уверенность.
        results = model(frame, conf=0.25, verbose=False) # verbose=False отключает вывод в консоль

        # Отрисовка результатов на кадре.
        for result in results:
            for box, conf, cls in zip(result.boxes.xyxy, result.boxes.conf, result.boxes.cls):
                x1, y1, x2, y2 = map(int, box)
                confidence = round(float(conf), 2)
                label = f"{result.names[int(cls)]} {confidence}"


                # Отрисовка прямоугольника и подписи.
                # Цвет и толщина подобраны для лучшей видимости.
                color = (0, 255, 0)  # Зеленый цвет (BGR).
                thickness = 2
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

                # Добавляем фон для текста
                text_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                text_w, text_h = text_size
                cv2.rectangle(frame, (x1, y1 - text_h - 10), (x1 + text_w, y1), color, -1)

                cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)


        # Запись обработанного кадра в выходной файл.
        out.write(frame)

    # Освобождение ресурсов.
    cap.release()
    out.release()
    print(f"Обработка завершена. Результат сохранен в {output_video_path}")



if __name__ == "__main__":
    # Проверяем наличие входного видеофайла
    input_file = "crowd.mp4"
    if not os.path.exists(input_file):
        print(f"Ошибка: Файл {input_file} не найден. Поместите видеофайл в директорию со скриптом.")
        exit(1)

    detect_and_draw(input_video_path="crowd.mp4", output_video_path="output.mp4")