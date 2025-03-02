import cv2
from ultralytics import YOLO
import os
import time
import multiprocessing
from multiprocessing import Pool, Manager
import numpy as np
from filterpy.kalman import KalmanFilter
from typing import List, Tuple, Dict, Any
import logging
import torch
from functools import partial

# Конфигурация логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(processName)s - %(levelname)s - %(message)s')


def apply_kalman_filter(bbox: np.ndarray, kf: KalmanFilter, dt: float) -> np.ndarray:
    """Применяет фильтр Калмана с учетом времени между кадрами."""
    kf.F = np.array([[1, 0, 0, 0, dt, 0, 0, 0],
                     [0, 1, 0, 0, 0, dt, 0, 0],
                     [0, 0, 1, 0, 0, 0, dt, 0],
                     [0, 0, 0, 1, 0, 0, 0, dt],
                     [0, 0, 0, 0, 1, 0, 0, 0],
                     [0, 0, 0, 0, 0, 1, 0, 0],
                     [0, 0, 0, 0, 0, 0, 1, 0],
                     [0, 0, 0, 0, 0, 0, 0, 1]])
    kf.predict()
    kf.update(bbox)
    return kf.x[:4].flatten()


def initialize_kalman_filter() -> KalmanFilter:
    """Инициализирует фильтр Калмана."""
    kf = KalmanFilter(dim_x=8, dim_z=4)
    dt = 1  # Начальное значение
    kf.F = np.array([[1, 0, 0, 0, dt, 0, 0, 0],
                     [0, 1, 0, 0, 0, dt, 0, 0],
                     [0, 0, 1, 0, 0, 0, dt, 0],
                     [0, 0, 0, 1, 0, 0, 0, dt],
                     [0, 0, 0, 0, 1, 0, 0, 0],
                     [0, 0, 0, 0, 0, 1, 0, 0],
                     [0, 0, 0, 0, 0, 0, 1, 0],
                     [0, 0, 0, 0, 0, 0, 0, 1]])

    kf.H = np.array([[1, 0, 0, 0, 0, 0, 0, 0],
                     [0, 1, 0, 0, 0, 0, 0, 0],
                     [0, 0, 1, 0, 0, 0, 0, 0],
                     [0, 0, 0, 1, 0, 0, 0, 0]])

    kf.R = np.eye(4) * 12
    kf.Q = np.array([[dt**4/4, 0, 0, 0, dt**3/2, 0, 0, 0],
                     [0, dt**4/4, 0, 0, 0, dt**3/2, 0, 0],
                     [0, 0, dt**4/4, 0, 0, 0, dt**3/2, 0],
                     [0, 0, 0, dt**4/4, 0, 0, 0, dt**3/2],
                     [dt**3/2, 0, 0, 0, dt**2, 0, 0, 0],
                     [0, dt**3/2, 0, 0, 0, dt**2, 0, 0],
                     [0, 0, dt**3/2, 0, 0, 0, dt**2, 0],
                     [0, 0, 0, dt**3/2, 0, 0, 0, dt**2]]) * 0.01
    kf.P *= 100.
    return kf

def process_frame(frame_data: Tuple[int, np.ndarray, float], model: YOLO, kalman_filters: Dict, iou_threshold:float, use_soft_nms:bool) -> Tuple[int, np.ndarray]:
    """Обрабатывает один кадр, обертка для использования с Pool.imap_unordered"""
    frame_num, frame, dt = frame_data
    try:
        with torch.no_grad():
            results = model.predict(frame, conf=0.2, verbose=False, classes=[0], iou=iou_threshold)[0]

        detections = []
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            conf = float(box.conf[0].cpu().numpy())
            detections.append((x1, y1, x2, y2, conf))
        detections = np.array(detections)


        if len(detections) > 0:
            if use_soft_nms:
                dets = soft_nms(detections, Nt=iou_threshold, sigma=0.3, method=2)
            else:
                indices = cv2.dnn.NMSBoxes(detections[:, :4].tolist(), detections[:, 4].tolist(), 0.2, iou_threshold)
                dets = detections[indices] if len(indices) > 0 else np.array([])
        else:
            dets = np.array([])

        current_ids = set()
        processed_frame = frame.copy()

        for det in dets:
            x1, y1, x2, y2, confidence = det
            bbox = np.array([x1, y1, x2 - x1, y2 - y1])
            track_id = None

            max_iou = 0
            best_id = None
            for existing_id, kf in kalman_filters.items():
                prev_bbox = kf.x[:4].flatten()
                prev_x1, prev_y1, prev_w, prev_h = prev_bbox
                prev_x2, prev_y2 = prev_x1 + prev_w, prev_y1 + prev_h
                iou = calculate_iou((x1, y1, x2, y2), (int(prev_x1), int(prev_y1), int(prev_x2), int(prev_y2)))
                if iou > max_iou:
                    max_iou = iou
                    best_id = existing_id

            if max_iou > 0.35:
                track_id = best_id
                filtered_bbox = apply_kalman_filter(bbox, kalman_filters[track_id], dt)
            else:
                track_id = len(kalman_filters) if len(kalman_filters) >0 else 0 #счетчик с нуля
                kf = initialize_kalman_filter()
                kf.x[:4] = np.array([x1, y1, x2 - x1, y2 - y1]).reshape(-1, 1)
                kalman_filters[track_id] = kf
                filtered_bbox = apply_kalman_filter(bbox, kf, dt)

            current_ids.add(track_id)
            fx1, fy1, fw, fh = map(int, filtered_bbox)
            fx2, fy2 = fx1 + fw, fy1 + fh

            label = f"person {confidence:.2f}"
            color = (0, 255, 0)
            thickness = 2
            cv2.rectangle(processed_frame, (fx1, fy1), (fx2, fy2), color, thickness)
            text_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            text_w, text_h = text_size
            cv2.rectangle(processed_frame, (fx1, fy1 - text_h - 10), (fx1 + text_w, fy1), color, -1)
            cv2.putText(processed_frame, label, (fx1, fy1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

        # ভালোভাবে ট্র্যাক রাখতে, বর্তমান ফ্রেমে শনাক্ত হওয়া আইডিগুলি একটি সেটে রাখি
        ids_to_remove = set(kalman_filters.keys()) - current_ids
        for track_id in ids_to_remove:
            del kalman_filters[track_id]

        return frame_num, processed_frame

    except Exception as e:
        logging.exception(f"process_frame ফাংশনে ত্রুটি (ফ্রেম {frame_num}): {e}")
        return frame_num, None # সমস্যার ক্ষেত্রে None ফেরত দিন


def group_detections(detections, iou_threshold=0.5):
    """Группирует близкие детекции."""
    grouped_dets = []
    assigned = [False] * len(detections)

    for i, det1 in enumerate(detections):
        if assigned[i]:
            continue

        group = [det1]
        assigned[i] = True
        x1_1, y1_1, x2_1, y2_1 = det1.xyxy[0].cpu().numpy()

        for j, det2 in enumerate(detections):
            if i != j and not assigned[j]:
                x1_2, y1_2, x2_2, y2_2 = det2.xyxy[0].cpu().numpy()
                iou = calculate_iou((x1_1, y1_1, x2_1, y2_1), (x1_2, y1_2, x2_2, y2_2))
                if iou > iou_threshold:
                    group.append(det2)
                    assigned[j] = True
        grouped_dets.append(group)
    return grouped_dets


def calculate_iou(box1: Tuple, box2: Tuple) -> float:
    """Вычисляет IoU."""
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    x_left = max(x1_1, x1_2)
    y_top = max(y1_1, y1_2)
    x_right = min(x2_1, x2_2)
    y_bottom = min(y2_1, y2_2)
    if x_right < x_left or y_bottom < y_top:
        return 0.0
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
    box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
    iou = intersection_area / float(box1_area + box2_area - intersection_area)
    return iou


def soft_nms(dets: np.ndarray, sigma: float = 0.5, Nt: float = 0.3, method: int = 2) -> np.ndarray:
    """Soft-NMS."""
    N = dets.shape[0]
    indexes = np.array(np.arange(N))
    for i in range(N):
        max_pos = i
        max_score = dets[i, 4]
        tx1, ty1, tx2, ty2 = dets[i, 0], dets[i, 1], dets[i, 2], dets[i, 3]
        ti = indexes[i]

        pos = i + 1
        while pos < N:
            if max_score < dets[pos, 4]:
                max_score = dets[pos, 4]
                max_pos = pos
            pos += 1

        dets[i, :], dets[max_pos, :] = dets[max_pos, :].copy(), dets[i, :].copy()
        indexes[i], indexes[max_pos] = indexes[max_pos], indexes[i]

        tx1, ty1, tx2, ty2 = dets[i, 0], dets[i, 1], dets[i, 2], dets[i, 3]
        ti = indexes[i]

        pos = i + 1
        while pos < N:
            x1, y1, x2, y2 = dets[pos, 0], dets[pos, 1], dets[pos, 2], dets[pos, 3]
            iou = calculate_iou((tx1, ty1, tx2, ty2), (x1, y1, x2, y2))

            if method == 1:  # linear
                if iou > Nt:
                    weight = 1 - iou
                else:
                    weight = 1
            elif method == 2:  # gaussian
                weight = np.exp(-(iou * iou) / sigma)
            else:  # original NMS (method == 3)
                if iou > Nt:
                    weight = 0
                else:
                    weight = 1

            dets[pos, 4] = dets[pos, 4] * weight
            if dets[pos, 4] < 0.001:
                dets[pos, :], dets[N - 1, :] = dets[N - 1, :].copy(), dets[pos, :].copy()
                indexes[pos], indexes[N - 1] = indexes[N - 1], indexes[pos]
                N -= 1
                pos -= 1
            pos += 1

    keep = [i for i in range(N)]
    return dets[keep]


def detect_and_track_parallel(input_video_path: str, output_video_path: str, model_paths: List[str] = ["yolo11m.pt"],
                              num_processes: int = None, iou_threshold: float = 0.4,
                              use_soft_nms: bool = True, output_format: str = "mp4v"):
    """Детектирует и отслеживает (сглаживает) людей на видео."""

    logging.info("Запуск detect_and_track_parallel...")
    logging.info(f"Входной файл: {input_video_path}")
    logging.info(f"Выходной файл: {output_video_path}")
    logging.info(f"Модели: {model_paths}")

    if not os.path.exists(input_video_path):
        logging.error(f"Ошибка: Файл {input_video_path} не найден.")
        return

    logging.info("Открытие видеофайла...")
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        logging.error(f"Не удалось открыть видео: {input_video_path}")
        return

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    logging.info(f"Разрешение видео: {frame_width}x{frame_height}, FPS: {fps}, Кадров: {total_frames}")

    # Выбор кодека
    if output_format == 'mp4v':
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    elif output_format == 'avc1':  # H.264
        fourcc = cv2.VideoWriter_fourcc(*'avc1')
    elif output_format == 'xvid':
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
    else:
        logging.warning(f"Неподдерживаемый формат выходного файла: {output_format}.  Используется mp4v.")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    logging.info(f"Создание выходного файла: {output_video_path}")
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height)) # Исходный размер
    if not out.isOpened():
        logging.error(f"Не удалось создать выходной файл: {output_video_path}")
        cap.release()
        return

    # Автоопределение количества процессов, если не задано
    if num_processes is None:
        num_processes = multiprocessing.cpu_count()
    logging.info(f"Используется {num_processes} процессов.")

    # --- ЗАГРУЗКА МОДЕЛЕЙ (в главном процессе) ---
    logging.info("Загрузка моделей YOLO...")
    models = [YOLO(mp) for mp in model_paths]
    logging.info("Модели YOLO загружены.")
    # ----------------------------------------------

    manager = Manager()  #  Manager нужен для общего словаря kalman_filters
    kalman_filters = manager.dict()  # Инициализируем словарь фильтров Калмана
    start_time = time.time()
    last_log_time = start_time

    # Подготовка данных для пула процессов
    frame_num = 0
    frames_data = []  # Список кортежей (номер кадра, кадр, dt)
    prev_time = start_time
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        current_time = time.time()
        dt = current_time - prev_time
        prev_time = current_time
        frames_data.append((frame_num, frame, dt))
        frame_num += 1

    # --- ИСПОЛЬЗОВАНИЕ Pool.imap_unordered ---
    with Pool(processes=num_processes) as pool:
        # Создаем partial функцию, фиксируя model, kalman_filters и т.д.
        # models[0] - берем первую модель из списка. Если у вас ансамблирование, нужно изменить.
        process_func = partial(process_frame, model=models[0], kalman_filters=kalman_filters,
                                iou_threshold=iou_threshold, use_soft_nms=use_soft_nms)

        # results - итератор, который возвращает результаты в *порядке их завершения*
        results = pool.imap_unordered(process_func, frames_data)
        # model.predict(frames, stream=True) -  для обработки видео в реальном времени

        logging.info("Начало обработки видео...")
        processed_count = 0
        for frame_num, processed_frame in results:  # Итерируемся по результатам *в порядке готовности*
            if processed_frame is not None:
                out.write(processed_frame)
            processed_count += 1

            # Логирование прогресса
            current_time = time.time()
            if current_time - last_log_time >= 3:
                elapsed_time = current_time - start_time
                if processed_count > 0 and fps > 0:
                    remaining_frames = total_frames - processed_count
                    time_per_frame = elapsed_time / processed_count
                    estimated_remaining_time = remaining_frames * time_per_frame
                    logging.info(f"Обработано: {processed_count}/{total_frames}. Осталось: {estimated_remaining_time:.2f} сек.")
                else:
                    logging.info(f"Обработано: {processed_count}/{total_frames}. Оценка времени пока недоступна.")
                last_log_time = current_time
    # ------------------------------------------

    logging.info(f"Общее время: {time.time() - start_time:.2f} сек.")
    cap.release()
    out.release()
    logging.info(f"Готово. Результат в {output_video_path}")


if __name__ == "__main__":
    # --- Параметры ---
    input_file = "crowd.mp4"  # Входной видеофайл
    output_file = "output.mp4" # Выходной видеофайл
    model_paths = ["yolo11m.pt"]  #  "n" слишком маленькая для этого видео, "m" - как раз
    # model_paths = ["yolo11s.pt", "yolo11n.pt"] # Ансамблирование (пример)
    num_processes = None        # Количество процессов. None = автоопределение
    iou_threshold = 0.4   # Порог IoU для NMS/Soft-NMS и фильтра Калмана
    use_soft_nms = True  # Использовать Soft-NMS (True/False)
    output_format = "mp4v"  # Формат выходного видео ('mp4v', 'avc1', 'xvid')

    detect_and_track_parallel(input_video_path=input_file, output_video_path=output_file,
                              model_paths=model_paths, num_processes=num_processes,
                              iou_threshold=iou_threshold, use_soft_nms=use_soft_nms, output_format=output_format)