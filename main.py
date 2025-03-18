import cv2
from ultralytics import YOLO
import os
import time
import multiprocessing
from multiprocessing import Pool, Manager
import numpy as np
from filterpy.kalman import KalmanFilter
from typing import List, Tuple, Dict
import logging
import torch
from functools import partial
import argparse

# Конфигурация логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(processName)s - %(levelname)s - %(message)s')


def apply_kalman_filter(bbox: np.ndarray, kf: KalmanFilter, dt: float) -> np.ndarray:
    """
    Применяет фильтр Калмана к bounding box с учетом времени между кадрами.
    
    Args:
        bbox: Координаты bounding box [x, y, w, h]
        kf: Инициализированный фильтр Калмана
        dt: Время между кадрами в секундах
        
    Returns:
        Сглаженные координаты bounding box
    """
    # Ограничиваем dt для предотвращения больших скачков
    dt = min(dt, 0.05)  # Еще более строгое ограничение
    
    # Обновляем матрицу перехода с учетом времени
    kf.F = np.array([[1, 0, 0, 0, dt, 0, 0, 0],
                     [0, 1, 0, 0, 0, dt, 0, 0],
                     [0, 0, 1, 0, 0, 0, dt, 0],
                     [0, 0, 0, 1, 0, 0, 0, dt],
                     [0, 0, 0, 0, 0.1, 0, 0, 0],  # Минимальное сохранение скорости
                     [0, 0, 0, 0, 0, 0.1, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0.1, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0.1]])
    
    # Предсказание следующего состояния (с минимальным шагом)
    kf.predict()
    
    # Используем практически только измерения
    predicted = kf.x[:4].flatten()
    alpha = 0.05  # Используем всего 5% от предсказания, 95% от измерения
    blended_bbox = alpha * predicted + (1 - alpha) * bbox
    
    # Обновление с использованием смешанного значения
    kf.update(blended_bbox)
    
    # Получаем сглаженные координаты
    smoothed = kf.x[:4].flatten()
    
    # Принудительно сдвигаем ближе к измеренной позиции для компенсации любого смещения
    smoothed[:2] = 0.9 * bbox[:2] + 0.1 * smoothed[:2]
    
    # Для размеров используем в основном текущие измерения
    smoothed[2:] = 0.8 * bbox[2:] + 0.2 * smoothed[2:]
    
    # Обеспечиваем минимальный размер
    smoothed[2] = max(smoothed[2], 10)  # Минимальная ширина 10 пикселей
    smoothed[3] = max(smoothed[3], 20)  # Минимальная высота 20 пикселей
    
    return smoothed


def initialize_kalman_filter() -> KalmanFilter:
    """
    Инициализирует фильтр Калмана для трекинга объектов.
    
    Returns:
        Инициализированный объект KalmanFilter
    """
    kf = KalmanFilter(dim_x=8, dim_z=4)
    dt = 0.05  # Еще сильнее уменьшаем начальное значение времени
    kf.F = np.array([[1, 0, 0, 0, dt, 0, 0, 0],
                     [0, 1, 0, 0, 0, dt, 0, 0],
                     [0, 0, 1, 0, 0, 0, dt, 0],
                     [0, 0, 0, 1, 0, 0, 0, dt],
                     [0, 0, 0, 0, 0.1, 0, 0, 0],  # Практически убираем сохранение скорости
                     [0, 0, 0, 0, 0, 0.1, 0, 0],  # Практически убираем сохранение скорости
                     [0, 0, 0, 0, 0, 0, 0.1, 0],  # Практически убираем сохранение скорости
                     [0, 0, 0, 0, 0, 0, 0, 0.1]]) # Практически убираем сохранение скорости

    kf.H = np.array([[1, 0, 0, 0, 0, 0, 0, 0],
                     [0, 1, 0, 0, 0, 0, 0, 0],
                     [0, 0, 1, 0, 0, 0, 0, 0],
                     [0, 0, 0, 1, 0, 0, 0, 0]])

    # Настраиваем фильтр для минимального прогнозирования и максимальной опоры на измерения
    kf.R = np.eye(4) * 0.1    # Почти не учитываем шум измерений - полностью доверяем измерениям
    kf.Q = np.array([[dt**4/4, 0, 0, 0, dt**3/2, 0, 0, 0],
                     [0, dt**4/4, 0, 0, 0, dt**3/2, 0, 0],
                     [0, 0, dt**4/4, 0, 0, 0, dt**3/2, 0],
                     [0, 0, 0, dt**4/4, 0, 0, 0, dt**3/2],
                     [dt**3/2, 0, 0, 0, dt**2, 0, 0, 0],
                     [0, dt**3/2, 0, 0, 0, dt**2, 0, 0],
                     [0, 0, dt**3/2, 0, 0, 0, dt**2, 0],
                     [0, 0, 0, dt**3/2, 0, 0, 0, dt**2]]) * 0.0001  # Минимальный шум процесса
    
    # Начальная ковариация - минимальная для стабильности
    kf.P = np.eye(8) * 1.0
    
    return kf

def process_frame(frame_data: Tuple[int, np.ndarray, float], models: List[YOLO], kalman_filters: Dict, iou_threshold: float, use_soft_nms: bool) -> Tuple[int, np.ndarray]:
    """
    Обрабатывает один кадр видео с детекцией и трекингом объектов.
    
    Args:
        frame_data: Кортеж (индекс_кадра, кадр, время_между_кадрами)
        models: Список моделей YOLO
        kalman_filters: Словарь с фильтрами Калмана для трекинга
        iou_threshold: Порог IoU для NMS
        use_soft_nms: Использовать ли Soft-NMS
        
    Returns:
        Кортеж (индекс_кадра, обработанный_кадр)
    """
    frame_idx, frame, dt = frame_data
    try:
        # Детекция объектов - с использованием ансамблирования если передано >1 модели
        if len(models) > 1:
            detections = ensemble_predictions(models, frame)
        else:
            detections = detect_objects(models[0], frame)
        
        if len(detections) > 0:
            # Применяем NMS или Soft-NMS для уменьшения перекрывающихся боксов
            if use_soft_nms:
                detections = soft_nms(detections, Nt=iou_threshold)
            else:
                # Используем стандартный NMS через группировку
                detections = group_detections(detections, iou_threshold)
            
            # Применяем трекинг
            tracked_objects = apply_tracking(detections, kalman_filters, frame_idx, dt)
            
            # Сортируем объекты по уверенности в порядке убывания
            tracked_objects.sort(key=lambda x: x[1], reverse=True)
            
            # Отрисовка результатов на кадре
            frame_copy = frame.copy()  # Создаем копию кадра
            
            # Сначала рисуем менее уверенные объекты (те, что могут быть перекрыты)
            for bbox, confidence, _ in reversed(tracked_objects):
                x, y, w, h = bbox
                
                # Проверяем корректность координат
                if x >= 0 and y >= 0 and w > 0 and h > 0:
                    # Приводим к целым числам
                    x, y, w, h = int(x), int(y), int(w), int(h)
                    
                    # Ограничиваем координаты рамками кадра
                    x = max(0, min(x, frame.shape[1] - 1))
                    y = max(0, min(y, frame.shape[0] - 1))
                    w = min(w, frame.shape[1] - x)
                    h = min(h, frame.shape[0] - y)
                    
                    # Проверяем минимальные размеры
                    if w > 5 and h > 5:
                        # Для перекрытых объектов используем оранжевый цвет и более тонкую линию
                        if confidence < 0.4:
                            color = (0, 165, 255)  # Оранжевый для перекрытых объектов (BGR)
                            thickness = 1
                        else:
                            color = (0, 255, 0)  # Зеленый для основных объектов
                            thickness = 2
                        
                        # Рисуем бокс
                        cv2.rectangle(frame_copy, (x, y), (x + w, y + h), color, thickness)
                        
                        # Рисуем метку с уверенностью (без идентификатора)
                        label = f"{confidence:.2f}"
                        
                        # Рисуем фон для текста
                        label_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                        y_label = max(y - 5, label_size[1])
                        cv2.rectangle(frame_copy, 
                                     (x, y_label - label_size[1] - 5), 
                                     (x + label_size[0], y_label + baseline - 5), 
                                     color, -1)
                        
                        # Рисуем текст
                        cv2.putText(frame_copy, label, (x, y_label - 5), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            
            return frame_idx, frame_copy
        
        return frame_idx, frame
    except Exception as e:
        logging.error(f"Ошибка обработки кадра {frame_idx}: {e}")
        return frame_idx, frame

def detect_objects(model: YOLO, frame: np.ndarray) -> np.ndarray:
    """
    Детекция объектов на кадре с помощью модели YOLO.
    
    Args:
        model: Загруженная модель YOLO
        frame: Кадр для обработки
        
    Returns:
        Массив детекций в формате [x, y, w, h, conf, class]
    """
    with torch.no_grad():
        # Уменьшаем порог уверенности до 0.2 для обеспечения оптимального баланса между скоростью и качеством
        results = model(frame, conf=0.2, verbose=False, classes=[0], batch=1)[0].cpu()
    
    # Явно освобождаем память GPU после инференса
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Получаем данные боксов напрямую из результатов модели
    boxes = results.boxes.data.numpy()
    
    # Оптимизируем обработку боксов с использованием векторизации
    if len(boxes) > 0:
        # Извлекаем координаты из всех боксов сразу
        x1y1x2y2 = boxes[:, :4]
        conf_class = boxes[:, 4:]
        
        # Вычисляем ширину и высоту для всех боксов одновременно
        wh = x1y1x2y2[:, 2:4] - x1y1x2y2[:, :2]
        
        # Вычисляем центры боксов
        centers = x1y1x2y2[:, :2] + wh / 2
        
        # Применяем масштабирование (0.95) к ширине и высоте
        scale_factor = 0.95
        new_wh = wh * scale_factor
        
        # Вычисляем новые x1, y1 на основе центров и новых размеров
        new_x1y1 = centers - new_wh / 2
        
        # Собираем результаты в нужном формате [x, y, w, h, conf, class]
        processed_boxes = np.hstack((new_x1y1, new_wh, conf_class))
        
        return processed_boxes
    
    return np.array([])

def apply_tracking(detections: np.ndarray, kalman_filters: Dict, frame_idx: int, dt: float) -> List[Tuple]:
    """
    Применяет трекинг с использованием фильтра Калмана.
    
    Args:
        detections: Массив детекций
        kalman_filters: Словарь с фильтрами Калмана
        frame_idx: Индекс текущего кадра
        dt: Время между кадрами
        
    Returns:
        Список обработанных детекций с трекинг-информацией
    """
    tracked_objects = []
    current_objects = {}
    
    # Найти соответствия между текущими детекциями и существующими треками
    # на основе IoU (для сохранения перекрытых объектов)
    for i, det in enumerate(detections):
        bbox = det[:4]  # x, y, w, h
        confidence = det[4]
        x, y, w, h = bbox
        best_iou = 0
        best_track_id = None
        
        # Ищем наилучшее соответствие среди существующих треков
        for track_id, kf in kalman_filters.items():
            if frame_idx > 0 and int(track_id.split('_')[0]) < frame_idx - 5:
                # Получаем последнее состояние фильтра
                prev_bbox = kf.x[:4].flatten()
                prev_x, prev_y, prev_w, prev_h = prev_bbox
                
                # Преобразуем в формат x1, y1, x2, y2 для расчета IoU
                box1 = (x, y, x + w, y + h)
                box2 = (prev_x, prev_y, prev_x + prev_w, prev_y + prev_h)
                
                iou = calculate_iou(box1, box2)
                
                if iou > best_iou:
                    best_iou = iou
                    best_track_id = track_id
        
        # Создаем новый трек или используем существующий
        if best_iou > 0.2:  # Порог IoU для сопоставления треков
            track_id = best_track_id
        else:
            # Создаем новый идентификатор трека
            track_id = f"{frame_idx}_{i}"
            kalman_filters[track_id] = initialize_kalman_filter()
        
        # Применяем фильтр Калмана
        smoothed_bbox = apply_kalman_filter(bbox, kalman_filters[track_id], dt)
        
        # Сохраняем текущие объекты
        current_objects[track_id] = True
        
        # Убираем дополнительное масштабирование
        x, y, w, h = smoothed_bbox
        
        # Добавляем проверку на минимальный размер бокса
        if w < 5 or h < 5:
            w = max(w, 10)
            h = max(h, 20)
        
        # Добавляем в список обработанных объектов
        tracked_objects.append(([x, y, w, h], confidence, track_id))
    
    # Добавляем объекты, которые были перекрыты, но всё ещё в треке
    # (т.е. мы их не обнаружили на этом кадре)
    if frame_idx > 0:
        for track_id, kf in list(kalman_filters.items()):
            frame_of_origin = int(track_id.split('_')[0])
            
            # Если трек из недавнего прошлого (не старше 15 кадров) и не в текущих объектах
            if frame_of_origin > frame_idx - 15 and track_id not in current_objects:
                # Получаем предсказание от фильтра Калмана
                kf.predict()
                predicted_bbox = kf.x[:4].flatten()
                x, y, w, h = predicted_bbox
                
                # Проверяем, находится ли объект в пределах кадра
                if x >= 0 and y >= 0 and w > 5 and h > 5:
                    # Добавляем объект с пониженной уверенностью
                    confidence = 0.3  # Уменьшенная уверенность для перекрытых объектов
                    tracked_objects.append(([x, y, w, h], confidence, track_id))
    
    return tracked_objects

def group_detections(detections, iou_threshold=0.5):
    """
    Группирует близкие детекции с применением NMS.
    
    Args:
        detections: Массив детекций в формате [x, y, w, h, conf, class]
        iou_threshold: Порог IoU для считания боксов перекрывающимися
        
    Returns:
        Отфильтрованные детекции после NMS
    """
    if len(detections) == 0:
        return np.array([])
        
    # Сортируем по уверенности (confidence)
    detections = detections[detections[:, 4].argsort()[::-1]]
    
    # Применяем NMS
    keep = []
    
    for i in range(len(detections)):
        # Если текущая детекция уже была удалена, пропускаем
        if i in keep:
            continue
            
        keep.append(i)
        x1, y1, w1, h1 = detections[i][:4]
        
        for j in range(i + 1, len(detections)):
            if j in keep:
                continue
                
            x2, y2, w2, h2 = detections[j][:4]
            
            # Преобразуем [x, y, w, h] в [x1, y1, x2, y2] для расчета IoU
            box1 = (x1, y1, x1 + w1, y1 + h1)
            box2 = (x2, y2, x2 + w2, y2 + h2)
            
            if calculate_iou(box1, box2) > iou_threshold:
                keep.append(j)  # Удаляем перекрывающийся бокс
    
    return detections[keep]

def calculate_iou(box1: Tuple, box2: Tuple) -> float:
    """
    Вычисляет IoU (пересечение над объединением) двух боксов.
    
    Args:
        box1: Первый бокс в формате (x1, y1, x2, y2)
        box2: Второй бокс в формате (x1, y1, x2, y2)
        
    Returns:
        Значение IoU от 0 до 1
    """
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    
    # Координаты пересечения
    x_left = max(x1_1, x1_2)
    y_top = max(y1_1, y1_2)
    x_right = min(x2_1, x2_2)
    y_bottom = min(y2_1, y2_2)
    
    # Если боксы не пересекаются
    if x_right < x_left or y_bottom < y_top:
        return 0.0
        
    # Площадь пересечения
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    
    # Площади боксов
    box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
    box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
    
    # IoU = пересечение / объединение
    iou = intersection_area / float(box1_area + box2_area - intersection_area)
    
    return iou

def soft_nms(dets: np.ndarray, sigma: float = 0.5, Nt: float = 0.3, method: int = 2) -> np.ndarray:
    """
    Оптимизированная реализация Soft-NMS для уменьшения перекрывающихся детекций.
    
    Args:
        dets: Массив детекций в формате [x, y, w, h, conf, class]
        sigma: Параметр для гауссова метода (метод 2)
        Nt: Порог IoU
        method: Метод (1: линейный, 2: гауссов, 3: оригинальный NMS)
        
    Returns:
        Отфильтрованные детекции
    """
    if len(dets) == 0:
        return np.array([])
    
    # Конвертируем [x, y, w, h] в [x1, y1, x2, y2] для NMS
    x1y1 = dets[:, :2]
    wh = dets[:, 2:4]
    x2y2 = x1y1 + wh
    boxes = np.hstack((x1y1, x2y2, dets[:, 4:]))
    
    N = boxes.shape[0]
    
    # Предварительно сортируем по уверенности для ускорения
    score_indices = boxes[:, 4].argsort()[::-1]
    boxes = boxes[score_indices]
    
    # Предварительно вычисляем площади боксов для ускорения IoU
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    
    for i in range(N):
        # Текущий бокс с высшей уверенностью
        tx1, ty1, tx2, ty2, tscore = boxes[i, :5]
        
        # Обрабатываем оставшиеся боксы
        pos = i + 1
        
        # Если мы уже обработали большинство боксов, можно досрочно завершить
        if pos >= N or boxes[pos, 4] < 0.1:
            break
            
        while pos < N:
            x1, y1, x2, y2 = boxes[pos, :4]
            
            # Быстрое вычисление IoU с оптимизацией
            xx1 = max(tx1, x1)
            yy1 = max(ty1, y1)
            xx2 = min(tx2, x2)
            yy2 = min(ty2, y2)
            
            # Проверка на перекрытие
            if xx2 > xx1 and yy2 > yy1:
                # Есть перекрытие - вычисляем IoU
                intersection = (xx2 - xx1) * (yy2 - yy1)
                area1 = (tx2 - tx1) * (ty2 - ty1)
                area2 = areas[pos]
                iou = intersection / (area1 + area2 - intersection)
                
                # Применяем соответствующий метод
                if method == 1:  # линейный
                    weight = 1 - iou if iou > Nt else 1
                elif method == 2:  # гауссов
                    weight = np.exp(-(iou * iou) / sigma)
                else:  # оригинальный NMS
                    weight = 0 if iou > Nt else 1
                
                # Обновляем уверенность
                boxes[pos, 4] *= weight
                
                # Если уверенность упала ниже порога, удаляем бокс
                if boxes[pos, 4] < 0.01:
                    boxes[pos] = boxes[-1].copy()
                    areas[pos] = areas[-1]
                    N -= 1
                    pos -= 1
            
            pos += 1
    
    # Конвертируем обратно в [x, y, w, h]
    result = []
    for i in range(N):
        if boxes[i, 4] >= 0.1:  # Возвращаем только боксы с уверенностью выше порога
            x1, y1, x2, y2 = boxes[i, :4]
            w, h = x2 - x1, y2 - y1
            result.append([x1, y1, w, h, boxes[i, 4], boxes[i, 5]])
    
    return np.array(result)


def detect_and_track_parallel(input_video_path: str, output_video_path: str, model_paths: List[str] = ["yolo11m.pt"],
                          num_processes: int = None, iou_threshold: float = 0.3,
                          use_soft_nms: bool = True, output_format: str = "mp4v", scale_factor: float = 1.0, 
                          max_buffer_size: int = 100, batch_size: int = 16):
    """
    Выполняет параллельную детекцию и трекинг людей на видео.
    
    Args:
        input_video_path: Путь к входному видео
        output_video_path: Путь к выходному видео
        model_paths: Список путей к моделям YOLO для ансамблирования
        num_processes: Количество процессов для параллельной обработки
        iou_threshold: Порог IoU для NMS/фильтра Калмана
        use_soft_nms: Использовать ли Soft-NMS
        output_format: Формат выходного видео (кодек)
        scale_factor: Коэффициент масштабирования кадра (1.0 = оригинальный размер)
        max_buffer_size: Максимальный размер буфера кадров
        batch_size: Фиксированный размер батча для обработки
    """
    # Проверка существования файла
    if not os.path.exists(input_video_path):
        logging.error(f"Входной файл {input_video_path} не найден")
        return

    # Открываем видео
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        logging.error(f"Не удалось открыть видео {input_video_path}")
        return

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    logging.info(f"Разрешение видео: {frame_width}x{frame_height}, FPS: {fps}, Кадров: {total_frames}")

    # Используем только mp4v как самый совместимый кодек
    logging.info("Используется кодек mp4v для совместимости")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    # Если указан масштаб отличный от 1.0, изменяем размер выходного видео
    output_width = int(frame_width * scale_factor)
    output_height = int(frame_height * scale_factor)
    
    logging.info(f"Создание выходного файла: {output_video_path} с размером {output_width}x{output_height}")
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (output_width, output_height))
    if not out.isOpened():
        logging.error(f"Не удалось создать выходной файл: {output_video_path}")
        cap.release()
        return

    # Оптимизируем количество процессов
    if num_processes is None:
        # Используем оптимальное количество процессов: CPU - 1, но не менее 2
        num_processes = max(2, multiprocessing.cpu_count() - 1)
    logging.info(f"Используется {num_processes} процессов.")

    # --- ЗАГРУЗКА МОДЕЛЕЙ (в главном процессе) ---
    logging.info("Загрузка моделей YOLO...")
    models = [YOLO(mp) for mp in model_paths]
    
    # Устанавливаем оптимальные параметры модели
    for model in models:
        # Оптимизируем модель для лучшей производительности
        if torch.cuda.is_available():
            model.to('cuda')  # Перемещаем на GPU, если доступен
        else:
            # Для CPU оптимизации
            model.to('cpu')
            if hasattr(torch, 'set_num_threads'):
                torch.set_num_threads(num_processes)  # Оптимизация для CPU
    
    logging.info(f"Загружено {len(models)} моделей YOLO: {model_paths}")

    manager = Manager()
    kalman_filters = manager.dict()  # Инициализируем словарь фильтров Калмана
    start_time = time.time()
    last_log_time = start_time
    prev_time = start_time
    
    # --- ОБРАБОТКА ВИДЕО БАТЧАМИ БЕЗ ЗАГРУЗКИ ВСЕГО ВИДЕО В ПАМЯТИ ---
    with Pool(processes=num_processes) as pool:
        # Создаем partial функцию, передавая список моделей для ансамблирования
        process_func = partial(process_frame, models=models, kalman_filters=kalman_filters,
                              iou_threshold=iou_threshold, use_soft_nms=use_soft_nms)
        
        # Обработка видео порциями для экономии памяти
        logging.info("Начало обработки видео...")
        
        # Буфер для сохранения порядка кадров (используем OrderedDict для лучшей производительности)
        from collections import OrderedDict
        frame_buffer = OrderedDict()
        next_frame_idx = 0
        
        logging.info(f"Используется фиксированный размер батча: {batch_size}")
        processed_count = 0
        
        # Предварительно выделяем массивы для кадров чтобы избежать повторной аллокации
        frames_batch = []
        
        while processed_count < total_frames:
            # Эффективный контроль буфера кадров
            while len(frame_buffer) >= max_buffer_size // 2:  # Ускоряем освобождение буфера
                while next_frame_idx in frame_buffer:
                    out.write(frame_buffer.pop(next_frame_idx))
                    next_frame_idx += 1
                
                if len(frame_buffer) >= max_buffer_size // 2:  # Если буфер всё ещё большой
                    time.sleep(0.005)  # Короткая пауза, чтобы не загружать процессор
                    break
            
            # Загружаем батч кадров
            frames_batch.clear()  # Очищаем список без пересоздания
            for _ in range(batch_size):
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Изменяем размер кадра, если нужно
                if scale_factor != 1.0:
                    frame = cv2.resize(frame, (output_width, output_height), interpolation=cv2.INTER_AREA)
                
                current_time = time.time()
                dt = current_time - prev_time
                prev_time = current_time
                
                frames_batch.append((processed_count, frame, dt))
                processed_count += 1
            
            if not frames_batch:
                break  # Конец видео
                
            # Параллельная обработка батча с использованием imap_unordered для максимальной скорости
            # Мы не беспокоимся о порядке здесь, так как используем индексы кадров
            for result in pool.imap_unordered(process_func, frames_batch):
                frame_idx, processed_frame = result
                frame_buffer[frame_idx] = processed_frame
                
                # Иногда проверяем и записываем готовые кадры
                if frame_idx % 4 == 0:  # Оптимизировано для уменьшения частых проверок
                    while next_frame_idx in frame_buffer:
                        out.write(frame_buffer.pop(next_frame_idx))
                        next_frame_idx += 1
            
            # Логирование прогресса
            current_time = time.time()
            if current_time - last_log_time >= 5:  # Реже логируем для ускорения
                elapsed_time = current_time - start_time
                if processed_count > 0 and fps > 0:
                    remaining_frames = total_frames - processed_count
                    time_per_frame = elapsed_time / processed_count
                    estimated_remaining_time = remaining_frames * time_per_frame
                    memory_usage = f"{len(frame_buffer)} кадров в буфере"
                    logging.info(f"Обработано: {processed_count}/{total_frames}. Осталось: {estimated_remaining_time:.2f} сек. {memory_usage}")
                else:
                    logging.info(f"Обработано: {processed_count}/{total_frames}. Оценка времени пока недоступна.")
                last_log_time = current_time
    
    # Записываем оставшиеся кадры в буфере (если есть)
    if frame_buffer:
        remaining_frames = sorted(frame_buffer.items())
        for _, frame in remaining_frames:
            out.write(frame)
    
    logging.info(f"Общее время: {time.time() - start_time:.2f} сек.")
    cap.release()
    out.release()
    logging.info(f"Готово. Результат в {output_video_path}")

def ensemble_predictions(models: List[YOLO], frame: np.ndarray) -> np.ndarray:
    """
    Выполняет оптимизированное ансамблирование предсказаний от нескольких моделей YOLO.
    
    Args:
        models: Список моделей YOLO
        frame: Кадр для обработки
        
    Returns:
        Объединенные результаты детекции
    """
    # Если одна модель, используем прямую детекцию
    if len(models) == 1:
        return detect_objects(models[0], frame)
    
    all_predictions = []
    
    # Получаем предсказания от каждой модели
    for model in models:
        with torch.no_grad():
            # Оптимизация инференса для повышения скорости
            results = model(frame, conf=0.2, verbose=False, classes=[0], batch=1)[0].cpu()
        
        # Явно освобождаем память после инференса
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Быстрое преобразование формата боксов с использованием векторизации
        boxes = results.boxes.data.numpy()
        
        if len(boxes) > 0:
            # Извлекаем координаты из всех боксов сразу
            x1y1x2y2 = boxes[:, :4]
            conf_class = boxes[:, 4:]
            
            # Вычисляем ширину и высоту, центры и новые координаты для всех боксов одновременно
            wh = x1y1x2y2[:, 2:4] - x1y1x2y2[:, :2]
            centers = x1y1x2y2[:, :2] + wh / 2
            new_wh = wh * 0.95  # 5% уменьшение размера бокса
            new_x1y1 = centers - new_wh / 2
            
            # Собираем результаты в формате [x, y, w, h, conf, class]
            processed = np.hstack((new_x1y1, new_wh, conf_class))
            all_predictions.append(processed)
    
    # Проверка наличия предсказаний
    if not all_predictions:
        return np.array([])
    
    # Эффективное объединение и применение NMS
    combined = np.vstack(all_predictions) if len(all_predictions) > 0 else np.array([])
    
    # Применяем оптимизированный NMS к объединенным результатам
    if len(combined) > 0:
        # Используем более высокий порог для более быстрой фильтрации
        return group_detections(combined, iou_threshold=0.5)
    
    return np.array([])

if __name__ == "__main__":
    # Парсинг аргументов командной строки
    parser = argparse.ArgumentParser(description='Детекция и трекинг людей на видео с использованием YOLO и фильтра Калмана')
    parser.add_argument('--input', type=str, default="crowd.mp4", help='Путь к входному видеофайлу')
    parser.add_argument('--output', type=str, default="output.mp4", help='Путь к выходному видеофайлу')
    parser.add_argument('--models', type=str, nargs='+', default=["yolo11m.pt"], 
                        help='Пути к моделям YOLO (для ансамблирования укажите несколько через пробел)')
    parser.add_argument('--processes', type=int, default=None, 
                        help='Количество процессов (None = автоопределение)')
    parser.add_argument('--iou', type=float, default=0.3, 
                        help='Порог IoU для NMS/Soft-NMS и фильтра Калмана')
    parser.add_argument('--soft-nms', action='store_true', default=True,
                        help='Использовать Soft-NMS (по умолчанию: True)')
    parser.add_argument('--scale', type=float, default=1.0,
                        help='Масштабный коэффициент для изменения размера кадра (1.0 = оригинальный размер)')
    parser.add_argument('--buffer-size', type=int, default=100,
                        help='Максимальный размер буфера кадров (влияет на потребление памяти)')
    parser.add_argument('--batch-size', type=int, default=16,
                        help='Фиксированный размер батча для обработки видео')
    
    args = parser.parse_args()
    
    # Запуск детекции и трекинга с параметрами из командной строки или значениями по умолчанию
    detect_and_track_parallel(
        input_video_path=args.input,
        output_video_path=args.output,
        model_paths=args.models,
        num_processes=args.processes,
        iou_threshold=args.iou,
        use_soft_nms=args.soft_nms,
        output_format="mp4v",  # Всегда используем mp4v как самый совместимый кодек
        scale_factor=args.scale,  # Используем масштаб из аргументов
        max_buffer_size=args.buffer_size,  # Используем размер буфера из аргументов
        batch_size=args.batch_size  # Используем фиксированный размер батча
    )