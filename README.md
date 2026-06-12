# IMG_1561 meerkat behavior analysis

Проект содержит полный воспроизводимый пакет для автоматизированного анализа видео `IMG_1561.MP4`: подготовку кадров, pseudo-разметку, дообучение детектора сурикатов, трекинг, классификацию поведенческих событий, графики, таблицы и текстовые фрагменты для работы.

Важно: модель обучена на автоматически полученной pseudo-разметке из одного видеоматериала. Результаты подходят как рабочий прототип и предварительный анализ, но для строгих научных выводов нужна ручная экспертная валидация разметки, зоны обогащения и поведенческих категорий.

## Структура

- `videos/IMG_1561.MP4` - исходное видео.
- `src/00_extract_frames.py` - извлечение кадров и обзорных листов.
- `src/01_generate_pseudo_labels.py` - автоматическая pseudo-разметка в формате YOLO.
- `src/02_train_yolo.py` и `train_yolo.py` - дообучение YOLOv8n.
- `src/03_analyze_video.py` и `analyze_video.py` - трекинг, классификация поведения и аннотированное видео.
- `src/04_make_visuals.py` - итоговые таблицы и графики.
- `dataset/` - изображения и YOLO-разметка для обучения/валидации.
- `runs/detect/IMG1561_meerkat_detector/weights/best.pt` - дообученная модель.
- `results/` - CSV-таблицы, графики, контактные листы и аннотированное видео.
- `docs/` - алгоритм, материалы и методы, результаты и список кода.

## Быстрый запуск

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe src\00_extract_frames.py
.\.venv\Scripts\python.exe src\01_generate_pseudo_labels.py
.\.venv\Scripts\python.exe train_yolo.py
.\.venv\Scripts\python.exe analyze_video.py
.\.venv\Scripts\python.exe src\04_make_visuals.py
```

## Ключевые результаты

- Видео: 1280x720 px, 24.985 fps, 13 263 кадра, 530.845 с.
- Извлечено: 531 кадр с частотой 1 кадр/с.
- Pseudo-разметка: 603 bounding boxes, 424 train-кадра, 107 val-кадров, 186 кадров без объектов.
- Модель: YOLOv8n, 35 эпох, `imgsz=512`, `batch=16`, CPU.
- Валидация на pseudo-разметке: Precision 0.747, Recall 0.589, mAP50 0.674, mAP50-95 0.509.
- Поведенческий анализ: 2745 записей-событий, 57 временных track ID, анализ с частотой 3 кадра/с.

Основные выходные файлы:

- `results/IMG1561_behavior_event_summary.csv`
- `results/IMG1561_interval_behavior_summary.csv`
- `results/IMG1561_behavior_counts.png`
- `results/IMG1561_interval_behavior_stacked.png`
- `results/IMG1561_detection_centers_trajectory.png`
- `results/IMG1561_annotated_video.mp4`

