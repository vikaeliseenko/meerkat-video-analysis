# Использованный код и воспроизводимость

## Окружение

Зависимости перечислены в `requirements.txt`. Основные библиотеки: `ultralytics`, `opencv-python`, `pandas`, `numpy`, `matplotlib`, `scikit-learn`, `lap`.

Рекомендуемый запуск в Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Скрипты

- `src/00_extract_frames.py` - открывает `videos/IMG_1561.MP4`, извлекает кадры 1 кадр/с, сохраняет `frames_for_markup/*.jpg`, `results/IMG1561_contact_sheet.jpg` и `results/IMG1561_overview_20s.jpg`.
- `src/01_generate_pseudo_labels.py` - создаёт pseudo-разметку через MOG2 + якорные зоны, формирует `dataset/images`, `dataset/labels`, `results/IMG1561_pseudo_bbox_annotations.csv` и `results/IMG1561_pseudo_label_summary.txt`.
- `src/02_train_yolo.py` - дообучает YOLOv8n на `dataset/data.yaml`, сохраняет результаты в `runs/detect/IMG1561_meerkat_detector/`.
- `train_yolo.py` - короткий wrapper для запуска `src/02_train_yolo.py` из корня проекта.
- `src/03_analyze_video.py` - применяет `best.pt` к видео, выполняет ByteTrack-трекинг, классифицирует поведение, создаёт `results/IMG1561_raw_frame_results.csv`, `results/IMG1561_ethogram_table.csv` и `results/IMG1561_annotated_video.mp4`.
- `analyze_video.py` - короткий wrapper для запуска `src/03_analyze_video.py` из корня проекта.
- `src/04_make_visuals.py` - строит итоговые CSV-сводки и графики поведения.

## Команды полного воспроизведения

```powershell
.\.venv\Scripts\python.exe src\00_extract_frames.py
.\.venv\Scripts\python.exe src\01_generate_pseudo_labels.py
.\.venv\Scripts\python.exe train_yolo.py
.\.venv\Scripts\python.exe analyze_video.py
.\.venv\Scripts\python.exe src\04_make_visuals.py
```

## Основные выходные таблицы

- `results/IMG1561_pseudo_bbox_annotations.csv` - таблица pseudo-разметки.
- `results/IMG1561_raw_frame_results.csv` - покадровые события анализа.
- `results/IMG1561_ethogram_table.csv` - агрегация по 30-секундным интервалам, ID и категориям.
- `results/IMG1561_behavior_event_summary.csv` - общая сводка по категориям поведения.
- `results/IMG1561_interval_behavior_summary.csv` - сводка по категориям поведения в 30-секундных интервалах.

## Модель

Финальные веса модели:

```text
runs/detect/IMG1561_meerkat_detector/weights/best.pt
```

Модель предназначена для предварительного выделения сурикатов в видео, визуально близких к `IMG_1561.MP4`. Для переноса на другие камеры, освещение и условия содержания рекомендуется добавить ручную разметку и выполнить повторное дообучение.

