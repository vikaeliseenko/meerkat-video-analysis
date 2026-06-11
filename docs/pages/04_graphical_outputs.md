# Графическое отображение результатов

В проекте сформированы следующие графические материалы.

## Контроль исходных кадров и разметки

- `results/IMG1561_contact_sheet.jpg` - контактный лист всех извлечённых кадров с частотой 1 кадр/с.
- `results/IMG1561_overview_20s.jpg` - обзорный контактный лист с шагом 20 с.
- `results/IMG1561_pseudo_bbox_contact_sheet_10s.jpg` - контроль pseudo-разметки с рамками через каждые 10 с.

## Графики поведения

- `results/IMG1561_behavior_counts.png` - распределение числа записей по категориям поведения.
- `results/IMG1561_interval_behavior_stacked.png` - stacked bar plot динамики поведенческих категорий по 30-секундным интервалам.
- `results/IMG1561_detection_centers_trajectory.png` - траектории центров обнаруженных рамок в координатах кадра.

## Видео и контрольные изображения анализа

- `results/IMG1561_annotated_video.mp4` - исходное видео с наложенными рамками, временными ID, категориями поведения и зоной обогащения.
- `results/IMG1561_annotated_contact_sheet.jpg` - контактный лист аннотированного видео с шагом 60 с.

## Графики обучения модели

- `runs/detect/IMG1561_meerkat_detector/results.png` - динамика loss-функций и метрик обучения.
- `runs/detect/IMG1561_meerkat_detector/BoxPR_curve.png` - precision-recall кривая.
- `runs/detect/IMG1561_meerkat_detector/BoxF1_curve.png` - F1-кривая.
- `runs/detect/IMG1561_meerkat_detector/confusion_matrix.png` - матрица ошибок на pseudo-val.
- `runs/detect/IMG1561_meerkat_detector/val_batch*_pred.jpg` - примеры предсказаний модели на валидационных изображениях.

Все графики построены автоматически скриптом `src/04_make_visuals.py` или сгенерированы Ultralytics YOLO во время обучения.

