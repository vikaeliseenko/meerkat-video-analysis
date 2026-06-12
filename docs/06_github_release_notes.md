# GitHub release notes

## Release contents

This repository stores the reproducible code, documentation, trained detector checkpoint, CSV outputs and lightweight visual summaries for the `IMG_1561.MP4` meerkat behavior analysis.

Large generated/input assets are intentionally not stored in the git history:

- `videos/IMG_1561.MP4`
- `frames_for_markup/`
- `dataset/images/`
- `results/IMG1561_annotated_video.mp4`
- `results/IMG1561_contact_sheet.jpg`

The full project package is attached to the GitHub release as:

```text
meerkat_IMG1561_development_project_archive_20260610.zip
```

The archive contains the complete local project, including source video, extracted frames, dataset images, annotated video, trained weights, result tables, figures, scripts and documents. It excludes the local virtual environment and cache folders.

## Scientific caveat

The detector was fine-tuned from pseudo-labels generated from a single video. The results are suitable as a preliminary automated analysis and reproducible prototype. Manual expert validation is required before using the metrics as strict scientific evidence.

