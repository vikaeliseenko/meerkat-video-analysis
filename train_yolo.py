from pathlib import Path
import runpy


ROOT = Path(__file__).resolve().parent
runpy.run_path(str(ROOT / "src" / "02_train_yolo.py"), run_name="__main__")
