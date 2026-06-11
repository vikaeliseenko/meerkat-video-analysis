from pathlib import Path
import runpy


ROOT = Path(__file__).resolve().parent
runpy.run_path(str(ROOT / "src" / "03_analyze_video.py"), run_name="__main__")
