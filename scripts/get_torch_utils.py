from pathlib import Path
from urllib import request

VERSION = "v0.29.2"
BASE = f"https://raw.githubusercontent.com/pytorch/vision/{VERSION}/references/detection"
FILES = ["engine.py", "utils.py", "coco_utils.py", "coco_eval.py", "transforms.py"]
DEST = Path(__file__).resolve().parent.parent / "src" / "methods"

DEST.mkdir(parents=True, exist_ok=True)
for name in FILES:
    request.urlretrieve(f"{BASE}/{name}", DEST / name)
