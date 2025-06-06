import re
from pathlib import Path

ROOT_FOLDER = Path(re.sub(r"(.*?/YOUR_PROJECT_ROOT_FOLDER/).*", r"\1", __file__))
