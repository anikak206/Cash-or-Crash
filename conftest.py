# Makes sure `import src` and `import app` work when pytest runs from anywhere.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
