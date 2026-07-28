"""Ensures the repo root is importable when running pytest without installing."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
