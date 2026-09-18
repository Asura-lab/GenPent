"""pytest-ийн import тохиргоо."""
import sys
from pathlib import Path

# Төслийн үндсэн хавтасыг sys.path-д нэмэх (agents, envs, evaluation import-д)
sys.path.insert(0, str(Path(__file__).parent.parent))
