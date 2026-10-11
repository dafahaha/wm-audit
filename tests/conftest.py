import os
import sys

# Make the project root (parent of tests/) importable so `import wm_audit`
# works whether tests are launched via `pytest` or `python -m pytest`.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
