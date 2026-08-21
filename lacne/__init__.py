"""Land-Air Collaboration Network design (LAC-CTNDP) solver.

Upper level: mixed-integer Bayesian optimization (MI-BO).
Lower level: LAC-NE user equilibrium solved with the greedy path-based
algorithm (IGP) provided by the ``IGP`` package.
"""
import sys
from pathlib import Path

# Make the sibling IGP package importable regardless of cwd.
_IGP_DIR = str(Path(__file__).resolve().parents[1] / "IGP")
if _IGP_DIR not in sys.path:
    sys.path.insert(0, _IGP_DIR)

__version__ = "0.1.0"
