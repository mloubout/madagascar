import subprocess
import sys
from pathlib import Path

import pytest


def test_two_layer_notebook_runs_with_nbval():
    trip_root = Path(__file__).resolve().parents[3]
    acd = trip_root / "iwave" / "acd" / "main" / "acd.x"
    asg = trip_root / "iwave" / "asg" / "main" / "asg.x"
    if not (acd.exists() and asg.exists()):
        pytest.skip("IWAVE binaries not built")
    nb_path = trip_root / "iwave" / "python" / "examples" / "two_layer_model.ipynb"
    subprocess.run(
        [sys.executable, "-m", "pytest", "--nbval-lax", str(nb_path)], check=True, timeout=300
    )
