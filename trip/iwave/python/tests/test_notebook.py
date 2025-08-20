import sys
from pathlib import Path

import nbformat
from nbclient import NotebookClient
import pytest
import os

# Ensure repository root in path
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))


@pytest.mark.skip(reason="Notebook execution requires built IWAVE binaries and may be slow")
def test_two_layer_notebook_executes():
    trip_root = Path(__file__).resolve().parents[3]
    acd = trip_root / "iwave" / "acd" / "main" / "acd.x"
    asg = trip_root / "iwave" / "asg" / "main" / "asg.x"
    if not (acd.exists() and asg.exists()):
        pytest.skip("IWAVE binaries not built")
    nb_path = trip_root / "iwave" / "python" / "examples" / "two_layer_model.ipynb"
    nb = nbformat.read(nb_path, as_version=4)
    os.environ["REPO_ROOT"] = str(trip_root.parent)
    client = NotebookClient(
        nb, timeout=600, kernel_name="python3", resources={"metadata": {"path": nb_path.parent}}
    )
    client.execute()
