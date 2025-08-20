import sys
from pathlib import Path

# Ensure repository root is on sys.path so ``trip`` package is importable
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

import numpy as np
from pathlib import Path
from unittest.mock import patch

from trip.iwave.python.iwave import run_iwave, write_rsf


def test_run_iwave_roundtrip(tmp_path):
    # Prepare input array
    csq = np.ones((2, 3), dtype=np.float32)

    def fake_run(cmd, check):
        # cmd[-1] expected to be output argument e.g., data=path
        for arg in cmd[1:]:
            if arg.startswith("data="):
                out_hdr = Path(arg.split("=", 1)[1])
                write_rsf(out_hdr, np.zeros((2, 3), dtype=np.float32))
        return 0

    with patch("subprocess.run", fake_run):
        outputs = run_iwave(
            "dummy_binary",
            inputs={"csq": csq},
            output_specs={"data": csq.shape},
        )
    assert "data" in outputs
    assert outputs["data"].shape == (2, 3)
    assert np.allclose(outputs["data"], 0.0)
