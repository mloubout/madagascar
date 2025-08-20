import subprocess
import tempfile
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np

__all__ = ["run_iwave", "write_rsf", "read_rsf"]


def write_rsf(path: Path, array: np.ndarray) -> Path:
    """Write ``array`` to an RSF file.

    Parameters
    ----------
    path:
        Path to the header file to be written. The associated binary file will
        have the same name with an additional ``.bin`` suffix.
    array:
        ``numpy`` array containing the data to be written. The array is written
        in native 32-bit floating point format.

    Returns
    -------
    Path
        Path to the created RSF header file.
    """
    path = Path(path)
    data_path = path.with_suffix(path.suffix + ".bin")
    array = np.asarray(array, dtype=np.float32)
    array.tofile(data_path)

    header_lines: List[str] = [f"in={data_path}", "data_format=native_float"]
    for i, n in enumerate(array.shape, start=1):
        header_lines.append(f"n{i}={int(n)}")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(header_lines) + "\n")
    return path


def read_rsf(path: Path) -> np.ndarray:
    """Read an RSF file created by :func:`write_rsf` into a ``numpy`` array."""
    path = Path(path)
    header: Dict[str, str] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if "=" in line:
                key, val = line.strip().split("=", 1)
                header[key.strip()] = val.strip()
    data_path = Path(header["in"])  # type: ignore[index]
    shape: List[int] = []
    for i in range(1, len(header) + 1):
        key = f"n{i}"
        if key in header:
            shape.append(int(header[key]))
        else:
            break
    data = np.fromfile(data_path, dtype=np.float32)
    if shape:
        data = data.reshape(shape)
    return data


def run_iwave(
    binary: str,
    inputs: Dict[str, np.ndarray],
    output_keys: Iterable[str],
    extra_args: Optional[Iterable[str]] = None,
) -> Dict[str, np.ndarray]:
    """Run an IWAVE program using ``numpy`` arrays.

    Parameters
    ----------
    binary:
        Path to the IWAVE executable (for example ``"acd"``).
    inputs:
        Mapping of IWAVE ``iokey`` names to ``numpy`` arrays. Each array will be
        written to a temporary RSF file and passed to the executable as
        ``key=filename``.
    output_keys:
        Iterable of ``iokey`` names that the executable will produce. These are
        also passed as ``key=filename`` arguments and read back after the
        executable finishes.
    extra_args:
        Additional command line arguments to append when invoking the binary.

    Returns
    -------
    Dict[str, np.ndarray]
        Mapping of ``output_keys`` to arrays read from the generated RSF files.
    """
    extra_args = list(extra_args) if extra_args is not None else []

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        cmd: List[str] = [binary]

        # Handle inputs
        for key, arr in inputs.items():
            hdr = tmp_path / f"{key}.rsf"
            write_rsf(hdr, arr)
            cmd.append(f"{key}={hdr}")

        # Reserve file names for outputs
        out_files: Dict[str, Path] = {}
        for key in output_keys:
            hdr = tmp_path / f"{key}.rsf"
            out_files[key] = hdr
            cmd.append(f"{key}={hdr}")

        cmd.extend(extra_args)

        subprocess.run(cmd, check=True)

        outputs: Dict[str, np.ndarray] = {}
        for key, hdr in out_files.items():
            outputs[key] = read_rsf(hdr)

    return outputs
