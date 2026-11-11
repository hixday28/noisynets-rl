"""Выполнение ноутбука в текущем окружении Python."""

import json
import tempfile
from pathlib import Path
import sys

import nbformat
from jupyter_client.kernelspec import KernelSpecManager
from nbclient import NotebookClient

root = Path(__file__).resolve().parents[1]
path = root / "notebooks/01_pilot_study.ipynb"
with tempfile.TemporaryDirectory(prefix="noisynets-kernel-") as temporary:
    spec = Path(temporary) / "noisynets"
    spec.mkdir()
    (spec / "kernel.json").write_text(json.dumps({
        "argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
        "display_name": "NoisyNets research", "language": "python"}))
    nb = nbformat.read(path, as_version=4)
    client = NotebookClient(nb, timeout=180, kernel_name="noisynets",
                            resources={"metadata": {"path": str(root)}})
    client.km = client.create_kernel_manager()
    client.km.kernel_spec_manager = KernelSpecManager(kernel_dirs=[temporary])
    client.execute()
    nbformat.write(nb, path)
print(path)
