"""One uv venv per model at ~/cmp/v100/<name>: fmugen (editable, this repository), FMPy 0.3.32,
psutil, PyInstaller 6.22.3 and the model's packages (torch from its CPU index).

    python venvs.py [name ...]
"""
import concurrent.futures as cf
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from models100 import MODELS  # noqa: E402

REPO = HERE.parents[1]
UV = str(Path.home() / ".local/bin/uv")
V = Path.home() / "cmp/v100"
LOGS = Path.home() / "cmp/w100/_venv"
BASE = ["fmpy==0.3.32", "psutil", "pyinstaller==6.22.3"]


def setup(m):
    name, pkgs, _, extra = m
    py = V / name / "bin/python"
    LOGS.mkdir(parents=True, exist_ok=True)
    with open(LOGS / f"{name}.log", "w") as f:
        def run(cmd):
            f.write("$ " + " ".join(cmd) + "\n"); f.flush()
            return subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, timeout=3600).returncode
        if py.exists() and subprocess.run([str(py), "-c", "import fmugen, fmpy"], capture_output=True).returncode == 0:
            return name, "exists"
        if not py.exists() and run([UV, "venv", "-q", str(V / name), "--python", "3.13", "--seed"]):
            return name, "venv failed"
        torch = ["--torch-backend", "cpu"] if extra.get("torch") else []
        if run([UV, "pip", "install", "--python", str(py), *torch, "-e", str(REPO), *BASE, *pkgs]):
            # a package without Python 3.13 wheels: try 3.12
            return name, "install failed"
        return name, "ok"


if __name__ == "__main__":
    names = sys.argv[1:]
    ms = [m for m in MODELS if not names or m[0] in names]
    status = {}
    with cf.ThreadPoolExecutor(8) as ex:
        for fut in cf.as_completed([ex.submit(setup, m) for m in ms]):
            try:
                n, s = fut.result()
            except Exception as e:
                n, s = "?", repr(e)
            status[n] = s
            print(n, s, flush=True)
    (LOGS / "status.json").write_text(json.dumps(status, indent=1))
