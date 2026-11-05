"""Test the 100 pending models of 2026-10-29 (docs/testing/run-pending-models.md) with pending-100's scripts.

    python runr4.py venvs   [name ...]   one uv venv per model (pending-100/venvs.py)
    python runr4.py init    [name ...]   fmugen init --probe and plain init
    python runr4.py run     [name ...]   build + simulate + direct run (twice) + compare, each config
    python runr4.py compile [name ...]   the best matching config: build --compile pyinstaller, simulate, compare
    python runr4.py all     [name ...]   the three in turn, per model
    python runr4.py report               results.json and results.csv next to this script

The models are in models_r4.py (models100.py's format), hand configs in hand/<name>/fmugen.toml.
Venvs: ~/cmp/vr4/<name>. Work: ~/cmp/wr4/m/<name>, data: ~/cmp/wr4/data. UniFMU: the repository's
unifmu/unifmu.exe (the Windows build, run through WSL interop) unless CMP_UNIFMU is set.
"""
import concurrent.futures as cf
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
P100 = HERE.parent / "pending-100"
HOME = Path.home()
os.environ.setdefault("CMP_VENVS", str(HOME / "cmp/vr4"))
os.environ.setdefault("CMP_WORK", str(HOME / "cmp/wr4"))
os.environ.setdefault("CMP_UNIFMU", str(HERE.parents[2] / "unifmu/unifmu.exe"))
sys.path.insert(0, str(HERE))
import models_r4  # noqa: E402
sys.modules["models100"] = models_r4
sys.path.insert(1, str(P100))
import run100  # noqa: E402
import venvs  # noqa: E402

run100.HERE = HERE                       # report() writes results.json/.csv here
run100.DATA = Path(os.environ["CMP_WORK"]) / "data"
run100.R.HAND = HERE / "hand"
run100.R.MODELS = [run100.as_run_all(m) for m in models_r4.MODELS]   # again: run100 expanded <data> with its own DATA
venvs.REPO = HERE.parents[2]             # venvs.py's own REPO is docs/ since it moved into docs/testing
venvs.V = Path(os.environ["CMP_VENVS"])
venvs.LOGS = Path(os.environ["CMP_WORK"]) / "_venv"

if __name__ == "__main__":
    cmd, names = sys.argv[1], sys.argv[2:]
    if cmd == "report":
        run100.report()
        # keep 20 of each config's output names and last values: neural networks output up to
        # 200,000 (SAM's masks)
        import json
        rows = json.loads((HERE / "results.json").read_text(encoding="utf-8"))
        for r in rows:
            for c in r["configs"]:
                last = c.get("last") or {}
                if len(last) > 20:
                    c["last"] = dict(list(last.items())[:20]) | {"...": f"{len(last) - 20} more"}
                outs = c.get("fmu_outputs") or []
                if len(outs) > 20:
                    c["fmu_outputs"] = outs[:20] + [f"... {len(outs) - 20} more"]
        (HERE / "results.json").write_text(json.dumps(rows, indent=1, ensure_ascii=False), encoding="utf-8")
        sys.exit()
    names = names or [m[0] for m in models_r4.MODELS]
    if cmd == "venvs":
        ms = [m for m in models_r4.MODELS if m[0] in names]
        with cf.ThreadPoolExecutor(int(os.environ.get("WORKERS", "8"))) as ex:
            for fut in cf.as_completed([ex.submit(venvs.setup, m) for m in ms]):
                print(*fut.result(), flush=True)
        sys.exit()
    fn = {"init": run100.do_init, "run": run100.do_run, "compile": run100.do_compile, "all": run100.do_all}[cmd]
    with cf.ThreadPoolExecutor(int(os.environ.get("WORKERS", "6"))) as ex:
        futs = {ex.submit(fn, n): n for n in names}
        for fut in cf.as_completed(futs):
            n = futs[fut]
            try:
                _, res = fut.result()
                if isinstance(res, dict):
                    print(n, "init", {k: v.get("rc") for k, v in res.items() if isinstance(v, dict)}, flush=True)
                else:
                    print(n, [(r.get("config"), r.get("verdict")) for r in res], flush=True)
            except Exception as e:
                print(n, "ERROR", repr(e), flush=True)
