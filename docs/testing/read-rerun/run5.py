"""Rerun a model with init --probe --read: venv, init, build, simulate 3 s, direct run twice, compare.

    python run5.py MODEL   (from a copy of this script in a work folder; each model gets MODEL/ there)

The 2026-11-04 rerun of tested-models.md's first failure cause, on Windows. flopy and OpenHENS were given
through EXTRA_MODELS (their --start paths point into the work folder), OpenHENS with CFG_DIR and MODEL_CWD
set to its cloned repository and couenne on PATH. FMUGEN_REPO: the fmugen checkout (default: this one).
"""
import json, subprocess, sys, os, shutil, math
from pathlib import Path

REPO = Path(os.environ.get("FMUGEN_REPO", Path(__file__).resolve().parents[3]))
HERE = Path(__file__).resolve().parent
DIRECT = REPO / "docs/testing/pending-100/harness/direct.py"
MODELS = {
    "openmdao": (["openmdao"], ["openmdao.api:Problem", "--const",
                 "model=call:openmdao.test_suite.components.sellar:SellarNoDerivatives()",
                 "--read", "y1=attr:y1.0", "--read", "y2=attr:y2.0", "--read", "obj=attr:obj.0",
                 "--read", "con1=attr:con1.0", "--read", "con2=attr:con2.0"]),
    "jsbsim": (["jsbsim"], ["jsbsim:FGFDMExec", "--const", "root_dir=call:jsbsim:get_default_root_dir()",
               "--setup", "load_model('c172x')", "--setup", "run_ic", "--call", "run",
               "--read", "sim_time=attr:simulation/sim-time-sec", "--read", "altitude_ft=attr:position/h-sl-ft",
               "--read", "airspeed_kts=attr:velocities/vc-kts", "--read", "pitch_rad=attr:attitude/theta-rad"]),
    "pybamm": (["pybamm==26.10.1.1"], ["pybamm:Simulation", "--call", "step",
               "--const", "model=call:pybamm.lithium_ion:SPM()", "--start=dt=100.0",
               "--read", "t=return:t.-1", "--read", "voltage=return:Voltage [V].entries.-1"]),
}
MODELS.update(json.loads(os.environ.get("EXTRA_MODELS", "{}")))


def sh(cmd, log, **kw):
    log.write(f"\n$ {' '.join(map(str, cmd))}\n"); log.flush()
    r = subprocess.run([str(c) for c in cmd], stdout=log, stderr=subprocess.STDOUT, **kw)
    return r.returncode


def main(name):
    deps, args = MODELS[name]
    d = HERE / name
    d.mkdir(exist_ok=True)
    py = d / ".venv/Scripts/python.exe"
    res = {"model": name}
    with open(d / "log.txt", "a", encoding="utf-8") as log:
        if not py.exists():
            sh(["uv", "venv", "--python", "3.13", d / ".venv"], log)
            if sh(["uv", "pip", "install", "--python", py, "-e", REPO, "fmpy", *deps], log):
                res["error"] = "install failed"; return res
        res["versions"] = subprocess.run([py, "-m", "pip", "--version"], capture_output=True, text=True).stdout.strip()
        res["freeze"] = subprocess.run(["uv", "pip", "freeze", "--python", str(py)], capture_output=True, text=True).stdout
        cfg = Path(os.environ["CFG_DIR"]) if os.environ.get("CFG_DIR") else d / "cfg"; cfg.mkdir(exist_ok=True)
        cwd = os.environ.get("MODEL_CWD") or cfg
        rc = sh([py, "-m", "fmugen", "init", *args, "--probe", "-o", cfg / "fmugen.toml", "--force"], log, cwd=cwd)
        if rc:
            res["error"] = "init failed"; return res
        res["config"] = (cfg / "fmugen.toml").read_text(encoding="utf-8")
        fmu = d / f"{name}.fmu"
        if sh([py, "-m", "fmugen", "build", cfg / "fmugen.toml", "-o", fmu, "--capture-output"], log, cwd=cfg):
            res["error"] = "build failed"; return res
        sim = d / "sim.json"
        code = ("import fmpy, json, sys, numpy as np\n"
                f"r = fmpy.simulate_fmu(r'{fmu}', stop_time=3.0, output_interval=1.0)\n"
                f"json.dump({{'names': list(r.dtype.names), 'rows': [[float(x) for x in row] for row in r]}}, open(r'{sim}', 'w'))\n")
        if sh([py, "-c", code], log, cwd=cfg):
            res["error"] = "simulate failed"; return res
        for i in (1,):   # direct.py runs it twice itself
            sh([py, DIRECT, cfg / "fmugen.toml", d / f"direct{i}.json", "--steps", "3", "--h", "1.0", "--stop", "3.0"], log,
               cwd=cfg)
    s = json.loads(sim.read_text())
    runs = json.loads((d / "direct1.json").read_text())["runs"]
    res["fmu"] = s
    res["direct_error"] = [r.get("error") for r in runs if r.get("error")]
    if res["direct_error"]:
        return res
    d1, d2 = runs
    res["random"] = d1["rows"] != d2["rows"]
    diffs, compared = [], 0
    for row in d1["rows"]:
        t = row["t"]
        srow = next((r for r in s["rows"] if abs(r[0] - t) < 1e-9), None)
        if srow is None:
            continue
        for k, v in row["values"].items():
            if isinstance(v, dict) and "error" in v:
                diffs.append((t, k, "direct error", v["error"])); continue
            if k in s["names"] and isinstance(v, (int, float)):
                f = srow[s["names"].index(k)]
                compared += 1
                if not math.isclose(f, v, rel_tol=1e-9, abs_tol=1e-12):
                    diffs.append((t, k, f, v))
    res["compared"], res["diffs"] = compared, diffs
    return res


if __name__ == "__main__":
    out = main(sys.argv[1])
    (HERE / sys.argv[1] / "result.json").write_text(json.dumps(out, indent=1, default=str))
    print(json.dumps({k: out.get(k) for k in ("model", "error", "compared", "diffs", "random", "direct_error")}, default=str))
