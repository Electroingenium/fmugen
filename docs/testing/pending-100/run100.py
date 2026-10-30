"""Test the 100 pending models as the others were (docs/testing/tested-models.md, "How the models were tested").

    python run100.py init    [name ...]   fmugen init --probe and plain init
    python run100.py run     [name ...]   build + simulate + direct run (twice) + compare, each config
    python run100.py compile [name ...]   the best matching config: build --compile pyinstaller, simulate, compare
    python run100.py all     [name ...]   the three in turn, per model
    python run100.py report               results.json and results.csv next to this script

Reuses the comparison harness (harness/: run_all.py, sim.py, direct.py). Venvs: ~/cmp/v100/<name>
(venvs.py). Work: ~/cmp/w100/m/<name>. FMI 3 models (extra fmi = 3) are initialized and built with
--fmi 3.
"""
import concurrent.futures as cf
import csv
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
HOME = Path.home()
os.environ.setdefault("CMP_VENVS", str(HOME / "cmp/v100"))
os.environ.setdefault("CMP_WORK", str(HOME / "cmp/w100"))
os.environ.setdefault("CMP_UNIFMU", str(HOME / "cmp/unifmu/unifmu"))
H = HERE / "harness"   # a copy of the comparison harness (~/cmp/harness), with the fixes of this run
sys.path.insert(0, str(H))
sys.path.insert(0, str(HERE))
import run_all as R  # noqa: E402
from models100 import MODELS as M100  # noqa: E402

DATA = HOME / "cmp/w100/data"
BUILD_TIMEOUT = 60 * 60
PREFERENCE = ("seeded", "probe", "plain", "hand")
EXTRA = {m[0]: m[3] for m in M100}


def expand(a, name):
    return a.replace("<data>", str(DATA)).replace("<m>", str(R.WORK / "m" / name))


def as_run_all(m):
    name, pkgs, args, extra = m
    args = [expand(a, name) for a in args]
    if extra.get("call_factory"):
        args += ["--call", extra["call_factory"]]
    if extra.get("fmi") == 3:
        args += ["--fmi", "3"]
    return (name, name, args, {k: v for k, v in extra.items() if k in ("seeded", "env", "hand")})


R.MODELS = [as_run_all(m) for m in M100]
R.HAND = HERE / "hand"   # hand configs: hand/<name>/fmugen.toml (extra hand = name)
_run = R.run


def run(cmd, log, timeout=R.TIMEOUT, cwd=None, retries=0, env=None):
    """run_all.run, building FMI 3 models as FMI 3 (run_all passes --fmi 2) and with longer timeouts."""
    cmd = list(cmd)
    if "build" in cmd and "--fmi" in cmd:
        name = Path(log).parent.name
        if EXTRA.get(name, {}).get("fmi") == 3:
            cmd[cmd.index("--fmi") + 1] = "3"
        timeout = max(timeout, BUILD_TIMEOUT)
    return _run(cmd, log, timeout=timeout, cwd=cwd, retries=0, env=env)


R.run = run


def prepare(name):
    d = R.mdir(name)
    w = EXTRA[name].get("wrap")
    if w:
        shutil.copy(HERE / "wrappers" / w, d / w)
    return d


def do_init(name):
    prepare(name)
    return R.do_init(name)


def do_run(name):
    return R.do_model(name)


def do_compile(name):
    d = R.mdir(name)
    f = d / "results.json"
    recs = json.loads(f.read_text()) if f.exists() else []
    ok = {r["config"]: r for r in recs if r.get("verdict") == "match"}
    # no matching config: the first config whose FMU simulated (not comparable, e.g. random or strings only)
    sims = {r["config"]: r for r in recs if r.get("fmu_rows")}
    cfg = next((c for c in PREFERENCE if c in ok), None) or next((c for c in PREFERENCE if c in sims), None)
    rec = {"model": name}
    if cfg is None:
        rec.update(verdict="no config to compile")
        (d / "compiled.json").write_text(json.dumps(rec, indent=1))
        return name, [rec]
    ref = ok.get(cfg) or sims[cfg]
    toml = ref["toml"]
    fmi = "3" if EXTRA[name].get("fmi") == 3 else "2"
    env = R.model_env(name)
    rec.update(config=cfg, toml=toml, fmi=fmi)
    fmu = d / f"{cfg}.compiled.fmu"
    if fmu.exists():
        fmu.unlink()
    rc, secs = _run([*R.fmugen(name), "build", toml, "-o", str(fmu), "--capture-output", "--fmi", fmi,
                     "--compile", "pyinstaller"], d / "build_compiled.log", timeout=BUILD_TIMEOUT, cwd=R.init_cwd(name), env=env)
    rec["build"] = {"rc": rc, "secs": secs}
    if rc != 0 or not fmu.exists():
        rec.update(verdict="build fails", notes=R.tail(d / "build_compiled.log", 10))
    else:
        rec["size_mb"] = round(fmu.stat().st_size / 1e6, 1)
        out = d / "sim_compiled.json"
        if out.exists():
            out.unlink()
        h, stop = R.sim_params(toml)
        sim_env = {k: v for k, v in env.items() if k != "PYTHONPATH"}
        rc, ssecs = _run([R.py(name), str(H / "sim.py"), str(fmu), str(out), "--stop", str(stop), "--h", str(h)],
                         d / "sim_compiled.log", timeout=R.TIMEOUT, cwd=str(d), env=sim_env)
        s = json.loads(out.read_text()) if out.exists() else {"error": f"sim.py rc={rc}: " + R.tail(d / "sim_compiled.log")}
        if "error" in s:
            log = [l for l in (s.get("fmu_log") or "").splitlines() if l.strip()]
            why = [l for l in log if "rror" in l][-4:] or log[-4:]
            rec.update(verdict="FMU fails", notes=(s["error"] + (" | FMU log: " + " / ".join(why) if why else ""))[:900])
        else:
            rnd = set(ref.get("random_outputs") or [])
            keep = lambda rows: [{"t": r["t"], "values": {n: v for n, v in r["values"].items() if n not in rnd}} for r in rows]
            c = R.compare_rows(keep(s["rows"]), keep(ref["fmu_rows"]))
            rec["compare"] = {k: v for k, v in c.items() if k != "skipped"}
            rec["fmu_rows"] = s["rows"]
            rec["verdict"] = "mismatch" if c["mismatches"] else ("match" if c["compared"] else "not comparable")
    (d / "compiled.json").write_text(json.dumps(rec, indent=1, default=repr))
    return name, [rec]


def do_all(name):
    do_init(name)
    do_run(name)
    return do_compile(name)


def report():
    rows = []
    for m in M100:
        d = R.WORK / "m" / m[0]
        init = json.loads((d / "init.json").read_text()) if (d / "init.json").exists() else {}
        recs = json.loads((d / "results.json").read_text()) if (d / "results.json").exists() else []
        comp = json.loads((d / "compiled.json").read_text()) if (d / "compiled.json").exists() else {}
        rows.append({"model": m[0], "entry": m[2][0], "packages": m[1], "fmi": m[3].get("fmi", 2),
                     "init": {k: v.get("rc") for k, v in init.items() if isinstance(v, dict)},
                     "configs": [{k: r.get(k) for k in ("config", "verdict", "stage", "notes", "fmu_outputs", "random_outputs", "direct_error")}
                                 | {"compared": (r.get("compare") or {}).get("compared"), "max_rel": (r.get("compare") or {}).get("max_rel"),
                                    "last": (r.get("fmu_rows") or [{}])[-1].get("values")} for r in recs],
                     "compiled": {k: comp.get(k) for k in ("config", "verdict", "size_mb", "notes", "build")}
                                 | {"compared": (comp.get("compare") or {}).get("compared"), "max_rel": (comp.get("compare") or {}).get("max_rel")},
                     "versions": versions(m[0], m[1])})
    (HERE / "results.json").write_text(json.dumps(rows, indent=1, default=repr, ensure_ascii=False))
    with open(HERE / "results.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["model", "config", "verdict", "outputs compared", "max rel diff", "notes"])
        for r in rows:
            for c in r["configs"] or [{"config": "-", "verdict": "not run"}]:
                w.writerow([r["model"], c["config"], c["verdict"], c.get("compared") or "", c.get("max_rel") if c.get("max_rel") is not None else "", (c.get("notes") or "")[:300]])
            cp = r["compiled"]
            w.writerow([r["model"], f"compiled ({cp.get('config')})", cp.get("verdict"), cp.get("compared") or "", cp.get("max_rel") if cp.get("max_rel") is not None else "", (cp.get("notes") or "")[:300]])
    print("wrote", HERE / "results.json")


def versions(name, pkgs):
    import re
    r = subprocess.run([str(HOME / ".local/bin/uv"), "pip", "list", "--python", R.py(name), "--format", "json"], capture_output=True, text=True)
    try:
        have = {x["name"].lower().replace("_", "-"): x["version"] for x in json.loads(r.stdout)}
    except Exception:
        return {}
    keys = [re.split(r"[\[<>=]", p)[0].lower().replace("_", "-") for p in pkgs] + ["torch"]
    return {k: have[k] for k in keys if k in have}


if __name__ == "__main__":
    cmd, names = sys.argv[1], sys.argv[2:]
    if cmd == "report":
        report(); sys.exit()
    names = names or [m[0] for m in M100]
    fn = {"init": do_init, "run": do_run, "compile": do_compile, "all": do_all}[cmd]
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
