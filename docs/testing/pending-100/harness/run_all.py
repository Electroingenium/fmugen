"""Drive the FMU-vs-direct comparison (fmu-vs-direct-brief.md).

    python run_all.py init   [name ...]   fmugen init --probe / plain for each model
    python run_all.py run    [name ...]   build + simulate + direct run + compare every config
    python run_all.py report              comparison/results.csv (+ results.json for report.md)

Work files: C:/w/m/<name>/ (configs, FMUs, logs, JSON results). Venvs: C:/v/<group>.
Every subprocess has a hard timeout and its whole process tree is killed when it expires.
"""
import concurrent.futures as cf
import json
import math
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import psutil

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from models import MODELS, GROUPS, VERSION_OF  # noqa: E402

ROOT = HERE.parents[1]
REPO = ROOT / "repo"
LINUX = os.name != "nt"
VROOT = Path(os.environ.get("CMP_VENVS", "C:/v"))
WORK = Path(os.environ.get("CMP_WORK", "C:/w"))
UNIFMU = os.environ.get("CMP_UNIFMU", "C:/v/unifmu/unifmu.exe")
EX = WORK / "examples"
GK = Path(os.environ.get("CMP_GK", str(Path.home() / "cmp/gk/w")))   # GEKKO work copies (gekko/prepare.sh)
TIMEOUT = 15 * 60
REL_TOL = 1e-9
HAND = HERE / "hand"          # hand configs (written from docs/testing/tested-models.md and docs/models.md)


def py(group):
    return str(VROOT / group / ("bin/python" if LINUX else "Scripts/python.exe"))


def fmugen(group):
    # `python -m fmugen`, not the fmugen.exe launcher: Smart App Control blocks unsigned launchers at times
    return [py(group), "-m", "fmugen"]


SAC_MARKERS = ("4551", "Control de aplicaciones", "Application Control")


def blocked(log):
    try:
        return any(m in Path(log).read_text(encoding="utf-8", errors="replace") for m in SAC_MARKERS)
    except OSError:
        return False


def run(cmd, log, timeout=TIMEOUT, cwd=None, retries=3, env=None):
    """run_once, retried when Windows Smart App Control blocked a binary (often transient)."""
    for attempt in range(retries + 1):
        try:
            rc, secs = run_once(cmd, log, timeout, cwd, env)
        except OSError as e:
            if getattr(e, "winerror", None) != 4551 or attempt == retries:
                raise
            time.sleep(10); continue
        if rc != 0 and blocked(log) and attempt < retries:
            time.sleep(10); continue
        return rc, secs


def run_once(cmd, log, timeout=TIMEOUT, cwd=None, extra_env=None):
    """Run cmd with output to log file; kill the whole tree on timeout. -> (returncode or 'timeout', seconds)."""
    t0 = time.time()
    with open(log, "w", encoding="utf-8", errors="replace") as f:
        f.write("$ " + subprocess.list2cmdline(cmd) + "\n"); f.flush()
        env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1", FMUGEN_UNIFMU=UNIFMU)
        for k, v in (extra_env or {}).items():
            env[k] = v + os.pathsep + env[k] if k in ("PATH", "LD_LIBRARY_PATH", "PYTHONPATH") and env.get(k) else v
        p = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, cwd=cwd, env=env)
        try:
            rc = p.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            rc = "timeout"
        finally:
            try:
                for c in psutil.Process(p.pid).children(recursive=True):
                    c.kill()
            except psutil.NoSuchProcess:
                pass
            if p.poll() is None:
                p.kill()
    return rc, round(time.time() - t0, 1)


def model(name):
    for m in MODELS:
        if m[0] == name:
            return m
    raise KeyError(name)


def mdir(name):
    d = WORK / "m" / name
    d.mkdir(parents=True, exist_ok=True)
    return d


def placeholders(group):
    """Values of <...> placeholders computed in the model's venv."""
    out = {}
    if group == "pvlib":
        out["<pvlib_tmy3>"] = subprocess.run([py(group), "-c", "import pvlib, pathlib; print((pathlib.Path(pvlib.__file__).parent/'data'/'723170TYA.CSV').as_posix())"], capture_output=True, text=True).stdout.strip()
    if group == "heartpy":
        out["<heartpy_csv>"] = subprocess.run([py(group), "-c", "import heartpy, pathlib; print((pathlib.Path(heartpy.__file__).parent/'data'/'data.csv').as_posix())"], capture_output=True, text=True).stdout.strip()
    return out


def expand(a):
    """<work> (the copy of the clone's examples/) and <gk> (the GEKKO work copies)."""
    return a.replace("<work>", EX.as_posix()).replace("<gk>", GK.as_posix())


def init_args(name):
    _, group, args, extra = model(name)
    ph = placeholders(group)
    res = []
    for a in args:
        a = expand(a)
        for k, v in ph.items():
            a = a.replace(k, v)
        res.append(a)
    return res


def init_cwd(name):
    """The folder `init` runs in: a package entry's repository copy (`extra["cwd"]`), else none."""
    cwd = model(name)[3].get("cwd")
    return expand(cwd) if cwd else None


def model_env(name):
    """Extra environment of every command for this model: `extra["env"]`, plus PYTHONPATH for a
    package entry imported from its repository copy."""
    extra = model(name)[3]
    env = {k: expand(v) for k, v in extra.get("env", {}).items()}
    if init_cwd(name):
        env["PYTHONPATH"] = init_cwd(name)
    return env


def config_dir(name):
    """Where init writes: next to a model file for file entries (init refuses elsewhere), in the
    repository copy for package entries run from it."""
    _, _, args, extra = model(name)
    if init_cwd(name):
        # the repository copy's root: a folder above a file entry is allowed too (CGE: next to Data/)
        return Path(init_cwd(name))
    if args and (args[0].startswith("<work>/") or args[0].startswith("<gk>/")):
        return Path(expand(args[0]).rsplit(":", 1)[0]).parent   # rsplit: "C:/gk/..." on Windows
    return mdir(name)


# ------------------------------------------------------------------ init

def do_init(name):
    _, group, args, extra = model(name)
    d = mdir(name)
    res = {}
    if args is None:
        return name, {"skipped": "skip_init"}
    cdir = config_dir(name)
    a = init_args(name)
    for kind, flag in (("probe", ["--probe"]), ("plain", [])):
        toml = cdir / f"{name}.{kind}.toml" if cdir != d else d / f"{kind}.toml"
        rc, secs = run([*fmugen(group), "init", *a, *flag, "--force", "-o", str(toml)], d / f"init_{kind}.log",
                       cwd=init_cwd(name), env=model_env(name))
        res[kind] = {"rc": rc, "secs": secs, "toml": str(toml) if toml.exists() else None}
    (d / "init.json").write_text(json.dumps(res, indent=1))
    return name, res


# ------------------------------------------------------------------ configs to test

def configs(name):
    _, group, args, extra = model(name)
    d = mdir(name)
    out = []
    init = json.loads((d / "init.json").read_text()) if (d / "init.json").exists() else {}
    for kind in ("probe", "plain"):
        t = init.get(kind, {}).get("toml")
        out.append((kind, t))
    if extra.get("repo") and not extra.get("fmi3"):
        out.append(("repo", str(EX / extra["repo"] / "fmugen.toml")))
    # models random by design: the probe config, seeded (brief: "compare seeded if possible")
    probe = init.get("probe", {}).get("toml")
    if (extra.get("seeded") or extra.get("seeded_append")) and probe and Path(probe).exists():
        text = Path(probe).read_text(encoding="utf-8")
        for old, new in extra.get("seeded", []):
            text = text.replace(old, new)
        text += extra.get("seeded_append", "")
        # next to the probe config, where its entry path points
        seeded = Path(probe).with_name("seeded.toml" if Path(probe).parent == d else f"{name}.seeded.toml")
        seeded.write_text("# the `init --probe` config, seeded by the comparison harness\n" + text, encoding="utf-8")
        out.append(("seeded", str(seeded)))
    # a hand config made from the probe config by adding outputs (OpenHENS: the one docs add by hand)
    if extra.get("hand_outputs") and probe and Path(probe).exists():
        hand = Path(probe).with_name(Path(probe).name.replace("probe", "hand"))
        text = Path(probe).read_text(encoding="utf-8")
        if "\n[outputs]\n" in text:
            text = text.replace("\n[outputs]\n", "\n[outputs]\n" + extra["hand_outputs"], 1)
        else:
            text += "\n[outputs]\n" + extra["hand_outputs"]
        hand.write_text("# the `init --probe` config, with the output docs/testing/tested-models.md adds by hand\n" + text,
                        encoding="utf-8")
        out.append(("hand", str(hand)))
    # hand configs: the first is config "hand", others "hand:<dir>" (e.g. JOS3's attr:t_core.0 variant)
    hands = extra.get("hand") or []
    for i, hd in enumerate([hands] if isinstance(hands, str) else hands):
        h = HAND / hd / "fmugen.toml"
        if h.exists() and extra.get("src_dir"):
            # a file entry: the config goes next to the model file, where its entry path points
            copy = Path(expand(extra["src_dir"])) / f"{name}.{hd}.hand.toml"
            copy.write_text(h.read_text(encoding="utf-8"), encoding="utf-8")
            h = copy
        out.append(("hand" if i == 0 else f"hand:{hd}", str(h) if h.exists() else None))
    return out


def sim_params(toml):
    import tomllib
    cfg = tomllib.loads(Path(toml).read_text(encoding="utf-8"))
    exp = cfg.get("experiment", {})
    if exp.get("fixed_step"):
        h = float(exp["step_size"])
        return h, 3 * h
    return 1.0, 3.0


# ------------------------------------------------------------------ compare

def num(x):
    if isinstance(x, bool):
        return float(x)
    if isinstance(x, (int, float)):
        return float(x)
    return None


def rel_diff(a, b):
    if a == b or (math.isnan(a) and math.isnan(b)):
        return 0.0
    if math.isinf(a) or math.isinf(b) or math.isnan(a) or math.isnan(b):
        return math.inf
    return abs(a - b) / max(abs(a), abs(b))


def compare_rows(fmu_rows, dir_rows):
    """-> dict with compared, max_rel, mismatches [(t, name, fmu, direct, rel)], skipped."""
    by_t = {round(r["t"], 6): r for r in dir_rows}
    compared, mism, max_rel, skipped = 0, [], 0.0, []
    for r in fmu_rows:
        t = round(r["t"], 6)
        d = by_t.get(t)
        for n, fv in r["values"].items():
            dv = None if d is None else d["values"].get(n)
            if dv is None or isinstance(dv, dict):
                skipped.append((t, n, dv.get("error") if isinstance(dv, dict) else "no direct value"))
                continue
            compared += 1
            if isinstance(fv, str) or isinstance(dv, str):
                if str(fv) != str(dv):
                    mism.append((t, n, fv, dv, None))
                continue
            a, b = num(fv), num(dv)
            if a is None or b is None:
                mism.append((t, n, fv, dv, None)); continue
            rd = rel_diff(a, b)
            max_rel = max(max_rel, rd)
            if rd > REL_TOL:
                mism.append((t, n, fv, dv, rd))
    return {"compared": compared, "max_rel": max_rel, "mismatches": mism, "skipped": skipped}


# ------------------------------------------------------------------ one config

def do_config(name, kind, toml):
    _, group, args, extra = model(name)
    d = mdir(name)
    rec = {"model": name, "config": kind, "toml": toml, "group": group}
    if toml is None or not Path(toml).exists():
        rec.update(verdict="no config", notes=f"`init` wrote no config (see init_{kind}.log)" if kind in ("probe", "plain") else "no hand config")
        return rec
    fmu = d / f"{kind}.fmu"
    if fmu.exists():
        fmu.unlink()
    env = model_env(name)
    rc, secs = run([*fmugen(group), "build", toml, "-o", str(fmu), "--capture-output", "--fmi", "2"], d / f"build_{kind}.log",
                   cwd=init_cwd(name), env=env)
    rec["build"] = {"rc": rc, "secs": secs}
    h, stop = sim_params(toml)
    rec["h"], rec["stop"] = h, stop
    # direct run, twice, in separate processes
    dres = []
    for i in (1, 2):
        out = d / f"direct_{kind}_{i}.json"
        if out.exists():
            out.unlink()
        rc2, s2 = run([py(group), str(HERE / "direct.py"), toml, str(out), "--runs", "1", "--h", str(h), "--stop", str(stop)],
                      d / f"direct_{kind}_{i}.log", cwd=str(d), env=env)
        r = json.loads(out.read_text()) if out.exists() else {"runs": [{"error": f"direct.py rc={rc2}"}]}
        dres.append(r["runs"][0] if r.get("runs") else {"error": f"direct.py rc={rc2}"})
    rec["direct_error"] = dres[0].get("error")
    if rc != 0 or not fmu.exists():
        rec.update(verdict="FMU fails", stage="build", notes=tail(d / f"build_{kind}.log"))
        return rec
    sim_out = d / f"sim_{kind}.json"
    if sim_out.exists():
        sim_out.unlink()
    # the FMU gets no PYTHONPATH: it has to bring the model's files itself
    sim_env = {k: v for k, v in env.items() if k != "PYTHONPATH"}
    rc, secs = run([py(group), str(HERE / "sim.py"), str(fmu), str(sim_out), "--stop", str(stop), "--h", str(h)], d / f"sim_{kind}.log",
                   cwd=str(d), env=sim_env)
    rec["sim"] = {"rc": rc, "secs": secs}
    s = json.loads(sim_out.read_text()) if sim_out.exists() else {"error": "timeout (15 min)" if rc == "timeout" else f"sim.py rc={rc}: " + tail(d / f'sim_{kind}.log')}
    rec["fmu_outputs"] = [o["name"] for o in s.get("outputs", [])]
    if "error" in s:
        log = [l for l in (s.get("fmu_log") or "").splitlines() if l.strip()]
        why = [l for l in log if "Error" in l or "error" in l or "Traceback" not in l][-3:]
        rec.update(verdict="FMU fails", stage="simulate", fmu_log=s.get("fmu_log"),
                   notes=(s["error"] + (" | FMU log: " + " / ".join(why) if why else ""))[:700])
        return rec
    rec["fmu_rows"] = s["rows"]
    if "error" in dres[0]:
        rec.update(verdict="direct fails", notes=dres[0]["error"][:400])
        return rec
    rec["direct_rows"] = dres[0]["rows"]
    # randomness: outputs whose two direct runs differ at all (even below the tolerance, e.g. a
    # clock-based seed) are random by design; they are left out, the others are still compared
    random_outs = set()
    if "error" not in dres[1]:
        by_t = {round(r["t"], 6): r["values"] for r in dres[1]["rows"]}
        for r in dres[0]["rows"]:
            other = by_t.get(round(r["t"], 6), {})
            for n, v in r["values"].items():
                if n in other and other[n] != v and not (v != v and other[n] != other[n]):
                    random_outs.add(n)
    rec["random_outputs"] = sorted(random_outs)
    keep = lambda rows: [{"t": r["t"], "values": {n: v for n, v in r["values"].items() if n not in random_outs}} for r in rows]
    c = compare_rows(keep(s["rows"]), keep(dres[0]["rows"]))
    rec["compare"] = {k: v for k, v in c.items()}
    random_by_design = bool(random_outs) and c["compared"] == 0
    if random_outs and c["compared"]:
        rec["notes"] = f"random by design, not compared: {', '.join(sorted(random_outs))[:200]}"
    if not s.get("outputs"):
        rec.update(verdict="not comparable", notes="the FMU has no outputs")
    elif random_by_design:
        rec.update(verdict="not comparable", notes="random: two direct runs differ")
    elif c["compared"] == 0:
        rec.update(verdict="not comparable", notes="no output could be read directly: " + "; ".join(sorted({str(x[2]) for x in c["skipped"]}))[:300])
    elif c["mismatches"]:
        rec.update(verdict="mismatch")
    else:
        rec.update(verdict="match")
    return rec


def tail(p, n=6):
    try:
        lines = [l for l in Path(p).read_text(encoding="utf-8", errors="replace").splitlines() if l.strip()]
        return " | ".join(lines[-n:])[-600:]
    except Exception:
        return ""


def do_model(name):
    _, group, args, extra = model(name)
    d = mdir(name)
    if extra.get("fmi3"):
        out = d / "fmi3.json"
        rc, secs = run([py(group), str(HERE / "fmi3_examples.py"), extra["repo"], str(EX / extra["repo"] / "fmugen.toml"), str(d), str(out)], d / "fmi3.log", cwd=str(d))
        recs = json.loads(out.read_text()) if out.exists() else [{"model": name, "config": "repo (FMI 3)", "verdict": "FMU fails", "notes": tail(d / "fmi3.log")}]
        for r in recs:
            r["model"] = name
        (d / "results.json").write_text(json.dumps(recs, indent=1, default=repr))
        return name, recs
    recs = [do_config(name, k, t) for k, t in configs(name)]
    (d / "results.json").write_text(json.dumps(recs, indent=1, default=repr))
    return name, recs


# ------------------------------------------------------------------ report

def versions():
    out = {}
    for g, reqs in GROUPS.items():
        p = Path(py(g))
        if not p.exists():
            continue
        r = subprocess.run([str(p), "-m", "pip", "list", "--format", "json"], capture_output=True, text=True)
        try:
            out[g] = {x["name"].lower().replace("_", "-"): x["version"] for x in json.loads(r.stdout)}
        except Exception:
            out[g] = {}
    return out


def lib_version(vers, name):
    _, group, args, extra = model(name)
    pk = vers.get(group, {})
    names = extra.get("dists") or ([VERSION_OF.get(group)] if VERSION_OF.get(group) else GROUPS[group][:1])
    parts = []
    for n in names:
        key = re.split(r"[<>=]", n)[0].lower().replace("_", "-")
        if key in pk:
            parts.append(f"{key} {pk[key]}")
    return ", ".join(parts) if parts else "(repo example)"


def report():
    import csv
    vers = versions()
    rows = []
    for m in MODELS:
        f = WORK / "m" / m[0] / "results.json"
        if not f.exists():
            rows.append({"model": m[0], "config": "-", "verdict": "not run"}); continue
        for r in json.loads(f.read_text()):
            rows.append(r)
    (HERE.parent / "results.json").write_text(json.dumps({"rows": rows, "versions": vers}, indent=1, default=repr))
    with open(HERE.parent / "results.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["model", "config used", "library version", "FMU simulates?", "outputs compared",
                    "max rel diff", "verdict", "notes"])
        for r in rows:
            c = r.get("compare", {})
            sims = "yes" if r.get("fmu_rows") else ("no (build)" if r.get("stage") == "build" else ("no" if r.get("stage") == "simulate" else ""))
            if r.get("fmi3_sim") is not None:
                sims = r["fmi3_sim"]
            mr = c.get("max_rel")
            w.writerow([r["model"], r.get("config"), lib_version(vers, r["model"]), sims,
                        c.get("compared", ""), "" if mr is None else f"{mr:.3g}", r.get("verdict"),
                        r.get("notes_final") or r.get("notes", "")])
    print("wrote", HERE.parent / "results.csv")


def merge(name, recs):
    """Replace (or add) these configs' records in the model's results.json, keeping the others."""
    f = mdir(name) / "results.json"
    old = json.loads(f.read_text()) if f.exists() else []
    new_cfgs = {r["config"] for r in recs}
    allr = [r for r in old if r.get("config") not in new_cfgs] + recs
    f.write_text(json.dumps(allr, indent=1, default=repr))
    return allr


def do_configs(pair):
    """'model:config': rerun just that config (build, simulate, direct, compare)."""
    name, kind = pair.split(":", 1)
    toml = dict(configs(name)).get(kind)
    merge(name, [do_config(name, kind, toml)])
    return pair, [r for r in json.loads((mdir(name) / "results.json").read_text()) if r["config"] == kind]


def do_relog(name):
    """Re-simulate the FMUs that failed in simulation, only to record the FMU's error message."""
    f = mdir(name) / "results.json"
    if not f.exists():
        return name, []
    recs = json.loads(f.read_text())
    for r in recs:
        if r.get("stage") != "simulate" or r.get("fmu_log"):
            continue
        d = mdir(name)
        out = d / f"relog_{r['config']}.json"
        run([py(r["group"]), str(HERE / "sim.py"), str(d / f"{r['config']}.fmu"), str(out),
             "--stop", str(r.get("stop", 3.0)), "--h", str(r.get("h", 1.0))], d / f"relog_{r['config']}.log", cwd=str(d))
        s = json.loads(out.read_text()) if out.exists() else {}
        log = [l for l in (s.get("fmu_log") or "").splitlines() if l.strip()]
        why = [l for l in log if "rror" in l][-3:] or log[-3:]
        r["fmu_log"] = s.get("fmu_log")
        if why:
            r["notes"] = (r.get("notes", "") + " | FMU log: " + " / ".join(why))[:700]
    f.write_text(json.dumps(recs, indent=1, default=repr))
    return name, recs


if __name__ == "__main__":
    cmd, names = sys.argv[1], sys.argv[2:]
    if cmd == "report":
        report(); sys.exit()
    if cmd in ("configs", "relog"):
        fn = do_configs if cmd == "configs" else do_relog
        names = names or [m[0] for m in MODELS]
        with cf.ThreadPoolExecutor(int(os.environ.get("WORKERS", "8"))) as ex:
            futs = {ex.submit(fn, n): n for n in names}
            for fut in cf.as_completed(futs):
                try:
                    k, res = fut.result()
                    print(k, [(r.get("config"), r.get("verdict")) for r in res], flush=True)
                except Exception as e:
                    print(futs[fut], "ERROR", repr(e), flush=True)
        sys.exit()
    names = names or [m[0] for m in MODELS]
    fn = {"init": do_init, "run": do_model}[cmd]
    workers = int(os.environ.get("WORKERS", "8"))
    with cf.ThreadPoolExecutor(workers) as ex:
        futs = {ex.submit(fn, n): n for n in names}
        for fut in cf.as_completed(futs):
            n = futs[fut]
            try:
                _, res = fut.result()
                if cmd == "run":
                    print(n, [(r.get("config"), r.get("verdict")) for r in res], flush=True)
                else:
                    print(n, {k: v.get("rc") for k, v in res.items() if isinstance(v, dict)}, flush=True)
            except Exception as e:
                print(n, "ERROR", repr(e), flush=True)
