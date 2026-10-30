"""Simulate an FMU with fmpy.simulate_fmu and record every output at every communication point.

    python sim.py <fmu> <out.json> [--stop 3] [--h 1]

Numeric outputs come from simulate_fmu's result; String outputs are read from the instance in
step_finished (t > start only). The instance is created here and freed in `finally` (FMPy
doesn't free it when the FMU errors), and the UniFMU backend processes this process started are
killed before exiting.
"""
import argparse
import json
import math
import traceback
from pathlib import Path

import psutil


def fmu_log(a, unzip):
    """After a failure: simulate once more with FMU logging on, to get the model's error message."""
    if unzip is None:
        return None
    msgs = []

    def logger(*args):   # FMI 2: (env, name, status, category, message); FMI 3: (env, status, category, message)
        message = args[-1]
        msgs.append(message.decode("utf-8", "replace") if isinstance(message, bytes) else str(message))

    inst = None
    try:
        from fmpy import instantiate_fmu, read_model_description, simulate_fmu
        md = read_model_description(unzip)
        inst = instantiate_fmu(unzip, md, fmi_type="CoSimulation", debug_logging=True, logger=logger)
        simulate_fmu(unzip, model_description=md, fmu_instance=inst, stop_time=a.stop, output_interval=a.h)
    except BaseException:
        pass
    finally:
        if inst is not None:
            try:
                inst.freeInstance()
            except Exception:
                pass
        for c in psutil.Process().children(recursive=True):
            try:
                c.kill()
            except Exception:
                pass
    return "\n".join(m for m in msgs if m.strip())[-3000:]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("fmu"); ap.add_argument("out")
    ap.add_argument("--stop", type=float, default=3.0)
    ap.add_argument("--h", type=float, default=1.0)
    a = ap.parse_args()
    out = {}
    inst = None
    try:
        from fmpy import extract, instantiate_fmu, read_model_description, simulate_fmu
        import shutil, time as _time
        unzip = str(Path(a.out).with_suffix("")) + "_unz"
        shutil.rmtree(unzip, ignore_errors=True)
        extract(a.fmu, unzipdir=unzip)
        md = read_model_description(unzip)
        out["fmi_version"] = md.fmiVersion
        outs = [v for v in md.modelVariables if v.causality == "output"]
        out["outputs"] = [{"name": v.name, "type": v.type} for v in outs]
        numeric = [v.name for v in outs if v.type not in ("String", "Binary", "Clock")]
        strings = [v for v in outs if v.type == "String"]
        binaries = [v for v in outs if v.type == "Binary"]   # recorded as sha256:<hex>:<length>
        str_rows = {}
        # Windows Application Control sometimes blocks the freshly extracted unifmu.dll
        # (WinError 4551) on the first load; retry a few times before calling it a failure
        for attempt in range(6):
            try:
                inst = instantiate_fmu(unzip, md, fmi_type="CoSimulation")
                break
            except Exception as e:
                if "4551" not in str(e) or attempt == 5:
                    raise
                out.setdefault("dll_retries", 0)
                out["dll_retries"] += 1
                _time.sleep(5)

        def step_finished(time, recorder):
            if strings:
                vals = inst.getString([v.valueReference for v in strings])
                vals = [x.decode("utf-8") if isinstance(x, bytes) else x for x in vals]   # fmpy returns bytes
                str_rows[repr(round(time, 9))] = dict(zip([v.name for v in strings], vals))
            if binaries:
                import hashlib
                vals = inst.getBinary([v.valueReference for v in binaries])
                str_rows.setdefault(repr(round(time, 9)), {}).update(
                    {v.name: f"sha256:{hashlib.sha256(bytes(b)).hexdigest()}:{len(b)}" for v, b in zip(binaries, vals)})
            return True

        res = simulate_fmu(unzip, model_description=md, fmu_instance=inst, stop_time=a.stop,
                           output_interval=a.h, output=numeric, step_finished=step_finished)
        rows = []
        for r in res:
            t = float(r["time"])
            vals = {}
            for n in numeric:
                x = r[n]
                if getattr(x, "ndim", 0):   # an FMI 3 array: name[i] / name[i,j], 1-based as FMI 2's scalars
                    import itertools
                    for idx in itertools.product(*[range(k) for k in x.shape]):
                        vals[f"{n}[{','.join(str(i + 1) for i in idx)}]"] = x[idx].item()
                    continue
                vals[n] = bool(x) if isinstance(x, (bool,)) or str(type(x)).endswith("bool_'>") else (float(x) if not isinstance(x, (int,)) else x)
                if hasattr(x, "item"):
                    vals[n] = x.item()
            vals.update(str_rows.get(repr(round(t, 9)), {}))
            rows.append({"t": t, "values": vals})
        out["rows"] = rows
    except BaseException as e:
        out["error"] = f"{type(e).__name__}: {e}"
        out["traceback"] = traceback.format_exc()
        out["fmu_log"] = fmu_log(a, unzip if "unzip" in dir() else None)
    finally:
        if inst is not None:
            try:
                inst.freeInstance()
            except Exception:
                pass
        for c in psutil.Process().children(recursive=True):
            try:
                c.kill()
            except Exception:
                pass
    Path(a.out).write_text(json.dumps(out, default=repr, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
