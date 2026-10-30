"""Run a model directly, as an fmugen.toml describes it, without fmugen.

Written from docs/config.md (the config reference), not from fmugen's runtime, so it is an
independent check of what the FMU computes.

    python direct.py <fmugen.toml> <out.json> [--steps 3] [--h 1.0] [--stop 3.0]

What it does, following the reference:
  * [model.constants] / [model.call_constants]: plain values, {python}, {ref}, {call}.
  * setup "module:function(args)" before construction, bare "method(args)" after it.
  * classes: construct (or [model] create classmethod) with constants + init:/pos: parameters;
    function entries with `call`: a factory called once with the parameters.
  * attr: / call: bindings applied after construction and before every step.
  * [model] init_call (default: true for functions with a step, false otherwise): one call at
    the start time, whose outputs are the t = start row; states are not advanced by it.
  * each step: the function / step method with arg:/pos: inputs, states, [time] arguments and
    call_constants; outputs read from `from` (return / return:path / attr:path), states from `next`.
  * [model] kind = "function": the class is constructed with the inputs on every step.

Writes {"rows": [{"t": t, "values": {name: value}}], "error": ..., "outputs": [...]}; array
outputs are flattened with FMI 2 names (x[1], x[1,2]). run_all.py runs this twice, in separate
processes, to detect models that are random by design.
"""
import ast
import builtins
import copy
import dataclasses
import datetime
import importlib
import importlib.util
import inspect
import json
import math
import os
import sys
import tomllib
import traceback
from pathlib import Path

try:
    import numpy as np
except ImportError:
    np = None


# ---------------------------------------------------------------- expressions

def import_dotted(path):
    """Import the longest importable module prefix of a dotted path and getattr the rest."""
    parts = path.split(".")
    for i in range(len(parts), 0, -1):
        try:
            obj = importlib.import_module(".".join(parts[:i]))
        except ImportError:
            continue
        for p in parts[i:]:
            obj = getattr(obj, p)
        return obj
    raise ImportError(f"cannot import {path!r}")


def resolve_name(dotted, root=None):
    """A dotted name: first on `root` (module or object), then builtins, then importable."""
    head, *rest = dotted.split(".")
    if rest and importlib.util.find_spec(head) is not None:   # pde.UnitGrid: the package, as fmugen reads it
        try:
            return import_dotted(dotted)
        except (ImportError, AttributeError):
            pass
    if root is not None and hasattr(root, head):
        obj = getattr(root, head)
    elif hasattr(builtins, head):
        obj = getattr(builtins, head)
    else:
        return import_dotted(dotted)
    for p in rest:
        obj = getattr(obj, p)
    return obj


def dotted_of(node):
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = dotted_of(node.value)
        return None if base is None else base + "." + node.attr
    return None


def ev(node, root=None):
    """Evaluate the restricted expression language of config.md."""
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        vals = [ev(e, root) for e in node.elts]
        return {ast.List: list, ast.Tuple: tuple, ast.Set: set}[type(node)](vals)
    if isinstance(node, ast.Dict):
        return {ev(k, root): ev(v, root) for k, v in zip(node.keys, node.values)}
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        v = ev(node.operand, root)
        return -v if isinstance(node.op, ast.USub) else +v
    if isinstance(node, ast.BinOp):  # e.g. 2*3 -- not in the reference, but harmless
        return eval(compile(ast.Expression(node), "<e>", "eval"), {})
    d = dotted_of(node)
    if d is not None:
        return resolve_name(d, root)
    if isinstance(node, ast.Call):
        fn = ev(node.func, root)
        args = [ev(a, root) for a in node.args]
        kwargs = {k.arg: ev(k.value, root) for k in node.keywords}
        return fn(*args, **kwargs)
    raise ValueError(f"unsupported expression: {ast.dump(node)}")


def eval_call_string(s, root_obj=None):
    """"module:attr.path(args)" (called) or "module:attr.path" (the object); for a bare
    "method(args)" root_obj is the object the method is looked up on."""
    if ":" in s.split("(")[0]:
        mod, rest = s.split(":", 1)
        root = import_dotted(mod)   # "pybamm.lithium_ion" is an attribute of pybamm, not a module
    else:
        rest, root = s, root_obj
    return ev(ast.parse(rest.strip(), mode="eval").body, root)


def const_value(v):
    if isinstance(v, dict):
        if set(v) == {"python"}:
            return ast.literal_eval(v["python"])
        if set(v) == {"ref"}:
            r = v["ref"]
            if ":" in r:
                mod, path = r.split(":", 1)
                return resolve_name(path, importlib.import_module(mod))
            return import_dotted(r)
        if set(v) == {"call"}:
            return eval_call_string(v["call"])
        return {k: const_value(x) for k, x in v.items()}
    return v


# ---------------------------------------------------------------- variables

def var_type(spec, section):
    if "type" in spec:
        return spec["type"]
    if "items" in spec or "enum" in spec:
        return "Enumeration"
    s = spec.get("start")
    if isinstance(s, list) and s:
        s = s[0]
    if isinstance(s, bool):
        return "Boolean"
    if isinstance(s, int) and not isinstance(s, bool):
        return "Integer"
    if isinstance(s, str):
        return "String"
    return "Real"


def default_start(t):
    return {"Boolean": False, "String": "", "Binary": b""}.get(t, 0.0 if t in ("Real", "Float64", "Float32") else 0)


def cast(t, v, spec=None):
    if t in ("Real", "Float64", "Float32"):
        return float(v)
    if t == "Boolean":
        return bool(v)
    if t == "String":
        return str(v)
    if t == "Binary":
        return bytes.fromhex(v) if isinstance(v, str) else bytes(v)
    if t == "Enumeration":
        if spec and "enum" in spec:
            mod, cls = spec["enum"].split(":")
            E = resolve_name(cls, importlib.import_module(mod))
            members = list(E)
            if isinstance(v, str):
                return E[v]
            return members[int(v) - 1]
        items = spec.get("items", []) if spec else []
        if isinstance(v, str):
            return items.index(v) + 1
        return int(v)
    return int(v)


def shape_of(spec):
    dims = spec.get("dimensions")
    return [int(d) for d in dims] if dims else None


def reshape(flat, shape):
    if len(shape) == 1:
        return flat
    n = len(flat) // shape[0]
    return [reshape(flat[i * n:(i + 1) * n], shape[1:]) for i in range(shape[0])]


def start_value(spec, section):
    t = var_type(spec, section)
    s = spec.get("start", None)
    shape = shape_of(spec)
    if shape:
        size = math.prod(shape)
        if s is None:
            flat = [default_start(t)] * size
        elif isinstance(s, list):
            flat = s
        else:
            flat = [s] * size
        flat = [cast(t, x, spec) for x in flat]
        val = reshape(flat, shape)
        if spec.get("numpy"):
            val = np.array(val)
    else:
        val = cast(t, default_start(t) if s is None else s, spec)
    conv = spec.get("convert")
    if conv == "pint":
        import pint
        ureg = pint.get_application_registry()
        val = ureg.Quantity(val, spec.get("unit", "dimensionless"))
    elif conv:
        mod, fn = conv.split(":")
        val = resolve_name(fn, importlib.import_module(mod))(val)
    return val


# ---------------------------------------------------------------- reading results

def step_path(obj, part):
    if isinstance(obj, dict):
        if part in obj:
            return obj[part]
        try:
            return obj[int(part)]
        except (ValueError, KeyError):
            return obj[part]
    if isinstance(obj, (tuple, list)) or (np is not None and isinstance(obj, np.ndarray)):
        try:
            return obj[int(part)]
        except ValueError:
            pass
    if part.lstrip("-").isdigit() and hasattr(obj, "__getitem__") and not hasattr(obj, part):
        return obj[int(part)]
    try:
        return getattr(obj, part)
    except AttributeError:
        if hasattr(obj, "__getitem__"):
            return obj[part]
        raise


def read_from(src, result, obj):
    if src == "return":
        return result
    kind, path = src.split(":", 1)
    base = result if kind == "return" else obj
    for p in path.split("."):
        base = step_path(base, p)
    return base


def to_plain(v, unit=None):
    """Value read from the model -> FMI-like Python value (pint, numpy, torch...)."""
    if hasattr(v, "magnitude") and hasattr(v, "to") and hasattr(v, "units"):
        v = v.to(unit).magnitude if unit else v.magnitude
    elif hasattr(v, "unit") and hasattr(v, "to_value"):   # an astropy Quantity (hapsira's orbits)
        v = v.to_value(unit) if unit else v.value
    if hasattr(v, "detach") and hasattr(v, "numpy"):
        v = v.detach().cpu().numpy()
    if np is not None:
        if isinstance(v, np.generic):
            return v.item()
        if isinstance(v, np.ndarray):
            return v.tolist()
    if hasattr(v, "tolist") and not isinstance(v, (str, bytes)):
        try:
            return v.tolist()
        except Exception:
            pass
    return v


def flatten(name, value, spec, t):
    """FMI 2 scalars: x[1], x[1,2] (1-based, row-major)."""
    shape = shape_of(spec)
    if isinstance(value, (bytes, bytearray)):   # a Binary output, as sim.py records it
        import hashlib
        return {name: f"sha256:{hashlib.sha256(bytes(value)).hexdigest()}:{len(value)}"}
    if not shape:
        if isinstance(value, list):
            # a scalar output reading a 1-element array
            flat = list(_iter_flat(value))
            if len(flat) == 1:
                value = flat[0]
        return {name: coerce(t, value, spec)}
    arr = value
    out = {}
    import itertools
    for idx in itertools.product(*[range(n) for n in shape]):
        x = arr
        for i in idx:
            x = x[i]
        out[f"{name}[{','.join(str(i + 1) for i in idx)}]"] = coerce(t, x, spec)
    return out


def _iter_flat(v):
    if isinstance(v, (list, tuple)):
        for x in v:
            yield from _iter_flat(x)
    else:
        yield v


def coerce(t, v, spec=None):
    if t == "Enumeration" and spec:
        # FMI Enumeration: the 1-based item number (config.md: items numbered from 1; with
        # `enum`, the items are the enum's members)
        if "enum" in spec:
            mod, cls = spec["enum"].split(":")
            members = list(resolve_name(cls, importlib.import_module(mod)))
            return members.index(v) + 1
        if isinstance(v, str):
            return spec.get("items", []).index(v) + 1
        return int(v)
    v = to_plain(v)
    if isinstance(v, list) and len(v) == 1:
        v = v[0]
    if t in ("Real", "Float64", "Float32"):
        return float(v)
    if t == "Boolean":
        return bool(v)
    if t == "String":
        return str(v)
    if t == "Enumeration":
        return v.value if hasattr(v, "value") else int(v)
    return int(v)


# ---------------------------------------------------------------- the run

class Runner:
    def __init__(self, cfg_path):
        self.path = Path(cfg_path).resolve()
        self.dir = self.path.parent
        self.cfg = tomllib.loads(self.path.read_text(encoding="utf-8"))
        self.m = self.cfg["model"]
        self.exp = self.cfg.get("experiment", {})
        self.timecfg = self.cfg.get("time", {})

    def import_entry(self):
        entry = self.m["entry"]
        target, name = entry.rsplit(":", 1)
        for s in self.m.get("sources", []):
            p = (self.dir / s)
            sys.path.insert(0, str(p if p.is_dir() else p.parent))
        if target.endswith(".py"):
            f = (self.dir / target).resolve()
            sys.path.insert(0, str(f.parent))
            mod = importlib.import_module(f.stem)
        else:
            mod = importlib.import_module(target)
        return resolve_name(name, mod)

    def variables(self):
        out = {}
        for sec in ("structural_parameters", "parameters", "inputs", "states"):
            for n, spec in self.cfg.get(sec, {}).items():
                out[n] = (sec, spec)
        return out

    def default_to(self, sec, name):
        if sec in ("parameters", "structural_parameters") and self.is_class and not self.kind_function:
            return f"init:{name}"
        return f"arg:{name}"

    def time_args(self, t, h):
        out = {}
        for arg, src in self.timecfg.items():
            epoch = None
            if isinstance(src, dict):
                epoch = src.get("epoch")
                src = src.get("source", "time")
            val = {"time": t, "step_size": h, "end_time": t + h, "start_time": self.t0,
                   "stop_time": self.stop, "tolerance": None}[src]
            if epoch is not None:
                val = datetime.datetime.fromisoformat(epoch) + datetime.timedelta(seconds=val)
            out[arg] = val
        return out

    def collect(self, prefix, values):
        """kwargs / positionals for bindings starting with prefix ('init' or 'arg')."""
        kwargs, pos, groups, fields = {}, {}, {}, {}
        for n, (sec, spec) in self.vars.items():
            to = spec.get("to", self.default_to(sec, n))
            v = values[n]
            if to.startswith(prefix + ":"):
                a = to[len(prefix) + 1:]
                if "[" in a:
                    base, key = a[:-1].split("[", 1)
                    key = key.strip("'\"")
                    groups.setdefault(base, {})[int(key) if key.lstrip("-").isdigit() else key] = v
                elif "." in a:
                    base, fld = a.split(".", 1)
                    fields.setdefault(base, {})[fld] = v
                else:
                    kwargs[a] = v
            elif to.startswith("pos:") and ((prefix == "init") == (sec in ("parameters", "structural_parameters") and self.is_class and not self.kind_function)):
                pos[int(to[4:])] = v
        for base, items in groups.items():
            if all(isinstance(k, int) for k in items):
                kwargs[base] = tuple(items[i] for i in range(len(items)))
            else:
                kwargs[base] = dict(items)
        for base, flds in fields.items():
            kwargs[base] = self.build_object(prefix, base, flds)
        return kwargs, pos

    def build_object(self, prefix, arg, flds):
        target = self.target_callable(prefix)
        p = inspect.signature(target).parameters[arg]
        default = p.default
        if default is not inspect.Parameter.empty and default is not None:
            if dataclasses.is_dataclass(default):
                return dataclasses.replace(default, **flds)
            if hasattr(default, "model_copy"):
                return default.model_copy(update=flds)
            if hasattr(default, "_replace"):
                return default._replace(**flds)
            try:
                import attrs
                return attrs.evolve(default, **flds)
            except Exception:
                pass
        ann = p.annotation
        if isinstance(ann, str):
            ann = eval(ann, getattr(target, "__globals__", {}))
        return ann(**flds)

    def target_callable(self, prefix):
        if prefix == "init":
            create = self.m.get("create")
            return getattr(self.entry, create) if create else self.entry
        if self.obj is not None:
            return getattr(self.obj, self.call)
        return self.entry

    def apply_setters(self, values, calls=True):
        """attr: setters; call: setters only when `calls` (config.md: a setter method runs once per
        model call, so not also after construction)."""
        for n, (sec, spec) in self.vars.items():
            to = spec.get("to", "")
            if to.startswith("attr:"):
                path = to[5:].split(".")
                o = self.obj
                for p in path[:-1]:
                    o = getattr(o, p)
                setattr(o, path[-1], values[n])
            elif to.startswith("call:") and calls:
                getattr(self.obj, to[5:])(values[n])

    def run_setup(self, after):
        for s in self.m.get("setup", []):
            if isinstance(s, dict):
                c = s["call"]
                args = [const_value(a) for a in s.get("args", [])]
                kwargs = {k: const_value(v) for k, v in s.get("kwargs", {}).items()}
                is_fn = ":" in c
                if is_fn != (not after):
                    continue
                if is_fn:
                    mod, path = c.split(":", 1)
                    resolve_name(path, importlib.import_module(mod))(*args, **kwargs)
                else:
                    getattr(self.obj, c)(*args, **kwargs)
                continue
            is_fn = ":" in s.split("(")[0]
            if is_fn != (not after):
                continue
            if is_fn:
                r = eval_call_string(s if "(" in s else s + "()")
            else:
                eval_call_string(s if "(" in s else s + "()", root_obj=self.obj)

    def invoke(self, fn, pos, kwargs):
        args = [pos[i] for i in range(len(pos))]
        return fn(*args, **kwargs)

    def call_model(self, t, h):
        values = self.values
        if self.obj is not None:
            self.apply_setters(values)
        kwargs, pos = self.collect("arg", values)
        kwargs.update(self.time_args(t, h))
        if self.obj is not None:
            # numbered call constants are positions (docs/config.md), counted with pos: inputs
            kwargs.update({k: v for k, v in self.call_constants.items() if not str(k).isdigit()})
            pos = {**{int(k): v for k, v in self.call_constants.items() if str(k).isdigit()}, **pos}
            # call_on_class: Model.step(...) on the class, not the object
            owner = type(self.obj) if self.m.get("call_on_class") else self.obj
            return self.invoke(getattr(owner, self.call), pos, kwargs)
        # function (or kind = "function" class)
        kw = dict(self.constants_kw)
        kw.update(kwargs)
        cpos = dict(self.constants_pos)
        cpos.update(pos)
        res = self.invoke(self.entry, cpos, kw)
        return res

    def read_outputs(self, result, have_result):
        row, ok = {}, True
        default_from = "return:{}" if (not self.is_class or self.kind_function) else "attr:{}"
        for n, spec in self.cfg.get("outputs", {}).items():
            src = spec.get("from", default_from.format(n))
            if src.startswith("return") and not have_result:
                continue
            base = result if (src.startswith("return") or self.obj is None) else None
            try:
                v = read_from(src, result, self.obj if self.obj is not None else result)
                v = to_plain(v, spec.get("unit") if hasattr(v, "units") else None)
                row.update(flatten(n, v, spec, var_type(spec, "outputs")))
            except Exception as e:
                row[n] = {"error": f"{type(e).__name__}: {e}"}
        return row

    def update_states(self, result):
        for n, spec in self.cfg.get("states", {}).items():
            v = read_from(spec["next"], result, self.obj)
            sp = dict(spec)
            # keep the value as the model returned it, cast scalars to the declared type
            if not shape_of(spec):
                v = cast(var_type(spec, "states"), to_plain(v) if not isinstance(to_plain(v), list) else to_plain(v)[0], spec)
            elif spec.get("numpy"):
                v = np.asarray(v)
            else:
                v = to_plain(v)
            self.values[n] = v

    def run(self, steps, h, stop):
        cwd = os.getcwd()
        if "cwd" in self.m:
            os.chdir(self.dir / self.m["cwd"])
        try:
            return self._run(steps, h, stop)
        finally:
            os.chdir(cwd)

    def _run(self, steps, h, stop):
        self.t0 = float(self.exp.get("start_time", 0.0))
        self.stop = stop
        self.entry = self.import_entry()
        self.is_class = inspect.isclass(self.entry)
        self.kind_function = self.m.get("kind") == "function"
        self.call = self.m.get("call", "__call__" if self.is_class else None)
        self.obj = None
        consts = {k: const_value(v) for k, v in self.m.get("constants", {}).items()}
        self.constants_kw = {k: v for k, v in consts.items() if not k.isdigit()}
        self.constants_pos = {int(k): v for k, v in consts.items() if k.isdigit()}
        self.call_constants = {k: const_value(v) for k, v in self.m.get("call_constants", {}).items()}
        self.vars = self.variables()
        self.values = {n: start_value(spec, sec) for n, (sec, spec) in self.vars.items()}
        self.run_setup(after=False)
        if self.is_class and not self.kind_function:
            kw, pos = self.collect("init", self.values)
            kwargs = dict(self.constants_kw); kwargs.update(kw)
            cpos = dict(self.constants_pos); cpos.update(pos)
            self.obj = self.invoke(self.target_callable("init"), cpos, kwargs)
            self.run_setup(after=True)
            self.apply_setters(self.values, calls=False)
        elif not self.is_class and self.m.get("call"):
            # factory: called once with the parameters
            kw, pos = {}, {}
            for n, (sec, spec) in self.vars.items():
                if sec in ("parameters", "structural_parameters"):
                    to = spec.get("to", f"arg:{n}")
                    if to.startswith("pos:"):
                        pos[int(to[4:])] = self.values[n]
                    else:
                        kw[to.split(":", 1)[1]] = self.values[n]
            kwargs = dict(self.constants_kw); kwargs.update(kw)
            cpos = dict(self.constants_pos); cpos.update(pos)
            # the factory's parameters are not passed to the step method
            self.vars = {n: v for n, v in self.vars.items() if v[0] not in ("parameters", "structural_parameters")}
            self.obj = self.invoke(self.entry, cpos, kwargs)
            self.run_setup(after=True)
            self.apply_setters(self.values, calls=False)
            self.is_class = True  # from here on it behaves like an object with a step method
        if "init_call" in self.m:
            init_call = self.m["init_call"]
        else:
            takes_h = any((v if isinstance(v, str) else v.get("source", "time")) == "step_size" for v in self.timecfg.values())
            init_call = (not self.is_class or self.kind_function) and not (takes_h and "step_size" not in self.exp)
        rows = []
        if init_call:
            res = self.call_model(self.t0, float(self.exp.get("step_size", 0.0)))
            rows.append({"t": self.t0, "values": self.read_outputs(res, True), "init_call": True})
        else:
            rows.append({"t": self.t0, "values": self.read_outputs(None, False), "init_call": False})
        t = self.t0
        for k in range(steps):
            res = self.call_model(t, h)
            row = self.read_outputs(res, True)
            self.update_states(res)
            t = t + h
            rows.append({"t": t, "values": row})
        return rows


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("toml"); ap.add_argument("out")
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--h", type=float, default=1.0)
    ap.add_argument("--stop", type=float, default=3.0)
    ap.add_argument("--runs", type=int, default=2)
    a = ap.parse_args()
    out = {"runs": []}
    for i in range(a.runs):
        try:
            r = Runner(a.toml)
            rows = r.run(a.steps, a.h, a.stop)
            out["runs"].append({"rows": rows})
        except BaseException as e:
            out["runs"].append({"error": f"{type(e).__name__}: {e}", "traceback": traceback.format_exc()})
            break
    Path(a.out).write_text(json.dumps(out, default=repr, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
