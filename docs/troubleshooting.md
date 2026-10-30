# Troubleshooting

Error messages, what causes them, and how to fix them. Questions about editing a generated config are answered in [init.md](init.md#9-faq).

- [Messages from `fmugen`](#messages-from-fmugen)
- [The FMU fails or hangs](#the-fmu-fails-or-hangs)
- [Known problems](#known-problems)

---

## Messages from `fmugen`

Every config error names the section and key, e.g. `[inputs] x: unknown key(s) ['strat']`. The checks are listed in [config.md](config.md#validation-rules). Add `--debug` to see the full traceback of an unexpected error.

| Message | Cause | Fix |
|---|---|---|
| `cannot import …: No module named …` | The model's package isn't in the environment running fmugen. | `pip install <pkg>` in that environment. |
| `probe call failed (…)` in the config written by `init --probe` | The model raised an error at the start values (division by zero, `log(0)`, out of range…). | Rerun `init` with realistic `--start` values. |
| `… must be called first` / `has not been defined` (raised by the model) | The library needs setup. | `[model] setup` / `init --setup`. |
| `--start NAME: arrays and bytes need --fmi 3` | An array or Binary value in an FMI 2 build. | Use `--fmi 3`. |
| `--start names that are not arguments of the model` | A typo, or the argument isn't in the inspected signature. | Check the names. (For C extensions without a signature, `--start` values are passed by position instead.) |
| `the config defines no FMU variables` | Nothing could be inferred (e.g. `**kwargs` only, or an uninspectable signature). | Add `[inputs]` and `[outputs]`, or use `--start`, `--kind function`. |
| `requires FMI 3 (…)` | FMI 3-only feature in an FMI 2 build. | `fmi_version = 3` or `build --fmi 3`. |
| `cannot inspect the signature of …` (comment in the generated config) | A C extension without signature information. | Give the arguments with `--start`, in order (they are passed by position), or write variables by hand with `to = "pos:N"`. |
| `unknown key(s) [...]` | A typo in a key. | The message lists the allowed keys. |
| `duplicate variable or clock names` | The same name in two sections, or a variable named like a clock. | Rename one; bind it with `to`/`from`. |
| `fmugen: warning: it's not clear which method of … runs a step` | `init` guessed the step method. | Rerun with `--call METHOD`, or edit `[model] call`. |
| `save_state = false` or `save_state = [...]` written by `init`, with a comment | The object holds something that can't be pickled: an ONNX session, a file, a socket, a generator. | Fine for most uses; the FMU just can't roll back. With a list, check that the attributes named as not saved don't change during a simulation (a session or device handle doesn't; a generator does): if they do, set `save_state = false`. |
| `build` refuses a file named `model.py` | The name clashes with UniFMU's backend module (also `backend`, `main`, `abstract_backend`, `schemas`, `fmugen_runtime`). | Rename the file. |

---

## The FMU fails or hangs

| Symptom | Cause | Fix |
|---|---|---|
| The importer reports a failed FMI call with no reason | UniFMU only forwards the model's error when the importer's logging is on. | Turn on debug logging (FMPy: `simulate_fmu(…, debug_logging=True, logger=print)`). |
| `ModuleNotFoundError` at runtime | The environment the FMU was built in has changed or was deleted, or a package is only named inside a pickle (`joblib.load` of a scikit-learn model). | Reinstall the package there, or add it to `[model] requirements` and rebuild. |
| `ModuleNotFoundError` for the model's own package | A cloned repository imported by name (`openhens:OpenHENS`) isn't installed: `init` found it only because it ran in that folder. | `pip install -e .` the repository. |
| The importer hangs; the console shows `panicked at ... zeromq-0.4.1/src/rep.rs:168:40: not yet implemented` | The model printed more than about 4 KB (warnings, progress bars); UniFMU 0.14 crashes on that. | Rebuild with `--capture-output`. |
| The importer aborts with no Python error | The backend exited during a call. | Run the backend directly to read its error: [limitations/unifmu.md §12](limitations/unifmu.md#12-a-backend-that-exits-during-a-call-can-abort-the-importer). |
| `python` processes (`main.py`) stay alive after a failed simulation | FMPy doesn't free the instance after an error. | [limitations/fmpy.md §1](limitations/fmpy.md#1-simulate_fmu-doesnt-free-the-instance-after-an-fmu-error). |
| The simulation hangs, or the importer reports an unknown command | The importer called an FMI function the UniFMU backend lacks. | See [fmi.md](fmi.md#unsupported-functions). |
| A step fails with `… is an array of shape […], not a single value` | Code-only `init` typed an array result as one Real. | `init --probe` finds the shape; or set `dimensions`. |
| A step fails because an output became `None` or changed length | FMI variables have fixed sizes. | Remove that output, or read one element (`attr:x.-1`). |
| The FMU gives different results from the model run in its repository | The model reads files from a folder it computes, which `init` doesn't copy. | Add them to `[model] sources`, with `cwd = "."`. |
| A compiled FMU fails, the uncompiled one works | The library reads its own `.py` files by path (py-pde), which a compiled FMU doesn't have; `init` warns about it. | Use `--vendor` instead. |
| A compiled FMU fails to start on Windows with no message | Smart App Control (or another policy) blocks the new unsigned `main.exe`. | Allow it, or build on a machine without that policy. |

---

## Known problems

- **`init` binds an argument named `t` to the FMU time**, also when it is a time vector (ssqueezepy's `ssq_cwt(x, t=None)`): remove it from `[time]`.
- **`--convert` is refused for a step method's argument given by its prefixed name** (`--convert calculate_optimal_actuation_x0=…`). Give a `call:` start instead (`--start calculate_optimal_actuation_x0=call:numpy:array(…)`).
- **PyInstaller stops on xgboost** unless hypothesis is installed: it imports every xgboost submodule, and `xgboost.testing` needs hypothesis.
- **Windows long paths:** some packages (cantera, pybamm) fail to import from a venv in a deeply nested folder, and PyInstaller can miss deep data files. Keep build folders short.
- **Module-level scripts can't be models:** a script that solves at import time has nothing to call. Wrap its code in a function or class.

Problems in UniFMU and FMPy themselves are in [limitations/unifmu.md](limitations/unifmu.md) and [limitations/fmpy.md](limitations/fmpy.md).
