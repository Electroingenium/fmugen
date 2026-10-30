# UniFMU: bugs and limitations

Problems that come from [UniFMU](https://github.com/INTO-CPS-Association/unifmu) **0.14.0** (the version fmugen requires), not from fmugen. Each entry says where it was found, its effect on fmugen FMUs, and the workaround, if any. Sources: fmugen's own testing ([tested-models.md](../testing/tested-models.md)) and the [UniFMU README](https://github.com/INTO-CPS-Association/unifmu/blob/master/README.md).

| # | Problem | Kind | Workaround in fmugen |
|---|---|---|---|
| 1 | [The importer hangs when the model prints more than ~4 KB](#1-the-importer-hangs-when-the-model-prints-more-than-4-kb) | Bug | `build --capture-output` |
| 2 | [The backend keeps running after an FMU error](#2-the-backend-keeps-running-after-an-fmu-error) | Limitation | Free the instance after an error ([FMPy doesn't](fmpy.md#1-simulate_fmu-doesnt-free-the-instance-after-an-fmu-error)) |
| 3 | [Error messages are lost when logging is off](#3-error-messages-are-lost-when-logging-is-off) | Limitation | None. Turn on debug logging in the importer |
| 4 | [Log messages only during an FMI call](#4-log-messages-only-during-an-fmi-call) | Limitation | None |
| 5 | [Unsupported FMI functions stop the backend](#5-unsupported-fmi-functions-stop-the-backend) | Limitation | Never declared, so compliant importers don't call them |
| 6 | [Co-Simulation only](#6-co-simulation-only) | Limitation | None |
| 7 | [Exact `protobuf==5.27.3` pin](#7-exact-protobuf5273-pin) | Limitation | None |
| 8 | [The FMU needs a Python interpreter](#8-the-fmu-needs-a-python-interpreter) | Limitation | `--vendor`, `--compile` |
| 9 | [Reserved module names](#9-reserved-module-names) | Limitation | None. Rename the model file |
| 10 | [Smaller limitations from the README](#10-smaller-limitations-from-the-readme) | Limitation | — |
| 11 | [FMI 2 log messages are used as printf format strings](#11-fmi-2-log-messages-are-used-as-printf-format-strings) | Bug | The FMI 2 adapter doubles each `%` |
| 12 | [A backend that exits during a call can abort the importer](#12-a-backend-that-exits-during-a-call-can-abort-the-importer) | Bug | None. Find the backend's error by running it directly |
| 13 | [The distributed proxy's port can't be set](#13-the-distributed-proxys-port-cant-be-set) | Limitation | None. Read the port from the importer's log |

---

## 1. The importer hangs when the model prints more than ~4 KB

When the FMU's Python process writes more than about 4 KB in total to stdout/stderr, the backend blocks and the importer waits forever, with no error.

**Cause:** UniFMU starts the backend with its stdout going to a pipe and stderr merged into it, and reads that pipe only after the backend has exited. In v0.14.0, `fmiapi/src/common/dispatcher/backend_subprocess.rs`, `BackendSubprocess::create` passes `stdout: Redirection::Pipe, stderr: Redirection::Merge`. The only read is in `monitor_subprocess`, once `poll()` returns an exit status, to put the output in the "backend exited unexpectedly" message. While the backend runs, nobody empties the pipe. When it is full, the backend's next write blocks, so it never answers the pending FMI call.

- Rechecked on 2026-10-09 with pvlib's `alt2pres` (FMI 2, built without `--capture-output`) and `setup = [{ call = "builtins:print", args = ["xxx…"] }]`: 1,000, 3,600, 3,700 and 3,800 characters printed simulate; 3,900, 4,200, 8,000 and 20,000 hang (killed after 120 s). The backend also writes its own startup line (`INFO:…main.py:dispatcher endpoint received: …`) to the same pipe, so the total at the limit is about 4,096 bytes, the default size of a Windows pipe.
- During the hang, `py-spy dump --native` of the backend shows its main thread in `ZwWriteFile` ← `WriteFile` ← `fflush`: it is blocked writing its output.
- The printed text never reaches the importer's console, even when the run succeeds: it stays in the pipe.
- Large messages through UniFMU's protocol are not affected: a 160 KB array output and a 70,000-character log message through the FMI logger both work.
- Typical triggers: library warnings, progress bars (Hugging Face model loading), printed tracebacks.
- Log records count too: the backend's `main.py` configures the root logger to write to the console, so the model's `logging` records go both to the importer and into the pipe. A function that logs one short warning per step hung FMPy after about 100 steps without `--capture-output` (2026-10-14).

**The panic is a consequence, not the cause.** This entry first said UniFMU panics here (`zeromq-0.4.1/src/rep.rs:168:40: not yet implemented`). The recheck didn't show a panic while the backend was blocked. The panic appeared, followed by the importer aborting (exit code `0xC0000409`), when the blocked backend was then killed: that is [§12](#12-a-backend-that-exits-during-a-call-can-abort-the-importer). So a hang that ends with the backend being killed, or dying, can look like a panic.

Found with Chronos-Bolt ([tested-models.md](../testing/tested-models.md)).

**Workaround:** `fmugen build --capture-output` redirects file descriptors 1 and 2 inside the backend to a temporary file, and sends the text to the importer's log as `[output]` messages after each call. It is off by default, so a model that prints or logs a lot still hangs unless built with it. Possible next steps in fmugen: have `init --probe` suggest the flag when its probe call writes a lot, or turn it on by default. Not reported upstream yet.

## 2. The backend keeps running after an FMU error

When an importer exits without calling `fmi2FreeInstance` / `fmi3FreeInstance`, UniFMU's Python backend stays alive: 2 processes (`venv\Scripts\python.exe main.py` and the base `python.exe main.py`). Seen with FMI 2 and FMI 3, on Windows with FMPy, after an FMU call failed (an error status from `exitInitializationMode` or `doStep`).

**The importer is the cause, not UniFMU:** FMPy's `simulate_fmu` skips `freeInstance` after an error: [fmpy-limitations.md §1](fmpy.md#1-simulate_fmu-doesnt-free-the-instance-after-an-fmu-error). When the instance is freed after the failed call, the backend exits within 3 s, in FMI 2 and FMI 3 (checked on 2026-10-13). This entry first said "probably `fmi2FreeInstance` isn't reached, or doesn't stop the backend": it isn't reached.

**UniFMU's part:** the backend can't tell that the importer has gone. Its loop (`abstract_backend.py`, `recv_command`) waits on a ZeroMQ `socket.recv()` with no timeout and doesn't watch its parent process, and on Windows a child process isn't stopped when its parent exits. So whenever an importer skips the free (an error path, a crash, a kill), the backend waits forever. A receive timeout or a parent check would make it exit.

Effects: the orphaned backend holds the importer's stdout/stderr, so anything reading them through a pipe waits forever, and the orphans pile up. Details in [fmpy-limitations.md §1](fmpy.md#1-simulate_fmu-doesnt-free-the-instance-after-an-fmu-error).

Found in [tested models](../testing/tested-models.md) (problem 1). Reproduce: `fmpy simulate failing.fmu 2>&1 | Out-Null` in PowerShell never returns, and the count of `main.py` processes goes up by 2.

**Workaround:** free the instance after an error (see the FMPy entry). With `fmpy simulate`, send the output to a file, not a pipe, and kill leftover `main.py` processes.

**Possible in fmugen:** fmugen's runtime runs inside the backend process, so it could stop an orphaned backend itself: a daemon thread started with the engine checks every few seconds whether the parent process is alive (`os.getppid()` on POSIX; `OpenProcess`/`WaitForSingleObject` through `ctypes` on Windows) and calls `os._exit` if it isn't. Two of fmugen's own process layers make orphans more likely on Windows, where killing a parent leaves its child running: `fmugen_launch.py` (`--vendor`) starts `main.py` with `subprocess.call`, and compiled FMUs start through `powershell -command …` (`sh -c` elsewhere). `os.execv` in the launcher on POSIX, and the watchdog or a Job object on Windows, would remove that. Starting the `.exe` directly would also save PowerShell's startup time per instance.

## 3. Error messages are lost when logging is off

fmugen sends the full Python traceback of a failure to the FMI log (`logStatusError`). UniFMU drops log messages when the importer instantiates the FMU with `loggingOn=False`, which `fmpy simulate` does, even with `--fmi-logging`. The user then sees only `fmi3ExitInitializationMode failed with status 3`, with no reason.

Found in [tested models](../testing/tested-models.md) (problem 7).

**Workaround:** turn on debug logging in the importer, e.g. `fmpy.simulate_fmu(…, debug_logging=True, logger=print)`.

**Possible in fmugen:** with an environment variable such as `FMUGEN_DEBUG=1`, the runtime could also write the log and tracebacks to a file in the temp folder, so the reason for an error is kept even when the importer drops the message (also useful for §12).

## 4. Log messages only during an FMI call

UniFMU can carry a log message to the importer only while it processes a command. Log records from the model's background threads between calls aren't forwarded. See [fmi.md](../fmi.md#logging).

## 5. Unsupported FMI functions stop the backend

If an importer calls a function the Python backend doesn't implement (anything marked x in the [support overview](../fmi.md#support-overview)), the backend stops with "unrecognized command" and the simulation fails, instead of the call returning an error status. fmugen never declares these capabilities in `modelDescription.xml`, so a compliant importer won't call them. See [fmi.md](../fmi.md#unsupported-functions).

## 6. Co-Simulation only

The Python backend implements Co-Simulation only (UniFMU README, "Supported Features"):

- **No Model Exchange** (FMI 2 or 3), so an ODE model can't be integrated by the importer's own solver. This is the capability users will ask for most.
- **No Scheduled Execution** (FMI 3).
- **No directional or adjoint derivatives**, no input/output derivatives (`fmi2SetRealInputDerivatives`, `fmi2GetRealOutputDerivatives`, `fmi3GetOutputDerivatives`).
- **No asynchronous steps or status queries**: `fmi2CancelStep`, `fmi2Get*Status`.
- **No runtime variable dependencies**: `fmi3GetNumberOfVariableDependencies`, `fmi3GetVariableDependencies`. fmugen declares them statically in `<ModelStructure>`.
- **No `fmi3EvaluateDiscreteStates`.**
- **No intermediate update or early return** (FMI 3).
- **No FMI 1.**

## 7. Exact `protobuf==5.27.3` pin

The backend's generated `*_pb2.py` files require exactly `protobuf==5.27.3` (plus `pyzmq`). Since the FMU runs with the model's own environment, this pin lands there and can conflict with packages that pin protobuf differently (TensorFlow, grpcio, …). Regenerating the schemas, or testing a range like `protobuf>=5.27,<6`, would loosen it.

## 8. The FMU needs a Python interpreter

UniFMU starts the backend with the command in `launch.toml`, so a Python FMU needs an interpreter and its packages on the target machine (`needsExecutionTool = true`). The README suggests bundling an interpreter yourself. fmugen offers `build --vendor` (wheels in the FMU, still needs Python) and `build --compile pyinstaller|nuitka` (no Python needed, but only runs on the OS it was built on). See [packaging.md](../packaging.md).

The backend also uses `match`, so it needs **Python 3.10 or newer**.

## 9. Reserved module names

The backend's own modules live in the FMU's `resources/` folder next to the model: `model`, `backend`, `main`, `abstract_backend` and `schemas` (plus fmugen's `fmugen_runtime`). A model whose entry module has one of these names can't be used; rename the file. See [models.md](../models.md).

## 10. Smaller limitations from the README

- **`launch.toml` commands run without a shell**, so wildcards, per-session environment variables and aliases don't work. To use a shell, give it as the first element of the command.
- **Distributed FMUs (`generate-distributed`) are FMI 2 only**, per the README. In 0.14.0, `unifmu generate-distributed python X fmi3` does generate an FMI 3 pair, and fmugen's `build --distributed` simulates with both versions ([packaging.md](../packaging.md#distributed-fmus---distributed)).
- **Building UniFMU itself** pins Protocol Buffers v27.3 (the source of the pin in 7), and on Windows, git's CRLF line endings break the Docker build's bash scripts.

## 11. FMI 2 log messages are used as printf format strings

In FMI 2, the importer's logger callback is variadic, like `printf`: `logger(env, instanceName, status, category, message, ...)`, where `message` is a format string. UniFMU passes the log text as the format string, with no arguments after it and without escaping `%`. Any `%` in a log message is then read as a format specifier by an importer that formats messages. `%s` makes it read a string pointer from an argument that doesn't exist, and it crashes or prints garbage, depending on what is on the stack.

Source (v0.14.0, unchanged on `master` as of 2026-10-07), `fmiapi/src/fmi2/fmi2_logger.rs`, `Fmi2Logger::log`:

```rust
unsafe { (self.callback)(
    self.environment,
    self.instance_name,
    status,
    c_category.as_ptr(),
    c_message.as_ptr()      // the text as the format; no arguments follow
); }
```

The callback type is variadic (`fmiapi/src/fmi2/fmi2_types.rs`, `Fmi2CallbackLogger`, `message: Fmi2String, ...`). Every message from the Python backend takes this path: `Fmi2Slave::handle_log_return` (`fmi2_slave.rs`) passes `Fmi2LogReturn.log_message` straight to `logger.log`. So do UniFMU's own messages built with `format!` from error text.

**FMI 3 is not affected:** its logger takes a plain message, not a format string.

**Found with:** `joblib:load` + `julien-c/skops-digits` in FMI 2 ([tested models](../testing/tested-models.md)). While the FMU initializes, `huggingface_hub` downloads the model and its HTTP library logs the download URL, which contains `%3B`, `%22s` and `%27s`. fmugen forwards the record. FMPy formats FMI 2 messages in a native proxy (`fmpy.logging.addLoggerProxy`) with the variadic arguments, so `simulate_fmu(…, debug_logging=True, logger=…)` crashed with `OSError: access violation reading 0x…21xx`, 4 times out of 4.
- With `HF_HUB_OFFLINE=1` (no download, no URL in the log), the same FMU ran 2 times out of 2 and predicted 0.
- Calling FMPy's proxy directly with a message containing `%27%27x` returns `%27x`, which shows the text is formatted.
- Plain `fmpy simulate` doesn't crash, because logging is off and nothing is formatted.

**Also found with:** psychrolib's `GetHumRatioFromRelHum` in FMI 2 ([tested models](../testing/tested-models.md)). The FMU fails at initialization for its own reason: the function is a numba ufunc, which takes no keyword arguments (`TypeError: unexpected keyword arguments to ufunc: 'Pressure', 'RelHum', 'TDryBulb'`). fmugen sends the traceback to the log, and the traceback quotes numba's source line `raise TypeError("unexpected keyword arguments to ufunc: %s" % …)`, so the message contains `%s`. Results:
- `simulate_fmu(…, debug_logging=True, logger=…)`: `OSError: exception: access violation reading 0x00000000000009AA`, before the logger gets any message.
- Logging off: a clean `fmi2ExitInitializationMode failed with status 3`, no crash.
- The same model in FMI 3, with logging on: the `TypeError` is reported cleanly.
- The message's text was read by running the backend directly (`UNIFMU_DISPATCHER_ENDPOINT` and a ZeroMQ REP socket, `logging_on = true`): it is the first log message and the only one before the crash.

So any error whose traceback quotes a line with a `%` (common in library code: `"%s" % x`, `"50%"`) turns an FMI 2 error status into an importer crash when logging is on.

**Who is affected:** any FMI 2 UniFMU FMU (not just fmugen's) whose code, or a library it uses, logs a `%`: URLs, `"50%"`, printf-style strings. This applies with any importer that formats FMI 2 messages as the standard says, not only FMPy.

**Fix upstream:** call `(self.callback)(env, name, status, category, c"%s".as_ptr(), c_message.as_ptr())`, or double each `%` to `%%` in `log()`. No upstream issue was found (a search of the UniFMU issues for "logger format", "printf", "access violation" and "percent" returned nothing).

**Workaround in fmugen (done, 2026-10-14):** the FMI 2 adapter (`model_fmi2.py`, `log()`) doubles each `%` before sending a message, so an importer that formats it prints a single `%`. Checked through UniFMU with FMPy and `debug_logging=True`: a model that logs `"done 50%s of %d"` crashed FMPy with an access violation without the escape, and with it the importer receives the text unchanged (`test_fmi2_log_messages_with_percent_signs_reach_the_importer`). Since fmugen pins UniFMU 0.14.0 exactly, this is safe, but it must be removed when UniFMU passes the text as an argument, or `%%` would show in the log. UniFMU's own messages (built in Rust from error text) are not escaped and can still contain a `%`.

## 12. A backend that exits during a call can abort the importer

When the backend process exits while UniFMU waits for its reply, UniFMU's native library may panic instead of returning an error status. The panic is `zeromq-0.4.1/src/rep.rs:168:40: not yet implemented` (also seen when a backend blocked by [§1](#1-the-importer-hangs-when-the-model-prints-more-than-4-kb) is killed), followed by `panic in a function that cannot unwind … aborting`. That line is the `Some((_peer_id, _)) => todo!()` arm of `RepSocket::recv` in zeromq 0.4.1: the socket received an error from the peer connection, and that case isn't implemented. Because the panic can't unwind through the FMI C function, **the importer's whole process is aborted**. The importer gets no error status, and the backend's own error (its Python traceback) is not shown anywhere.

The underlying failure is in the backend, not in UniFMU. UniFMU's part is turning it into a process abort instead of an error. Other backend exits are reported cleanly: in the same runs, three PyTorch FMUs whose backend exited at instantiation got `Instantiation of fmi2 slave failed; … backend exited unexpectedly with exit status 1`, and so did the round-3 FMUs whose `{ call = … }` constant failed. What decides between the clean error and the panic wasn't established.

**Found with:** 12 FMUs built with `fmugen build --compile pyinstaller` in [tested models](../testing/tested-models.md). The same configs without `--compile` simulate. Running each frozen backend (`resources/dist/main/main.exe`) directly against a ZeroMQ REP socket, and sending it UniFMU's own FMI 2 commands, showed that each backend **exits** during a call:

| FMU | Call | Backend error |
|---|---|---|
| iapws `_TSat_P`, `IAPWS97` | instantiate | `FileNotFoundError: …\_internal\iapws\VERSION` |
| ahrs Madgwick, Mahony | instantiate | `FileNotFoundError: …\_internal\ahrs\utils\WMM2025\WMM.COF` |
| openap `Thrust` | instantiate | `FileNotFoundError: …\_internal\openap\data/aircraft/_synonym.csv` |
| radioactivedecay `Inventory` | instantiate | `FileNotFoundError: …\icrp107_ame2020_nubase2020\decay_data.npz` |
| pyromat `get` | instantiate | `FileNotFoundError: …\_internal\pyromat\registry` |
| tellurium `loada` | instantiate | `FileNotFoundError: …\_internal\antimony` |
| mendeleev `element` | instantiate | SQLAlchemy can't connect: `mendeleev/elements.db` isn't in the frozen FMU |
| py_vollib `black_scholes` | instantiate | `ModuleNotFoundError: No module named 'py_vollib'` |
| cantera `Solution` | instantiate | `ImportError: dynamic module does not define module export function (PyInit_drawnetwork)` |
| pymsis `calculate` | exitInitializationMode | `MSIS parameter set …\pymsis/msis21.parm not found. Stopping.` (Fortran `STOP`, exit code 0) |

Every one of these is a file, module or extension that PyInstaller left out of the frozen FMU, which is an fmugen `--compile` problem ([tested models](../testing/tested-models.md)). fmugen now collects them ([packaging.md](../packaging.md#what-the-compiler-is-told)), and all 12 compiled FMUs simulate. The UniFMU behaviour described here is unchanged: any other backend exit still aborts the importer. Through FMPy, each of these 12 FMUs aborted the Python process with the panic above, every time. `_TSat_P` was rerun alone, with no other load: 3 aborts out of 3. Building without `--capture-output` didn't make the backend's output visible either.

**Who is affected:** any UniFMU FMU whose backend can exit during a call, e.g. a frozen backend with a missing file, an uncaught exception in the backend's own code, or native code that calls `exit`. Since the importer process dies, a co-simulation master loses every other FMU too.

**Workaround in fmugen:** none. To find the backend's error, run it outside the importer: set `UNIFMU_DISPATCHER_ENDPOINT` to a ZeroMQ REP socket you bind, start `resources/dist/main/main.exe` (or `python main.py` for an uncompiled FMU), answer the handshake and send an `Fmi2Instantiate` command built with `resources/schemas/fmi2_messages_pb2.py` from an uncompiled build of the same config. The backend prints its traceback to its stdout.

**Fix upstream:** handle the error arm in UniFMU's receive path (return an error status when the peer connection fails or the backend exits), and don't let a panic cross the `extern "C"` FMI functions. Not reported upstream yet.

## 13. The distributed proxy's port can't be set

In UniFMU 0.14 the proxy port isn't configurable. The proxy of a distributed FMU binds its socket with a hardcoded port 0, so the OS assigns a new port on every run. No environment variable or `launch.toml` field changes it (`UNIFMU_DISPATCHER_ENDPOINT` is only how a local backend is told the port).

Source (v0.14.0), `fmiapi/src/common/dispatcher/mod.rs`, `RemoteDispatcher::create`:

```rust
BackendSocket::create("tcp://0.0.0.0:0")
// ...
let port = &format!("{}", &socket.endpoint)[14..];
remote_connection_notifier(port);
```

The notifier logs `ACTION REQUIRED ~ Connect remote backend to dispatcher through port: PORT` under the `logUnifmuMessages` category. The private side must then be started with that port.

**Who is affected:** FMUs built with `fmugen build --distributed`. The private side can't be started before the simulation, a firewall can't be opened for one known port, and the importer must have logging on, including `logUnifmuMessages`, or the port is never shown.

**Workaround in fmugen:** none, so fmugen has no port flag. A possible helper: a command that reads the port from the importer's log and starts the private side with it. The private side takes the port at start-up (`python main.py PORT`); see [packaging.md](../packaging.md#distributed-fmus---distributed) for starting it from the importer's logger.
