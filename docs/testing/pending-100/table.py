"""The tested-models.md tables of this batch, from results.json and the notes below.

    python table.py > tables.md
"""
import json
import re
from pathlib import Path

HERE = Path(__file__).parent
ROWS = {r["model"]: r for r in json.loads((HERE / "results.json").read_text())}
import sys
sys.path.insert(0, str(HERE))
from models100 import NN, OTHER  # noqa: E402

W = "pending-100/wrappers"
H = "pending-100/hand"

# what each model is, and anything worth knowing; wrappers: why one was needed
NOTES = {
    "yolo": "YOLO11n on ultralytics' `bus.jpg`. `predict` returns `Results` objects whose detections are tensors (`results[0].boxes.xyxy`): the hand config reads the 5 boxes, classes and confidences.",
    "hf_pipeline": "`\"sentiment-analysis\"` (DistilBERT SST-2), `--call __call__`: label and score. Plain `init` stops: the object the factory returns is only known by running it.",
    "sentence_tf": "`all-MiniLM-L6-v2`, `--call encode`: the 384-value embedding.",
    "timm": "`resnet18`, pretrained, on a 224×224 image tensor: 1000 logits.",
    "tv_resnet18": "ImageNet weights on a 224×224 image tensor: 1000 logits.",
    "nhits": "Wrapper ([nf_nhits.py](%s/nf_nhits.py)): neuralforecast fits on a DataFrame and a `NeuralForecast` object, which `init` can't build. N-HiTS trained 5 steps, seeded; forecasts 4 values." % W,
    "nbeats": "`--setup \"fit(darts.TimeSeries.from_values(numpy.array([...])))\"`, `--call predict`: 2 epochs, seeded. **Needs `darts[torch]`**: plain `darts` gives placeholders for its torch models.",
    "tft": "Wrapper ([pf_tft.py](%s/pf_tft.py)): the Temporal Fusion Transformer is built from a `TimeSeriesDataSet` and trained with a Lightning `Trainer`. Hidden size 8, 1 epoch, seeded." % W,
    "deepar": "Wrapper ([gluon_deepar.py](%s/gluon_deepar.py)): the estimator trains into a predictor that takes GluonTS datasets. 1 epoch, seeded; mean forecast of 4 values." % W,
    "timesfm": "TimesFM 2.5 (200M), `--create from_pretrained --call forecast` with `--setup \"compile(timesfm.ForecastConfig(max_context=64, max_horizon=8))\"` (a setup call with a call as argument). Point and quantile forecasts of 4 values.",
    "moirai": "Wrapper ([moirai_fc.py](%s/moirai_fc.py)): Moirai's `forward` takes GluonTS-prepared tensors; its documented use is `create_predictor`. Moirai 1.0 R small, zero-shot, median of 20 seeded samples. **uni2ts pins numpy 1.26**, which has no Python 3.13 wheels: installed with `--no-deps`." % W,
    "moment": "MOMENT-1-small, `--create from_pretrained --call forward`, `model_kwargs={'task_name': 'reconstruction'}`, `--setup init`: the reconstruction of a 512-point input. **momentfm pins numpy 1.25**: installed with `--no-deps`.",
    "tirex": "`load_model(\"NX-AI/TiRex\")`, `--call forecast` on a 24-point series: quantiles and mean of 4 values.",
    "mace": "MACE-MP-0 small, `--call get_potential_energy` on ASE's fcc copper.",
    "chgnet": "`--create load --call predict_structure` on fcc aluminium (pymatgen): energy, forces, stress, magnetic moments.",
    "matgl": "TensorNet-PES-MatPES-PBE-2025.2 through its ASE calculator (`matgl.ext.ase:PESCalculator`, `--call get_potential_energy`) on fcc aluminium: −3.7124 eV. `load_model` returns a `Potential` with no `predict_structure`.",
    "deepxde_fnn": "A 1-16-16-1 tanh network (PyTorch backend, `DDE_BACKEND=pytorch`). Random weights: compared seeded (`torch:manual_seed(0)` in `setup`).",
    "fno": "Fourier neural operator, 8 modes, on a 16-point input. Random weights: compared seeded.",
    "torchdiffeq": "Wrapper ([neural_ode.py](%s/neural_ode.py)): `odeint` needs a right-hand side `f(t, y)` written as code. A seeded 2-16-2 MLP, integrated over each step with dopri5." % W,
    "torchdyn": "`NeuralODE` around a `torch.nn.Linear(2, 2)` vector field. Random weights: compared seeded.",
    "sbi_npe": "Wrapper ([sbi_gauss.py](%s/sbi_gauss.py)): NPE needs simulations and a training run before it gives a posterior. Trained on 500 seeded simulations; the posterior mean for an observation." % W,
    "onnx_mnist": "The ONNX model zoo's MNIST network, `--call run` with `input_feed=call:builtins:dict(Input3=numpy.zeros(...))`: 10 logits.",
    "keras_mobilenet": "MobileNetV2 (ImageNet), `--call predict`, **Keras on its JAX backend** (`KERAS_BACKEND=jax`): TensorFlow needs protobuf ≥ 5.28 and fmugen pins 5.27.3 (UniFMU 0.14's).",
    "equinox_mlp": "A 2-8-8-1 MLP, seeded by its `PRNGKey(0)`.",
    "skorch": "A `torch.nn.Linear(4, 1)` behind `NeuralNetRegressor`, `fit` in `setup`, `--call predict`. **Not comparable**: the network's weights are drawn when the `module` constant is built, before `setup`'s seed, so two runs differ. Builds and simulates, compiled too.",
    "tabnet": "`fit` in `setup` (3 epochs, seeded), `--call predict`. **Needs pandas installed by hand.**",
    "mtcnn": "Face detection on ultralytics' `zidane.jpg`: 3 boxes and probabilities (numpy arrays of `dtype=object`).",
    "whisper": "`tiny`, `--call transcribe` on 5 s of whisper's `jfk.flac` (16 kHz numpy array): the text. A file path needs ffmpeg.",
    "speechbrain": "ECAPA speaker embedding, `--create from_hparams --call encode_batch`: 192 values.",
    "wav2vec2": "Wrapper ([w2v_asr.py](%s/w2v_asr.py)): the pretrained model is a bundle constant's `get_model()`, and CTC decoding is code. The transcript of the speech clip." % W,
    "sb3_sac": "SAC on Pendulum from the Hugging Face hub (`sb3/sac-Pendulum-v1`), `--create load --call predict`. **Needs gym and shimmy**: the saved model unpickles with them.",
    "recurrent_ppo": "LSTM PPO on CartPole (`sb3/ppo_lstm-CartPoleNoVel-v1`), `--create load --call predict`: action and LSTM states, carried between steps. **Needs gym and shimmy.**",
    "d3rlpy_cql": "`CQLConfig(...).create()`: the CQL algorithm object's settings.",
    "snntorch": "A leaky integrate-and-fire neuron with its hidden state: spikes and membrane potential between steps.",
    "norse_lif": "LIF cell, `--call forward` with `input_tensor`: spikes, voltage and current.",
    "nengo": "`nengo.Simulator` of an empty network, `--call run_steps`: steps and the simulator's clock, an output `model_time` (fmugen now renames a `time` variable in FMI 2 too: FMPy's results call the simulation time `time`).",
    "reservoirpy": "A 10-unit reservoir, `--call step` on a 1-D input.",
    "tcn": "Causal TCN with channels `[4, 4]` on a 16-point input. Random weights: compared seeded.",
    "kan": "A 2-3-1 KAN (grid 3, k 3, seed 0). The `init --probe` config fails (`missing input data x`: the probe's call sets `x` the step reads); plain `init` writes a working one. **Needs scikit-learn, pandas, matplotlib, pyyaml and tqdm installed by hand.**",
    "cfc": "Closed-form continuous-time network, 8 units, on a 4-step sequence. Random weights: compared seeded.",
    "neuromancer_mlp": "A 2-8-1 MLP block. Random weights: compared seeded. **Needs wandb installed by hand.**",
    "pyod_ae": "Autoencoder detector, `fit` in `setup` on 40 rows, `--call decision_function`. With 8 rows its training loop fails (`loss` unbound).",
    "rembg": "U²-Net background removal: PNG bytes in, PNG bytes out (FMI 3 Binary). Compared as SHA-256 hashes: identical.",
    "easyocr": "OCR of a rendered `FMUGEN 2026` image (bytes): the text.",
    "qrcode": "`make(\"fmugen\")`: the QR image's properties (size, box size, …).",
    "pyzbar": "Decodes the QR image: `fmugen`. **Needs the zbar library** (`libzbar0t64`, here extracted without root and found through `LD_LIBRARY_PATH`).",
    "pil_grayscale": "A 64×64 image to grayscale: the image's properties.",
    "mfcc": "MFCC matrix of 5 s of speech: 499 × 13 (FMI 3 2-D array).",
    "ssq_cwt": "Synchrosqueezed CWT of a 64-sample sine. `init` binds the argument `t` (the signal's time vector, default None) to the FMU time, so the model gets a float: the hand config leaves `[time]` out and outputs the 162 frequencies and scales.",
    "harold_lqr": "LQR of a 2-state system: gain K (1×2), Riccati solution X (2×2), closed-loop eigenvalues (complex, split). **Needs numpy < 2.3** (harold imports `numpy.linalg.linalg`, removed in 2.3; FMPy needs ≥ 2.1.3).",
    "slycot": "`sb02md`, the continuous Riccati equation (SLICOT, Fortran): X, eigenvalues, condition.",
    "do_mpc": "Wrapper ([mpc_do.py](%s/mpc_do.py)): a do-mpc model and controller are built by method calls. Double-integrator MPC, horizon 10; `make_step` gives the control. Not on a clock: run at the communication step." % W,
    "casadi_nlpsol": "Wrapper ([casadi_ocp.py](%s/casadi_ocp.py)): the NLP is built from CasADi expressions. Horizon N = 10 (a parameter, not structural); IPOPT; first control and cost." % W,
    "osqp": "`--setup \"setup(scipy.sparse.csc_matrix(...), ...)\"`, `--call solve`: the QP's solution and info. Not on a clock.",
    "ruckig": "Wrapper ([ruckig_axis.py](%s/ruckig_axis.py)): Ruckig's methods are C built-ins taking parameter objects. One axis, 10 ms control cycle run inside each step: position, velocity, acceleration." % W,
    "vqf": "Wrapper ([vqf_imu.py](%s/vqf_imu.py)): `VQF` is a C built-in, and the orientation comes from a method (`getQuat6D()`). 100 Hz samples inside each step." % W,
    "simdkalman": "Kalman smoother of a 24-point series. `smooth` returns a `Result` whose estimates are two levels down (`result.observations.mean`): the hand config reads them (24 levels, 24×2 states).",
    "imufusion": "Wrapper ([fusion_ahrs.py](%s/fusion_ahrs.py)): `Ahrs` is a C built-in, and the orientation comes from a method (`get_quaternion()`). 100 Hz samples inside each step." % W,
    "py_pde": "`DiffusionPDE`, `--call evolution_rate` on a 16-cell scalar field. **Compiled FMU fails**: py-pde loads `pde/backends/*/config.py` by path, and a compiled FMU ships no `.py` files.",
    "fipy": "Wrapper ([fipy_heat.py](%s/fipy_heat.py)): a FiPy problem is a mesh, variables and an equation built in code. 1-D heat equation, 10 cells." % W,
    "skfem": "`MeshTri().refined(2)`: the refined mesh's nodes and elements (2-D arrays).",
    "pydmd": "DMD with 2 modes of 8 snapshots. The `init --probe` config reads DMD's properties at initialization, before `fit()` has run; the hand config runs `fit()` then too (`init_call = true`): eigenvalues and amplitudes, complex, split.",
    "numpy_rng": "`default_rng(seed=12345678901234567890)`, `--call random`: the seed is a UInt64 parameter (FMI 3).",
    "pysindy": "SINDy fitted to two decaying exponentials: the coefficient matrix (2×10).",
    "mujoco": "Wrapper ([mujoco_pendulum.py](%s/mujoco_pendulum.py)): MuJoCo steps a model and its data with `mj_step(model, data)`, C built-ins. A motor-driven pendulum: angle and velocity." % W,
    "wntr": "EPANET simulation of `Net1.inp`: pressures at the network's junctions (2-D, time × node).",
    "pymoo": "NSGA-II on ZDT1, 5 generations, seed 1: the Pareto set and front.",
    "diffrax": "Wrapper ([diffrax_decay.py](%s/diffrax_decay.py)): an ODE term is a Python function. dy/dt = −ky with Tsit5." % W,
    "sdeint": "Wrapper ([sde_ou.py](%s/sde_ou.py)): the drift and diffusion are Python functions. An Ornstein–Uhlenbeck process, seeded." % W,
    "jitcode": "Wrapper ([jit_osc.py](%s/jit_osc.py)): the ODE is SymPy expressions compiled to C. A damped oscillator." % W,
    "pinocchio": "Inverse dynamics (`rnea`) of the sample manipulator: Boost.Python functions with no signature, so `init` stops. The hand config passes model and data as positional constants and q, v, a by position.",
    "pybullet": "Wrapper ([bullet_drop.py](%s/bullet_drop.py)): `stepSimulation` returns nothing; the state is read with functions. A box dropped on a plane in `DIRECT` mode. **No wheel for Python 3.13**: built from source (here with zig's C++ compiler, `-fno-sanitize=undefined`)." % W,
    "skrf": "A 90° line (`DefinedGammaZ0.line`) at 3 frequencies: S-parameters, complex, split.",
    "prophet": "`fit` on 24 days in `setup` (a DataFrame built by a setup call), `--call predict` for 2 days.",
    "statsforecast": "AutoARIMA, `--call forecast`, h = 4.",
    "pmdarima": "`auto_arima`, `--call predict`.",
    "lifelines": "Kaplan–Meier, `--call fit`: the survival function.",
    "emcee": "Wrapper ([mcmc_gauss.py](%s/mcmc_gauss.py)): the log-probability is a Python function. 8 walkers, seeded; the running posterior mean." % W,
    "chaospy": "`Normal(0, 1)`, `--call pdf`.",
    "salib_sobol": "Wrapper ([sobol_ishigami.py](%s/sobol_ishigami.py)): Sobol analysis needs a sample and the model's outputs on it. First-order and total indices of the Ishigami function." % W,
    "pysam_pvwatts": "Wrapper ([pvwatts.py](%s/pvwatts.py)): `Pvwattsv8.default(\"PVWattsNone\")` is a C built-in, and the model needs weather data set as an attribute. A 4 kW system, a synthetic year." % W,
    "floris": "Wrapper ([floris_farm.py](%s/floris_farm.py)): the farm's conditions are set with `set()` before `run()`. 3 turbines: powers and farm power." % W,
    "pypsa": "Wrapper ([pypsa_pf.py](%s/pypsa_pf.py)): a network is built with `add()` calls. AC power flow of 2 buses." % W,
    "pandapipes": "Wrapper ([pipes_flow.py](%s/pipes_flow.py)): `pipeflow(net)` returns nothing and writes its results onto `net`, as pandapower's `runpp`. Gas flow in the meshed delta network." % W,
    "pyswmm": "`Simulation(inputfile)`, `--call __next__`: steps pyswmm's `model_weir_setting.inp`. The `init --probe` config fails (it reads the current time before the simulation starts); plain `init` works.",
    "pastas": "`Model(oseries)`, `--call simulate`: a Series indexed by dates, read as an array of 24 values.",
    "landlab": "Wrapper ([landlab_flow.py](%s/landlab_flow.py)): a grid and its fields are built in code. Flow accumulation on a 5×5 grid." % W,
    "tespy": "Wrapper ([tespy_hx.py](%s/tespy_hx.py)): a TESPy network is components and connections built in code. A water heater's outlet temperature." % W,
    "xgboost": "`fit` in `setup`, `--call predict`. **Compiled: needs hypothesis installed**: PyInstaller imports every xgboost submodule, and `xgboost.testing` calls `pytest.importorskip(\"hypothesis\")`, whose exception stops PyInstaller.",
    "lightgbm": "`fit` in `setup`, `--call predict`. **Needs scikit-learn installed by hand.**",
    "catboost": "`fit` in `setup`, `--call predict`.",
    "gpy": "GP regression, `--call predict`: mean and variance. **Needs matplotlib installed by hand.**",
    "hapsira": "`Orbit.from_classical` (astropy units), `--call propagate` 600 s. **Needs numpy < 2.4** (`numpy.in1d`) **and astropy < 7** (`matrix_product`).",
    "openseespy": "Wrapper ([opensees_cantilever.py](%s/opensees_cantilever.py)): an OpenSees model is a script of module-level commands. A cantilever's tip deflection and base moment." % W,
    "pynite": "Wrapper ([pynite_beam.py](%s/pynite_beam.py)): the model is built with method calls. A simply supported beam." % W,
    "sectionproperties": "Wrapper ([section_rect.py](%s/section_rect.py)): the geometry must be meshed (`create_mesh`) before the `Section` is made. A 100×50 rectangle." % W,
    "arch": "`arch_model(y)`, `--call fit`: the GARCH fit's parameters.",
    "ephem": "`Sun`, `--call compute`: the position is in C attributes that exist only after `compute()`; the hand config reads them, with `init_call = true`.",
    "sympy_lambdify": "`lambdify` makes a function of x and y by position, which `init` can't see; the hand config binds them with `to = \"pos:N\"`.",
}

# Failures on unmodified fmugen: the problem, and the fmugen fix with which the model
# passed. (init cell, compiled cell, note put before the model's note)
P = "the fixes"
SETUP = ("✗ build fails", "—", "**Fails: `build` refuses the config `init --probe` wrote**: its `setup` has a call as an "
         "argument (`fit(numpy.array(...))`), which the probe ran (`[model] setup …: arguments must be literals or "
         f"names`). Passes with {P}.")
INTS = ("✗ simulate fails", "—", "**Fails: the integer list start is typed Real**, so the model gets float sizes "
        f"(`empty(): argument 'size' must be tuple of ints`). Passes with {P} (integer lists typed Integer).")
SILENT = "**Compiled FMU fails to instantiate, with no error message** (`--capture-output` hides it: \"backend exited " \
         "unexpectedly … no output\"). "
FAILS = {
    **{n: SETUP for n in ("nbeats", "timesfm", "skorch", "tabnet", "pyod_ae", "osqp", "prophet", "xgboost", "lightgbm", "catboost")},
    **{n: INTS for n in ("deepxde_fnn", "fno", "tcn", "neuromancer_mlp")},
    "kan": ("✗ simulate fails", "—", "**Fails: `width=[2, 3, 1]` is typed Real** (`'float' object is not subscriptable`). "
            f"Passes with {P}."),
    "rembg": ("✗ no useful outputs", "✓ simulates (no outputs)", "**Fails: `init --probe` finds no output**: the result is "
              f"`bytes`, which could be an FMI 3 Binary output. Passes with {P}."),
    "pastas": ("✗ no useful outputs", "✓ simulates (no outputs)", "**Fails: `init --probe` finds no output**: the result is "
               f"a pandas Series indexed by dates. Passes with {P} (read as an array)."),
    "mtcnn": ("✗ no useful outputs", "✗ instantiate fails", "**Fails: `init --probe` finds no output**: boxes and "
              f"probabilities are numpy arrays of `dtype=object`. Passes with {P}. " + SILENT),
    "nengo": ("✗ wrong values", "✗ wrong values, as uncompiled", "**Fails: the output `time` (the simulator's clock) "
              "shows the simulation time** in FMPy's results, which name the simulation time `time`: 1.0 instead of 0.01. "
              f"Passes with {P} (renamed `model_time`)."),
    **{n: (None, "✗ instantiate fails", SILENT + f"With {P}: torchvision's `_C_stable.so`, loaded by path, isn't collected.")
       for n in ("timm", "yolo", "easyocr")},
    **{n: (None, "✗ instantiate fails", SILENT + f"With {P}: chgnet opens `chgnet/model/../pretrained/…`, and `model/` "
           "doesn't exist in the frozen app (code-only folder).") for n in ("chgnet",)},
    "speechbrain": (None, "✗ instantiate fails", SILENT + f"With {P}: speechbrain lists its `utils/` folder, which doesn't "
                    "exist in the frozen app."),
    **{n: (None, "✗ instantiate fails", SILENT + f"With {P}: TorchScript needs the `.py` sources of the functions it "
           "compiles at import.") for n in ("mace", "norse_lif", "matgl")},
    "nhits": (None, "✗ instantiate fails", SILENT + f"With {P}: ray imports `ray._private` modules by name."),
    "wav2vec2": ("✗ invalid config", "—", "**Fails: `init --probe` writes invalid TOML**: a comment quotes a multi-line "
                 "error (why the object can't be pickled). Plain `init` types the transcript as Real. "
                 f"Passes with {P}."),
    "hf_pipeline": (None, "✗ instantiate fails", SILENT + f"Passes with {P}."),
    "torchdyn": (None, "✗ instantiate fails", SILENT + f"Passes with {P}."),
}
# notes that described behavior of the fixed fmugen
NOTES["nengo"] = "`nengo.Simulator` of an empty network, `--call run_steps`."
NOTES["rembg"] = "U²-Net background removal: PNG bytes in, PNG bytes out."
NOTES["kan"] = ("A 2-3-1 KAN (grid 3, k 3, seed 0). **Needs scikit-learn, pandas, matplotlib, pyyaml and tqdm installed "
                "by hand.**")
NOTES["pastas"] = "`Model(oseries)`, `--call simulate`."
NOTES["do_mpc"] += (" Through the plain `init` config: `init --probe` writes invalid TOML (a comment quoting casadi's "
                    f"multi-line pickling error), fixed in {P}.")
NOTES["skorch"] ="A `torch.nn.Linear(4, 1)` behind `NeuralNetRegressor`, `fit` in `setup`, `--call predict`."


WRAPPED = {m[0] for m in NN + OTHER if m[3].get("wrap")}
WRAP_NOTE = ("**Fails: the library has no function or class that is the model**, so `init` has nothing to "
             "point at. Tried through a wrapper written for this test: ")


def cell_init(r):
    if r["model"] in WRAPPED:
        return "✗ no model to point `init` at"
    cfgs = {c["config"]: c["verdict"] for c in r["configs"]}
    if cfgs.get("probe") == "match" or cfgs.get("plain") == "match":
        return "✓"
    if cfgs.get("seeded") == "match":
        return "✓ (seeded)"
    if FAILS.get(r["model"], (None,))[0]:
        return FAILS[r["model"]][0]
    if "hand" in cfgs:
        if cfgs.get("probe") in (None, "no config"):
            return "✗ no config"
        return "✗ no useful outputs" if cfgs.get("probe") == "not comparable" else "✗ simulate fails"
    return "✗"


def cell_hand(r):
    cfgs = {c["config"]: c["verdict"] for c in r["configs"]}
    if "hand" not in cfgs:
        return "—"
    return f"✓ ([config]({H}/{r['model']}/fmugen.toml))" if cfgs["hand"] == "match" else "✗"


def cell_compiled(r):
    cp = r["compiled"]
    size = cp.get("size_mb")
    size = f"{size / 1000:.1f} GB" if size and size >= 1000 else f"{size} MB"
    cfg = {"probe": "`init --probe`", "plain": "`init`", "seeded": "seeded", "hand": "hand"}.get(cp.get("config"), cp.get("config"))
    if r["model"] in WRAPPED:
        return "✗ (wrapper only)"
    if r["model"] in FAILS:
        return FAILS[r["model"]][1]
    if cp.get("verdict") == "no config to compile":
        return "—"
    if cp.get("verdict") == "match":
        return f"✓ ({cfg} config, {size})"
    if cp.get("verdict") == "not comparable":
        return f"✓ simulates ({cfg} config, {size})"
    return f"✗ {cp.get('verdict')}"


def entry(m):
    name, pkgs, args, extra = m
    if extra.get("wrap"):
        cls = args[0].split(":")[1]
        return f"`{extra['wrap']}:{cls}` ({pkgs[0].split('[')[0]})"
    return f"`{args[0]}`"


def table(models):
    out = ["| Model | `init` / `init --probe` | Hand config | Compiled (PyInstaller) | Notes |", "|---|---|---|---|---|"]
    for m in models:
        r = ROWS[m[0]]
        fmi = " (FMI 3)" if m[3].get("fmi") == 3 else ""
        note = " ".join(x for x in (FAILS.get(m[0], (None, None, ""))[2], NOTES.get(m[0], "")) if x)
        if m[0] in WRAPPED:
            note = WRAP_NOTE + note
        out.append(f"| {entry(m)}{fmi} | {cell_init(r)} | {cell_hand(r)} | {cell_compiled(r)} | {note} |")
    return "\n".join(out)


if __name__ == "__main__":
    missing = [m[0] for m in NN + OTHER if m[0] not in NOTES]
    assert not missing, missing
    print("### Neural networks\n")
    print(table(NN))
    print("\n### Other models\n")
    print(table(OTHER))
    # summary counts
    allm = NN + OTHER
    rs = [ROWS[m[0]] for m in allm]
    n_init = sum(cell_init(r).startswith("✓") and "random" not in cell_init(r) for r in rs)
    n_hand = sum(cell_hand(r).startswith("✓") and not cell_init(r).startswith("✓") for r in rs)
    n_comp = sum(r["compiled"].get("verdict") == "match" and r["model"] not in FAILS and r["model"] not in WRAPPED
                 for r in rs)
    n_unc = sum(cell_init(r).startswith("✓") or cell_hand(r).startswith("✓") for r in rs)
    print(f"<!-- simulate={n_unc} -->")
    print(f"<!-- fails={len(FAILS)} -->")
    n_wrap = sum(bool(m[3].get("wrap")) for m in allm)
    mx = max((c.get("max_rel") or 0) for r in rs for c in r["configs"] if c["verdict"] == "match")
    vals = sum((c.get("compared") or 0) for r in rs for c in r["configs"] if c["verdict"] == "match")
    print(f"\n<!-- models={len(allm)} init={n_init} hand_only={n_hand} compiled_match={n_comp} wrappers={n_wrap} "
          f"max_rel={mx} values={vals} -->")
