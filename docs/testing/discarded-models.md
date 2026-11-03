# Discarded models

These libraries were tested as models on 2026-10-23 with the models in [tested-models.md](tested-models.md), but they are not models: they are frameworks or tools for building one (solvers that take your equations, toolkits where the model is assembled in code, machine-learning frameworks that need a dataset and training). There is no function or class that takes inputs and returns the model's result, so `fmugen init` has nothing to point at. These and the empty containers below are left out of the counts in [tested-models.md](tested-models.md).

Each was tried through a short file written for the test, in [pending-100/wrappers](pending-100/wrappers/), which builds one model with the library as its documentation does. The wrappers are models written for the test, not published ones, so their results don't count. The notes say what the wrapper did and whether fmugen handled it.

| Model | `init` / `init --probe` | Hand config | Compiled (PyInstaller) | Notes |
|---|---|---|---|---|
| `bullet_drop.py:BoxDrop` (pybullet) | ✗ no model to point `init` at | — | ✗ (wrapper only) | Wrapper ([bullet_drop.py](pending-100/wrappers/bullet_drop.py)): `stepSimulation` returns nothing; the state is read with functions. A box dropped on a plane in `DIRECT` mode. **No wheel for Python 3.13**: built from source (here with zig's C++ compiler, `-fno-sanitize=undefined`). |
| `casadi_ocp.py:DoubleIntegratorOCP` (casadi) (FMI 3) | ✗ no model to point `init` at | — | ✗ (wrapper only) | Wrapper ([casadi_ocp.py](pending-100/wrappers/casadi_ocp.py)): the NLP is built from CasADi expressions. Horizon N = 10 (a parameter, not structural); IPOPT; first control and cost. |
| `diffrax_decay.py:decay` (diffrax) | ✗ no model to point `init` at | — | ✗ (wrapper only) | Wrapper ([diffrax_decay.py](pending-100/wrappers/diffrax_decay.py)): an ODE term is a Python function. dy/dt = −ky with Tsit5. |
| `fipy_heat.py:Rod` (fipy) (FMI 3) | ✗ no model to point `init` at | — | ✗ (wrapper only) | Wrapper ([fipy_heat.py](pending-100/wrappers/fipy_heat.py)): a FiPy problem is a mesh, variables and an equation built in code. 1-D heat equation, 10 cells. |
| `jit_osc.py:Oscillator` (jitcode) | ✗ no model to point `init` at | — | ✗ (wrapper only) | Wrapper ([jit_osc.py](pending-100/wrappers/jit_osc.py)): the ODE is SymPy expressions compiled to C. A damped oscillator. |
| `landlab_flow.py:Hillslope` (landlab) | ✗ no model to point `init` at | — | ✗ (wrapper only) | Wrapper ([landlab_flow.py](pending-100/wrappers/landlab_flow.py)): a grid and its fields are built in code. Flow accumulation on a 5×5 grid. |
| `mcmc_gauss.py:GaussMCMC` (emcee) | ✗ no model to point `init` at | — | ✗ (wrapper only) | Wrapper ([mcmc_gauss.py](pending-100/wrappers/mcmc_gauss.py)): the log-probability is a Python function. 8 walkers, seeded; the running posterior mean. |
| `mpc_do.py:DoubleIntegratorMPC` (do-mpc) (FMI 3) | ✗ no model to point `init` at | — | ✗ (wrapper only) | Wrapper ([mpc_do.py](pending-100/wrappers/mpc_do.py)): a do-mpc model and controller are built by method calls. Double-integrator MPC, horizon 10; `make_step` gives the control. Not on a clock: run at the communication step. Through the plain `init` config: `init --probe` writes invalid TOML (a comment quoting casadi's multi-line pickling error), since fixed. |
| `opensees_cantilever.py:cantilever` (openseespy) | ✗ no model to point `init` at | — | ✗ (wrapper only) | Wrapper ([opensees_cantilever.py](pending-100/wrappers/opensees_cantilever.py)): an OpenSees model is a script of module-level commands. A cantilever's tip deflection and base moment. |
| `ruckig_axis.py:Axis` (ruckig) (FMI 3) | ✗ no model to point `init` at | — | ✗ (wrapper only) | Wrapper ([ruckig_axis.py](pending-100/wrappers/ruckig_axis.py)): Ruckig's methods are C built-ins taking parameter objects. One axis, 10 ms control cycle run inside each step: position, velocity, acceleration. |
| `sde_ou.py:ou` (sdeint) | ✗ no model to point `init` at | — | ✗ (wrapper only) | Wrapper ([sde_ou.py](pending-100/wrappers/sde_ou.py)): the drift and diffusion are Python functions. An Ornstein–Uhlenbeck process, seeded. |
| `sobol_ishigami.py:ishigami_sobol` (SALib) | ✗ no model to point `init` at | — | ✗ (wrapper only) | Wrapper ([sobol_ishigami.py](pending-100/wrappers/sobol_ishigami.py)): Sobol analysis needs a sample and the model's outputs on it. First-order and total indices of the Ishigami function. |
| `tespy_hx.py:heater` (tespy) | ✗ no model to point `init` at | — | ✗ (wrapper only) | Wrapper ([tespy_hx.py](pending-100/wrappers/tespy_hx.py)): a TESPy network is components and connections built in code. A water heater's outlet temperature. |
| `gluon_deepar.py:DeepARForecast` (gluonts) | ✗ no model to point `init` at | — | ✗ (wrapper only) | Wrapper ([gluon_deepar.py](pending-100/wrappers/gluon_deepar.py)): the estimator trains into a predictor that takes GluonTS datasets. 1 epoch, seeded; mean forecast of 4 values. |
| `neural_ode.py:NeuralODE` (torchdiffeq) | ✗ no model to point `init` at | — | ✗ (wrapper only) | Wrapper ([neural_ode.py](pending-100/wrappers/neural_ode.py)): `odeint` needs a right-hand side `f(t, y)` written as code. A seeded 2-16-2 MLP, integrated over each step with dopri5. |
| `pf_tft.py:TFTForecast` (pytorch-forecasting) | ✗ no model to point `init` at | — | ✗ (wrapper only) | Wrapper ([pf_tft.py](pending-100/wrappers/pf_tft.py)): the Temporal Fusion Transformer is built from a `TimeSeriesDataSet` and trained with a Lightning `Trainer`. Hidden size 8, 1 epoch, seeded. |
| `sbi_gauss.py:Posterior` (sbi) | ✗ no model to point `init` at | — | ✗ (wrapper only) | Wrapper ([sbi_gauss.py](pending-100/wrappers/sbi_gauss.py)): NPE needs simulations and a training run before it gives a posterior. Trained on 500 seeded simulations; the posterior mean for an observation. |

## Impractical to run

These two ran and passed (see their notes), but `init --probe` on each used all the memory of a 23 GB WSL 2 machine on 2026-10-28, and the out-of-memory kill crashed WSL. They were moved here from [tested-models.md](tested-models.md) because checking them again isn't practical; their last results are kept below. sinergym (tested 2026-11-03) can't run unattended from pip: it needs EnergyPlus installed separately.

| Model | `init` / `init --probe` | … with other options | Hand config | Compiled (PyInstaller) | Notes |
|---|---|---|---|---|---|
| `facenet_pytorch:MTCNN` | needs options | ✓ (`--call`, `--const`) | — | ✓ (`init --probe` config, 301 MB) | Failed before the fixes: `init --probe` finds no output: boxes and probabilities are numpy arrays of `dtype=object`. Face detection on ultralytics' `zidane.jpg`: 3 boxes and probabilities (numpy arrays of `dtype=object`). |
| `ultralytics:YOLO` (FMI 3) | needs options | ✓ (`init --probe`, since 2026-10-27) (`--call`, `--const`, `--start` ×2) | ✓ ([config](pending-100/hand/yolo/fmugen.toml)) | ✓ (hand config) | Before the fixes, the compiled FMU failed to instantiate with no error message: torchvision's `_C_stable.so`, loaded by path, isn't collected. YOLO11n on ultralytics' `bus.jpg`. `predict` returns `Results` objects whose detections are tensors (`results[0].boxes.xyxy`): the hand config reads the 5 boxes, classes and confidences. **2026-10-27:** `init --probe` alone: the `Results` objects in the returned list are read, giving the boxes, classes and confidences (`y0.boxes.xyxy`, …) and the timings; FMU identical to the direct run (201 values, timings and paths left out). |
| `sinergym.envs:EplusEnv` | ✗ import fails | — | — | — | Tested 2026-11-03. Needs EnergyPlus and its Python API (`pyenergyplus`), installed outside pip: `No module named 'pyenergyplus'`. |

## Empty containers

These are containers the user fills before there is anything to simulate: a physics space with no bodies, a frame with no nodes or members, a power network with no buses. Called as they are, the step does nothing, so `init --probe` finds only settings (`damping`, `is_solved`). They ran with hand configs that built a model in `setup`, but that model was written for the test, so it doesn't count either.

| Model | `init` / `init --probe` | Hand config | Compiled (PyInstaller) | Notes |
|---|---|---|---|---|
| `pymunk:Space` | ✗ no useful outputs | ✓ ([config](match-testing-rerun/configs/pymunk.toml)) | ✓ (hand config, 48.7 MB) | `init`: outputs are only constant settings (`damping`, `collision_slop`, …). The hand config adds gravity and one body in `setup` and reads `attr:bodies.0.position.y`: `Space.bodies` is a `dict_keys` view, indexed by position. |
| `Pynite:FEModel3D` | ✗ no useful outputs | ✓ [config](pending-100/hand/pynite_lib/fmugen.toml) | ✓ 125 MB | Frame built in `setup`, `analyze` each step, results from the nodes' result dicts (`attr:nodes.N2.DY.Combo 1`). |
| `pypsa:Network` | ✗ simulate fails | ✓ [config](pending-100/hand/pypsa_lib/fmugen.toml) | ✓ 432 MB | Network built in `setup` (`add` calls), `pf` each step, load as an input through `attr:loads.p_set.l`. `init` wrote `call_on_class = true` because `pf`'s first parameter is `n`, and passed it a float; fixed since: rerun, neither config has `call_on_class` (`init` checked only, not built). Needed a wrapper before `attr:` could set a key. |
| `gillespy2:Model` | ✗ no useful outputs | — | — | Tested 2026-11-03. A model with no species or reactions (`add_species`, `add_reaction` in code): the probe finds a JSON hash. |
| `rebound:Simulation` | ✗ no useful outputs | — | — | Tested 2026-11-03. A simulation with no particles (`add(m=...)` in code): the outputs are its empty coordinates and settings. |
| `uxsim:World` | ✗ simulate fails | — | — | Tested 2026-11-03. A traffic world with no nodes, links or demand (`addNode`, `addLink`, `adddemand` in code): the step `addVehicle` needs an origin and destination. |
| `PySpice.Spice.Netlist:Circuit` | ✗ no useful outputs | — | — | Tested 2026-11-03. A circuit with no elements (`R(...)`, `V(...)` in code), simulated by ngspice, not installed with it: outputs `gnd` and an empty netlist. |
| `bluesky:init` | ✗ no useful outputs | — | — | Tested 2026-11-03. Starts the air traffic simulator with no aircraft (created by stack commands such as `CRE`): returns nothing. |

## Frameworks found while testing

Of the 100 models tested on 2026-11-03 ([pending-r4](pending-r4/)), these turned out to be tools for building a model: the class or function needs objects the user assembles in code (a kernel around a Python model function, equations, a mesh, airfoils, a rocket, an airplane, a raster). Called with what `init` can give, the step fails or outputs only settings. They were not tried through wrappers.

| Model | `init` / `init --probe` | Notes |
|---|---|---|
| `numpyro.infer:MCMC` | ✗ simulate fails | Needs a kernel (`NUTS(model)`) around a model written as a Python function, then `run` with data. |
| `brian2:NeuronGroup` | ✗ no useful outputs | Neurons from equations given as a string (`dv/dt = (1 - v) / (10*ms) : 1`); the simulation runs through a `Network` or the global `run()`. The probe finds only `active`. |
| `chemprop.models:MPNN` | ✗ simulate fails | Built from message-passing, aggregation and predictor modules, then trained. |
| `capytaine:BEMSolver` | ✗ simulate fails | Solves problems built from a floating body's mesh made in code. |
| `wisdem.ccblade.ccblade:CCBlade` | ✗ simulate fails | Needs the blade's airfoils as `CCAirfoil` objects loaded from data files; `ccblade` isn't on PyPI, it comes with WISDEM. |
| `rocketpy:Flight` | ✗ simulate fails | Needs a `Rocket` assembled from a motor, fins and parachutes, and an `Environment`. |
| `aerosandbox:AeroBuildup` | ✗ no useful outputs | Needs an `Airplane` built from wings and fuselages; with what the probe makes up, the outputs are the operating point. |
| `pysheds.grid:Grid` | ✗ simulate fails | A container of rasters (DEMs) on which the user calls hydrological operations. |

## Not installable for Python 3.13

Tested on 2026-11-03 ([pending-r4](pending-r4/)): no version installs together with fmugen on Python 3.13.

| Model | Notes |
|---|---|
| `alibi_detect.cd:KSDrift` | alibi-detect depends on TensorFlow, which needs protobuf ≥ 5.28; fmugen pins protobuf 5.27.3. |
| `basic_pitch.inference:predict` | basic-pitch pins numpy < 1.24 and a TensorFlow without Python 3.13 wheels; FMPy 0.3.32 needs numpy ≥ 2.1.3. |
| `citylearn.citylearn:CityLearnEnv` | citylearn pins pandas 1.3.5, which doesn't build for Python 3.13. |
| `cityflow:Engine` ([cityflow-project/CityFlow](https://github.com/cityflow-project/CityFlow)) | Built from source: its bundled pybind11 doesn't compile for Python ≥ 3.11 (`PyFrameObject` is opaque). |
| `reaktoro:EquilibriumSolver` | Not on PyPI: only on conda-forge. |

## Discarded before testing

Found on 2026-10-29 with the 100 models tested on 2026-11-03 ([pending-r4](pending-r4/)) and left out without running them.

| Model | Reason |
|---|---|
| `gpytorch.models:ExactGP` (gpytorch) | A model is a subclass written in code (`__init__` and `forward`), then trained. |
| `lag_llama.gluon.estimator:LagLlamaEstimator` (lag-llama) | Goes through a GluonTS estimator and predictor, like `gluon_deepar.py` above, which needed a wrapper. |
| `pyomo.environ:ConcreteModel` (pyomo) | An empty container: variables, constraints and objective are added in code. |
| `control:NonlinearIOSystem` (python-control) | The update and output functions are written in code; python-control is already tested through `control:StateSpace`. |
