# Compiled FMUs (PyInstaller), the rest of the models: fmugen `dev` at `6dfc18c`

The second batch after [../report.md](../match-testing-compile/report.md): every other model of the last comparison run (`rerun/results.json`, i.e. `docs/match-testing-rerun/`) that has a working config, compiled with `fmugen build --compile pyinstaller`. Each compiled FMU was simulated and compared, output by output and step by step, with the last run's uncompiled FMU of the same config. Every model is in [results.csv](results.csv), with the same columns as the first batch's; every value is in [results.json](results.json).

## Summary

**All 100 compiled FMUs build, simulate and match their uncompiled FMU exactly**: 229,810 values compared, largest relative difference 0. No build failed or timed out (the slowest took 227 s).

| Verdict | Models |
|---|---|
| **match** | **100** |
| mismatch | 0 |
| build fails | 0 |
| timeout | 0 |
| FMU fails | 0 |

**Which models:** every model of the last run that isn't in the first batch and isn't GEKKO (those were all in the first batch). That's 100: the repository examples (including both FMI 3 repository configs), the Python libraries and the 6 neural networks.
- **Left out:** pyet, sgp4 and nashpy, which have no working config (their uncompiled FMUs fail too).
- **Config:** the one that matched in the last run, by preference seeded (models random by design), then `--probe`, then repository, then plain. That gives 92 `--probe` configs, 5 seeded (CartPole, Pendulum, HighwayEnv, mesa, surfaces), TCLab's plain config (its `--probe` config is random) and the two FMI 3 repository configs.

Together with the first batch (20 of 33 builds match, all 13 failures in GEKKO models), **120 of the 133 compiled FMUs tested give exactly the uncompiled FMU's values**.

## What needed changing

- **skops:** `joblib:load` unpickles a scikit-learn model, so the classes are only named in the pickle. As built from its `--probe` config, the FMU failed at initialization: `ModuleNotFoundError: No module named 'sklearn'`. With `scikit-learn==1.9.1` added to `[model] requirements` by hand (`~/cmp/w2/m/skops/probe.sklearn.toml`), as docs/testing/tested-models.md says, it matches (112.8 MB). The verdict is for that config.
- **Neural networks:** the first attempt, with 6 builds in parallel, ran out of space in WSL's `/tmp` (a 12 GB tmpfs; `OSError: [Errno 28] No space left on device`), since fmugen builds in a temporary folder. Rerun two at a time with `TMPDIR` on the main disk, all 5 built and match. That was a setup problem, not fmugen's.
- **FMI 3 repository configs (kalman, sampled_pid):**
  - Built with `--compile pyinstaller` by `comparison/harness/fmi3_examples.py` (new: `FMI3_BUILD_ARGS`), and driven exactly like the uncompiled FMUs: `run.py`'s seed, clock ticks and plant.
  - Their compiled values equal the last run's uncompiled FMU values at every communication point.
  - Their "build time" includes that loop.

## Sizes and build times

| | Size | Build time |
|---|---|---|
| Smallest | 44.8 MB (haversine, meteocalc, TCLab) | 9 s |
| Median | 104.5 MB | 23 s |
| Largest without neural networks | 613.9 MB (surfaces), 265.0 MB (pybamm) | 58 s (surfaces), 55 s (pybamm) |
| Neural networks | **4.27–4.63 GB**: granite_ttm 4269.9, torchani 4405.6, sb3_ppo 4490.1, silero_vad 4515.0, chronos 4633.4 MB | 196–227 s |

The neural-network FMUs are so large because of CUDA:
- On Linux, `pip install torch` from PyPI brings 15 NVIDIA CUDA packages (`nvidia-cublas`, `nvidia-cudnn-cu13`, …: 3.2 GB installed) and triton (0.9 GB). fmugen collects the binaries of every dependency, so they're all in the FMU.
- In chronos's FMU: `nvidia/` 2.19 GB compressed, torch 0.55 GB, triton 0.25 GB, the weights 33 MB.
- The docs' Windows builds were around 300 MB (chronos 322.8 MB), since the Windows torch wheel has no CUDA.
- The venvs here are the ones the reference values came from, so they were kept. A CPU-only torch (`--index-url https://download.pytorch.org/whl/cpu`) would make these FMUs several GB smaller; not tested.
- Each FMU also bundles its Hugging Face weights (chronos, granite, sb3), so it runs offline.

The other models' sizes and times are in results.csv.

## How it was run

As the first batch ([../report.md](../match-testing-compile/report.md), "How it was run"), with `comparison/harness/compile_rest.py`, which reuses `compile_all.py`:
- **Where:** WSL 2, `~/cmp/repo` at `6dfc18c`, PyInstaller 6.22.3, FMPy 0.3.32, UniFMU 0.14.0.
- **Venvs:** one minimal venv per model (`~/cmp/vc/<model>`) with the packages of the venv the last run used (`models.GROUPS`), constrained to that venv's versions.
- **Build:** `fmugen build <toml> --capture-output --fmi 2 --compile pyinstaller` with a 60-minute timeout. 6 in parallel, then 2 for the networks, with `TMPDIR` on disk.
- **Simulation and comparison:** `fmpy.simulate_fmu(stop_time=3, output_interval=1)` (rc_building at its fixed 3600 s step); every output at every step against the last run's uncompiled FMU; relative difference ≤ 10⁻⁹, strings exactly. Outputs the last run treated as random (TCLab's wall clock) are left out.
- **What's where:** the FMUs and logs are in WSL at `~/cmp/c/m/<model>/` (110 GB with the first batch, mostly the neural networks). No published model was changed.
