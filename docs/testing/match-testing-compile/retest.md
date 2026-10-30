# Compiled GEKKO FMUs, retest after the fixes: fmugen `dev` at `0ab151f`

The 9 GEKKO models that failed in the first compile run ([report.md](report.md)), compiled again with `fmugen build --compile pyinstaller` after commit `19ccfd9`. That commit keeps bundled executables runnable, finds `[model] cwd` data and `hf_cache`, collects editable installs and collects pyomo's plugins. Each compiled FMU was compared with the uncompiled FMU of the same config from the last comparison run (`rerun/results.json` = `docs/match-testing-rerun/results.json`).

## Result

**8 of 9 now match exactly; CGE couldn't be run: Windows Smart App Control blocks the new compiled binary.**

| Model | Config | Build | Size (MB) | Values compared | Max rel diff | Verdict |
|---|---|---|---|---|---|---|
| gk_distillation | probe | 14 s | 62.1 | 28 | 0 | match |
| gk_distcol_gym | seeded | 14 s | 63.1 | 39 | 0 | match |
| gk_bioreactor | probe | 24 s | 97.8 | 20 | 0 | match |
| gk_cftoc | hand | 14 s | 62.1 | 12 | 0 | match |
| gk_apsis_mpc | probe | 13 s | 62.1 | 336 | 0 | match |
| gk_mpc | probe | 13 s | 62.1 | 48 | 0 | match |
| gk_ethanol | probe | 13 s | 62.1 | 124 | 0 | match |
| gk_nhes | probe | 40 s | 128.5 | 4,892 (`solve_time` excluded, random) | 0 | match |
| gk_cge (Windows) | probe | 53 s | 66.8 | | | **blocked** |

- **Gekko's solver runs.** The compiled FMUs now launch with `chmod -R u+x dist/main && ./dist/main/main`, so the solver works without any change by hand. NHES also finds its `cwd` data (`grid_data/more_data_scaled.csv`).
- **CGE** builds on Windows (53 s, 66.8 MB), but its backend exits before UniFMU's handshake: `Spawning fmi2 slave failed; handshake failed; … backend exited unexpectedly with exit status 1`. Started by hand, the reason shows: `Error al ejecutar el programa 'main.exe': Una directiva de Control de aplicaciones bloqueó este archivo`.
  - Smart App Control (enforcement mode) blocks the new unsigned `main.exe`, which has no reputation. Last time it let that run's compiled binary run.
  - As planned, this was reported and not worked around, so whether the `cwd` fix works for CGE on Windows wasn't tested here. On Linux, NHES shows that a compiled `cwd` model now finds its data.
- **Not in this list, so not retested:** gk_fopdt, gk_debut_steady, gk_debut_reflux and gk_openhens. They also failed in the first run, through the same causes (execute bits; editable installs; pyomo's plugins and `cwd`). The fix commit addresses them too, but they weren't run again.

## How it was run

- **Environment:** the work folders of the earlier runs had been deleted, so the environment was rebuilt:
  - the source repositories at the same commits (`comparison/harness/gekko_setup.sh`) and the same work copies (`gekko/prepare.sh`: wrappers, renames, generated NHES data);
  - a shared GEKKO venv pinned to the versions the rerun recorded (gekko 1.3.2, numpy 2.5.4, pandas 2.3.3, matplotlib 3.11.2, gym 0.26.2, …);
  - the configs recreated with `init --probe`, the seeded gym config and CFTOC's hand config as before.
- **Reference values:** the last run's `fmu_rows`. The value counts are the same as the diagnosis copies gave in the first compile run, and every value is identical.
- **Build and comparison:** as in [report.md](report.md): one minimal venv per model (constrained to the pinned versions), PyInstaller 6.22.3, FMI 2, `fmpy.simulate_fmu(stop_time=3, output_interval=1)`, every output at every step, relative difference ≤ 10⁻⁹.
- **CGE:** on Windows, in a short folder. The first attempt, in a deeply nested folder, failed to build only because that path is too long for Windows (PyInstaller couldn't find lxml's deepest data file).


## CGE on another PC

CGE was then compiled on another Windows 11 PC, where Smart App Control didn't block it: fmugen at `0ab151f`, the repository at `60d611d` with `Code/model.py` renamed to `Code/cge_model.py`, pandas 2.3.3, gekko 1.3.2, PyInstaller 6.22.3, in `C:\fmgw` (a short path). `init --probe` from the repository root (`cwd = "."`), then `build` with and without `--compile pyinstaller`. The compiled FMU (64 MB, 6.5 min) simulates and equals the uncompiled one: 27 outputs at 4 points, all identical. So the `cwd` fix works on Windows too.
