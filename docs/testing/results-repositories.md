# Tested models: GitHub and unpublished

Columns and abbreviations are explained in [methodology.md](methodology.md#columns); "probe config" is the config `init --probe` wrote. Back to [tested-models.md](tested-models.md).

## GitHub repositories

Models published as source on GitHub, not as a package.

| Model | `init` / `init --probe` | … with other options | Hand config | Compiled (PyInstaller) | Notes |
|---|---|---|---|---|---|
| `building_physics.py:Zone` ([architecture-building-systems/RC_BuildingSimulator](https://github.com/architecture-building-systems/RC_BuildingSimulator) `97c1cdd`) | needs options | ✓ (`--call`) | ✓† | ✓ (probe config, 108.8 MB) | ISO 13790 5R1C thermal model of a building zone, `--call solve_energy`. The hand config feeds `t_m_next` back as `t_m_prev` (`[states]`) and runs two winter days of hourly weather. †It fixes `step_size = 3600`, so the generic 1 s run is refused (correct); simulated at 3600 s steps. |
| `gym_pybullet_drones.envs:HoverAviary` ([utiasDSL/gym-pybullet-drones](https://github.com/utiasDSL/gym-pybullet-drones)) | needs options | ✓ (`--setup`, `--call`, `--convert`, `--start`) | — | ✓ (probe config, 4605.2 MB) | `HoverAviary` reset (seed 0), `step` with action `[[0, 0, 0, 0]]` (`--convert action=numpy:array`): observation, reward, flags. pybullet built from source (no Python 3.13 wheel). |
| `metaworld:make_mt_envs` ([Farama-Foundation/Metaworld](https://github.com/Farama-Foundation/Metaworld)) | needs options | ✓ (`--setup`, `--call`, `--start` ×3) | — | ✓ (probe config, 159.9 MB) | Entry: MT1 is a benchmark: a set of tasks. `make_mt_envs('reach-v3', seed=0)`, reset, `step` with a zero action: observation, reward, success flags. |
| `rrmpg.models:GR4J` ([kratzert/RRMPG](https://github.com/kratzert/RRMPG)) | needs options | ✓ (`--const` ×3, `--call`) | — | ✓ (probe config, 254.2 MB) | GR4J with Perrin et al.'s median parameters (without parameters it draws random ones), `simulate` on 30 days of the CAMELS basin 01031500 shipped with rrmpg: 30 discharges. |
| `spikingjelly.activation_based.neuron:LIFNode` ([fangwei123456/spikingjelly](https://github.com/fangwei123456/spikingjelly)) | needs options | ✓ (`--call`, `--start` ×2) | — | ✓ (probe config, 406.5 MB) | `LIFNode(tau=2.0)`, `forward([1.5])` (passed as a tensor by the probe): spike and membrane potential (0.75). |

## Unpublished models

Models with no public source: they aren't counted as published models.

| Model | `init` / `init --probe` | … with other options | Hand config | Compiled (PyInstaller) | Notes |
|---|---|---|---|---|---|
| `psychrometry.py:compute_balances_simplified` (Lucia Royo-Pascual, EIUM) | ✓ | — | ✓ | ✓ (probe config, 108.7 MB) | Mass and energy balance of an air-based drying process: a function of 18 temperatures, humidities and flow rates returning a dict. |
