# fmusim: limitations

Problems that come from `fmusim` **0.0.39** ([Reference FMUs](https://github.com/modelica/Reference-FMUs)), not from fmugen or UniFMU. Releases 0.0.40 to 0.0.42 have no `fmusim` executable, so 0.0.39 is the newest one to download.

| # | Problem | Kind | Workaround |
|---|---|---|---|
| 1 | [Input clocks don't tick in co-simulation](#1-input-clocks-dont-tick-in-co-simulation) | Limitation | Use another importer for FMUs with input clocks |

---

## 1. Input clocks don't tick in co-simulation

With `--interface-type cs`, `fmusim` never calls `fmi3SetClock`, and it enters event mode only when `fmi3DoStep` returns `eventHandlingNeeded = true`, even with `--event-mode-used`. A fmugen FMU whose code runs on an input clock therefore simulates without an error, but that code never runs.

Found on 2026-11-06 with a periodic input clock, using `--log-fmi-calls`. Output clocks work: the FMU returns `eventHandlingNeeded = true`, and `fmusim` enters event mode, with `--event-mode-used` (`tests/test_importers.py`).
