# OMSimulator: limitations

Problems that come from [OMSimulator](https://github.com/OpenModelica/OMSimulator) **3.0.0.post198** (the Python package), not from fmugen or UniFMU.

| # | Problem | Kind | Workaround |
|---|---|---|---|
| 1 | [FMI 3 clocks can't be read](#1-fmi-3-clocks-cant-be-read) | Limitation | Use another importer for FMUs with clocks |

---

## 1. FMI 3 clocks can't be read

An FMI 3 FMU with a `<Clock>` variable fails while OMSimulator reads its model description, before anything runs: `KeyError: 'Clock'` from `SignalType[signal_type]` in `OMSimulator/variable.py`, called from `_parse_variables_fmi3` in `fmu.py`. Any FMU with clocks is affected, whichever tool built it.

Found on 2026-11-06 with a fmugen FMU with one periodic input clock (`tests/test_importers.py`). The same model without clocks simulates in FMI 2 and FMI 3 and matches FMPy.
