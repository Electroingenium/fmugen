# 9 models compiled with PyInstaller: conclusion

fmugen `dev` at `7f07e1d`, WSL 2 (Ubuntu), PyInstaller 6.22.3.

## Conclusion

All 9 models (CFTOC (GEKKO), commpy `QAMModem`, impedance `Randles`, uncertainties `ufloat`, ambiance `Atmosphere`, `fluids.units` `Reynolds` and `head_from_P`, metpy `dewpoint_from_relative_humidity`, river `LinearRegression`) work with `fmugen init --probe` and `fmugen build --compile pyinstaller`, with no hand config. Each compiled FMU gives exactly the values of the model run directly: 157 values over t = 0–3, largest relative difference 0.

## Bug: `init --probe` scans a venv inside the model's folder (fixed)

**Symptom:** with the venv inside the model's folder (`cftoc/.venv` next to `cftoc/CFTOCSolverV2.py`), `fmugen init --probe` fails:

```
fmugen: error: ValueError: __main__.__spec__ is None
```

**Cause:** the probe lists the model's folder as `sources`. `_add_requirements` (`src/fmugen/interface.py:567`–`579`) then parses every `.py` file under it, the venv's site-packages included, and collects thousands of import names. One of them is `__main__`. When run from the `fmugen` console script, `sys.modules["__main__"].__spec__` is `None`, so `importlib.util.find_spec("__main__")` (`interface.py:596`) raises `ValueError`. Even without that error, the scan would likely add packages the model doesn't use to `requirements` (not checked).

**Fix:** `init` now skips virtual environments (folders with `pyvenv.cfg`) when it collects `sources` and the imports for `requirements`, and ignores a name `find_spec` can't look up. A module imported from the venv had also made `init` write `sources = [".venv"]`, which listed the venv's packages (pip, protobuf, pyzmq) in `requirements`. Rechecked with CFTOC and `cftoc/.venv`: `requirements = ["gekko==1.3.2", "numpy==2.5.4"]`, no `sources`, nothing from the venv in the FMU, and the same result as before.
