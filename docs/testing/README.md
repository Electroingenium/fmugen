# Testing with published models

Records of how fmugen was tested on unmodified published models. Each report is a dated snapshot at the commit it names; later fixes may have changed what it describes. The current state is in [tested-models.md](tested-models.md).

| Page | Contents |
|---|---|
| [tested-models.md](tested-models.md) | Every published model tried, with its most recent result |
| [discarded-models.md](discarded-models.md) | Libraries tried that turned out not to be models (frameworks, empty containers), and two too heavy to rerun |
| [match-testing/report.md](match-testing/report.md) | FMU vs the model run directly, first run (`adeebf3`) |
| [match-testing-rerun/report.md](match-testing-rerun/report.md) | The same comparison, rerun (`6dfc18c`), with the configs used |
| [match-testing-compile/report.md](match-testing-compile/report.md), [retest.md](match-testing-compile/retest.md) | Compiled (PyInstaller) FMUs vs uncompiled, GEKKO and hand configs |
| [match-testing-compile-rest/report.md](match-testing-compile-rest/report.md) | Compiled vs uncompiled, the remaining 100 models |
| [probe-9/CONCLUSION.md](probe-9/CONCLUSION.md) | 9 models with `init --probe` and `--compile pyinstaller` |
| [pending-100/](pending-100/) | Scripts, wrappers, hand configs and results of the 82 models tested later |
