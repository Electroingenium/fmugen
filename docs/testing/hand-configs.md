# Hand configs

Configs written by hand where `init` can't infer everything. They are linked from the **Hand config** column. The ones not shown here are in [match-testing-rerun/configs](match-testing-rerun/configs/), each with a comment saying what it changes and why. `PropsSI`'s is [shown in models.md](../models.md#positional-only-arguments).

## fluids.units

`Reynolds` and `head_from_P` take and return pint quantities: each input gets `convert = "pint"` and a `unit`, and `head_from_P`'s output `unit = "cm"`. Write the units the way pint reads them (`kg/m**3`, `Pa*s`, `m/s**2`): pint rejects the Modelica spelling (`kg/m3`, `Pa.s`), and fmugen then writes no `<BaseUnit>` for them. Checked: `Reynolds` gives 76401.3 and `head_from_P` 1019.7 cm, the values of `fluids.units` called directly.

## JOS3

Inputs bound to attributes (`to = "attr:tdb"`, …) and `setup = ["options.update({'limit_dshiv/dt': True})"]`; `init` writes `globals` for the module state `PRE_SHIV`. Checked: 40 one-minute steps at 10 °C give the core temperature of JOS3 run directly, also after a rollback.

## PyTCI

```toml
[model]
entry = "PyTCI.models.propofol:Marsh"
fmi_version = 2
call = "wait_time"

[parameters]
weight = { start = 70.0 }

[inputs]
time_seconds = { start = 1, type = "Integer" }
dose = { start = 2.0, unit = "mg", to = "call:give_drug" }

[outputs]
x1 = {}
x2 = {}
x3 = {}
xeo = {}
```

The model advances by `time_seconds`, not by the FMI step size. `give_drug` adds a bolus each time it's called, and a `to = "call:…"` setter runs once per step: 60 steps give 60 doses.

The first dose is given at initialization, so the outputs at t = 0 already include it: x1 = 2 mg / (0.228 L/kg × 70 kg) = 0.1253. The first step doesn't give it again. A loop that gives each dose at the start of its step (`give_drug(dose); wait_time(1)`) shows x1 = 0 at t = 0; from t = 1 the two are equal.

## Cantera

```toml
[model]
entry = "cantera:Solution"
fmi_version = 3
call = "equilibrate"
setup = ["__setattr__('TPX', (300.0, 101325.0, 'CH4:1, O2:2, N2:7.52'))"]

[parameters]
infile = { start = "gri30.yaml" }

[inputs]
XY = { start = "HP" }

[outputs]
T = { unit = "K" }
P = { unit = "Pa" }
density = {}
mean_molecular_weight = {}
```

The fuel/air state can only be set in `setup`: Cantera's `TPX` is a tuple property, and there is no way to bind inputs to it.

## pvlib read_tmy3

```toml
[model]
entry = "pvlib.iotools:read_tmy3"

[inputs]
filename = { start = "703165TY.csv" }  # pvlib's data/703165TY.csv
map_variables = { start = true }

[outputs]
ghi = { from = "return:0.ghi", dimensions = [8760], unit = "W/m2" }
dni = { from = "return:0.dni", dimensions = [8760], unit = "W/m2" }
dhi = { from = "return:0.dhi", dimensions = [8760], unit = "W/m2" }
temp_air = { from = "return:0.temp_air", dimensions = [8760], unit = "degC" }
latitude = { from = "return:1.latitude", unit = "deg" }
longitude = { from = "return:1.longitude", unit = "deg" }
altitude = { from = "return:1.altitude", unit = "m" }
TZ = { from = "return:1.TZ" }
```

`init --probe` outputs all of the TMY3 table's columns, about 260,000 values; this keeps the irradiance, the air temperature and where the site is.

## radioactivedecay

```toml
[model]
entry = "radioactivedecay:Inventory"
call = "decay"
init_call = true

[model.constants]
contents = { call = "builtins:dict([(\"C-14\",1.0)])" }

[parameters]
units = { start = "Bq" }
check = { start = true }

[inputs]
decay_time = { start = 1000.0 }
decay_units = { start = "s", to = "arg:units" }

[outputs]
"contents.C_14" = { from = "return:contents.C-14" }
"contents.N_14" = { from = "return:contents.N-14" }
```

The `init --probe` config without its 1,590 other outputs: the inventory's whole nuclide database (`decay_data`, `decay_matrices`), which isn't a result. Each step decays the starting inventory by `decay_time`; `decay` returns a new inventory and leaves the model's unchanged.

## metpy

```toml
[inputs]
temperature = { start = 25.0, unit = "degC", convert = "pint" }
relative_humidity = { start = 0.5, unit = "dimensionless", convert = "pint" }

[outputs]
y = { from = "return", unit = "degC" }
```
