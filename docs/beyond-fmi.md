# Beyond FMI's types

FMI variables hold numbers, booleans, strings and enumerations; FMI 3 adds arrays and bytes. Python models take and return much more. These are the ways fmugen passes what FMI's type system can't express.

## Arguments FMI can't hold

- **Arbitrary Python objects** (a molecule, an estimator, a network, a function): FMI has no object type. They live in [`[model.constants]` and `call_constants`](config.md#modelconstants-and-modelcall_constants): as literals (`{ python = "(-100.0, 100.0)" }`), as importable objects (`{ ref = "module:Name" }`), or as calls computed when the FMU initializes (`{ call = "ase.build:bulk('Cu', 'fcc', a=3.6)" }`, including from the model object itself with `self.…`). The importer never sees them.
- **Structured objects built from variables:** a dataclass, pydantic or attrs argument is split into one FMU variable per field (`to = "arg:sport.clo"`), then rebuilt before each call ([models.md](models.md#objects-built-from-their-fields)).
- **Dicts and tuples:** split into one variable per item (`to = "arg:params[x1]"`), then put back together for the call.
- **Types FMI doesn't have:** the FMU holds plain numbers, and fmugen converts them for each call:
  - torch tensors and numpy arrays (`convert = "torch:tensor"`, `numpy = true`, `convert = "numpy:atleast_1d"`)
  - pint quantities in the variable's unit (`convert = "pint"`)
  - pandas Series dated by FMU time (`convert = "series"`)
- **Date-times:** FMI has none. An argument gets `epoch + t` as a `datetime` ([`[time]`](config.md#time)).
- **`None` and other non-FMI defaults** are left to the model's own default instead of becoming variables.

## Results FMI can't hold

- **Nested results** flattened into variables: dict keys, tuple positions, NamedTuple and dataclass fields, attributes (`attr:`), getter results (`get:FinalResult`), and paths through methods (`attr:get_model().output.head().get_data()`).
- **Tables:** pandas DataFrames and Series, and xarray Datasets, read column by column.
- **Complex numbers:** split into real and imaginary outputs.
- **Generators:** read whole, within a limit.
- **Images:** a PIL image read as its pixel array.
- **Objects with no readable fields:** read through `to_json()`, `to_dict()`, `as_dict()` or `model_dump()`.
- **Arrays in FMI 2**, which has none: one scalar variable per element (`y[1,1]`, `y[1,2]`, …).
