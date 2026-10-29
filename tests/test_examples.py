"""init on what the code says beyond signatures: the package's own usage examples (README, doctests),
documented defaults, units and value checks for start values, and, with --probe, other step methods and
classmethods when the first choice fails."""
import textwrap

from fmugen.builder import init, isolated_imports


def write(tmp_path, name, source):
    (tmp_path / name).write_text(textwrap.dedent(source))
    return tmp_path / name


def run(target, tmp_path, **kwargs):
    with isolated_imports():
        output, data = init(target, output=tmp_path / "fmugen.toml", force=True, **kwargs)
    return data, output.read_text()


TANK = """
    class Tank:
        def __init__(self, area, level0=0.0):
            self.level, self.area = level0, area

        def configure(self, outflow):
            self.k = outflow

        def advance(self, q):
            self.level += (q - self.k * self.level) / self.area
            return self.level

        def report(self):
            return str(self.level)
"""


def test_the_readme_gives_the_step_its_set_up_and_start_values(tmp_path):
    path = write(tmp_path, "tank.py", TANK)
    write(tmp_path, "README.md", """
        # tank
        ```python
        from tank import Tank
        t = Tank(area=2.5)
        t.configure(0.3)
        for i in range(10):
            h = t.advance(q=1.2)
        print(t.report())
        ```
    """)
    data, text = run(f"{path}:Tank", tmp_path, probe=True)
    assert data["model"]["call"] == "advance" and data["model"]["setup"] == ["configure(0.3)"]
    assert data["parameters"]["area"]["start"] == 2.5 and data["inputs"]["q"]["start"] == 1.2
    assert "start from the example in README.md" in text


def test_options_given_win_over_the_readme(tmp_path):
    path = write(tmp_path, "tank.py", TANK)
    write(tmp_path, "README.md", """
        ```python
        from tank import Tank
        t = Tank(area=2.5)
        t.configure(0.3)
        h = t.advance(q=1.2)
        ```
    """)
    data, _ = run(f"{path}:Tank", tmp_path, probe=True, starts={"area": 4.0}, setup=["configure(0.5)"])
    assert data["parameters"]["area"]["start"] == 4.0 and data["model"]["setup"] == ["configure(0.5)"]
    assert data["inputs"]["q"]["start"] == 1.2


def test_a_doctest_gives_a_computed_argument_as_a_constant(tmp_path):
    path = write(tmp_path, "smooth.py", '''
        def smooth(x, factor):
            """Smooth a signal.

            >>> import numpy as np
            >>> data = np.linspace(0.0, 1.0, 4)
            >>> smooth(data, factor=0.5)
            """
            return {"y": float(sum(x)) * factor}
    ''')
    data, text = run(f"{path}:smooth", tmp_path, probe=True)
    assert data["model"]["constants"]["x"] == {"call": "numpy:linspace(0.0, 1.0, 4)"}
    assert data["inputs"]["factor"]["start"] == 0.5 and "from the example in the docstring of smooth" in text


def test_an_example_that_fails_in_the_probe_is_dropped(tmp_path):
    path = write(tmp_path, "tank.py", TANK.replace("self.level, self.area = level0, area", """if area <= 0:
                raise ValueError("area must be positive")
            self.level, self.area = level0, area"""))
    write(tmp_path, "README.md", """
        ```python
        from tank import Tank
        t = Tank(area=0.0)
        t.configure(0.3)
        h = t.advance(q=1.2)
        ```
    """)
    data, text = run(f"{path}:Tank", tmp_path, probe=True, call="advance", setup=["configure(0.3)"])
    assert data["parameters"]["area"]["start"] == 1.0 and "README" not in text


def test_starts_from_documented_defaults_units_and_value_checks(tmp_path):
    path = write(tmp_path, "gas.py", '''
        class Gas:
            """A gas.

            Parameters
            ----------
            T : float
                Temperature, [K]
            tol : float, default 1e-6
                Tolerance.
            """
            def __init__(self, T, tol, n):
                if n < 3:
                    raise ValueError("n must be at least 3")
                self.T, self.tol, self.n = T, tol, n

            def density(self, p):
                return {"rho": p / (287.0 * self.T)}
    ''')
    data, text = run(f"{path}:Gas", tmp_path, probe=True, call="density")
    parameters = data["parameters"]
    assert (parameters["T"]["start"], parameters["tol"]["start"], parameters["n"]["start"]) == (293.15, 1e-6, 10.0)
    assert "its docstring's default" in text and "passes the code's checks" in text
    assert data["inputs"]["p"]["start"] == 1.0   # nothing said of it: made up


def test_a_classmethod_builds_the_object_and_the_probe_picks_a_working_step(tmp_path):
    path = write(tmp_path, "plant.py", """
        class Plant:
            def __init__(self, config):
                if not isinstance(config, dict):
                    raise TypeError("config must be a dict")
                self.k = config["k"]

            @classmethod
            def default(cls):
                return cls({"k": 2.0})

            def compute(self, u):
                raise RuntimeError("needs a license")

            def other(self, u):
                return None

            def respond(self, u):
                return {"y": self.k * u}
    """)
    data, text = run(f"{path}:Plant", tmp_path, probe=True)
    assert data["model"]["create"] == "default" and data["model"]["call"] == "respond"
    assert "picked by probing: compute(), other() failed or gave no outputs" in text


def test_an_iterator_steps_with_next(tmp_path):
    path = write(tmp_path, "count.py", """
        class Counter:
            def __init__(self):
                self.n = 0

            def __iter__(self):
                return self

            def __next__(self):
                self.n += 1
                return self.n
    """)
    data, _ = run(f"{path}:Counter", tmp_path, probe=True)
    assert data["model"]["call"] == "__next__" and data["outputs"]


def test_a_method_showing_itself_is_the_step_unless_it_is_a_variant(tmp_path):
    # like pyproj's transform (shown in its own docstring: the step) and timezonefinder's timezones_at
    # (shown in its own docstring, but a variant of timezone_at)
    path = write(tmp_path, "geo.py", '''
        class Converter:
            def __init__(self, scale=2.0):
                self.scale = scale

            def convert(self, x):
                """Convert.

                >>> c = Converter()
                >>> c.convert(3.5)
                """
                return {"y": self.scale * x}

            def describe(self):
                return "converter"


        class Finder:
            def zone_at(self, lng, lat):
                return {"zone": lng + lat}

            def zones_at(self, lng, lat):
                """All zones.

                >>> f = Finder()
                >>> f.zones_at(lng=1.5, lat=2.5)
                """
                return {"zones": [lng, lat]}
    ''')
    data, _ = run(f"{path}:Converter", tmp_path, probe=True)
    assert data["model"]["call"] == "convert" and data["inputs"]["x"]["start"] == 3.5
    data, _ = run(f"{path}:Finder", tmp_path, probe=True)
    assert data["model"].get("call") != "zones_at"


def test_an_example_step_that_gives_no_outputs_is_dropped(tmp_path):
    # like openturns' Normal: its docstring's getSample(5) returns an object FMI can't hold
    path = write(tmp_path, "dist.py", '''
        class Sample:
            pass


        class Dist:
            """A distribution.

            >>> d = Dist()
            >>> s = d.get_sample(5)
            """
            def __init__(self, mu=0.0):
                self.mu = mu

            def get_sample(self, size):
                return Sample()

            def compute_pdf(self, x):
                return {"pdf": x - self.mu}
    ''')
    data, _ = run(f"{path}:Dist", tmp_path, probe=True)
    assert data["model"]["call"] != "get_sample" and data["outputs"]


def test_a_step_failing_before_its_set_up_is_kept(tmp_path):
    # like tabnet's predict() before fit(): no network yet; predict_func() works but isn't the step
    path = write(tmp_path, "net.py", '''
        class Regressor:
            def predict(self, X):
                return {"y": self.network(X)}

            def predict_func(self, outputs):
                return {"z": outputs}

            def explain(self, X):
                return {"mask": self.network(X)}
    ''')
    data, text = run(f"{path}:Regressor", tmp_path, probe=True)
    assert data["model"]["call"] == "predict" and "object has no attribute 'network'" in text


def test_a_call_passing_its_arguments_to_forward_takes_forwards(tmp_path):
    # like a torch module: __call__(*args, **kwargs) runs forward(input_tensor)
    path = write(tmp_path, "cell.py", """
        class Module:
            def __call__(self, *args, **kwargs):
                return self.forward(*args, **kwargs)


        class Cell(Module):
            def __init__(self, gain=2.0):
                self.gain = gain

            def forward(self, input_tensor):
                return {"v": self.gain * input_tensor}
    """)
    data, _ = run(f"{path}:Cell", tmp_path, probe=True)
    assert data["inputs"]["input_tensor"]["start"] == 1.0 and data["outputs"]["v"] == {"from": "return:v"}


def test_sizes_are_made_up_as_integers_then_as_lists(tmp_path):
    # like torch layers (hidden_size must be an int) and FNO (n_modes must be a sequence)
    path = write(tmp_path, "net.py", """
        class Net:
            def __init__(self, hidden_size, n_modes):
                if not isinstance(hidden_size, int):
                    raise TypeError("size must be an int")
                self.weights = [0.5] * hidden_size * len(n_modes)

            def predict(self, x):
                return {"y": sum(self.weights) * x}
    """)
    data, text = run(f"{path}:Net", tmp_path, probe=True)
    assert data["parameters"]["hidden_size"]["start"] == 1 and data["outputs"]
    assert "dimensions" in data["parameters"]["n_modes"]   # a sequence: arrays of ones, else lists


def test_an_environment_steps_with_an_action_from_its_space_after_reset(tmp_path):
    # like gymnasium's CartPole: step() before reset() fails, and the action must be in its Discrete space
    path = write(tmp_path, "env.py", """
        class Discrete:
            def __init__(self, n, start=0):
                self.n, self.start = n, start

            def contains(self, x):
                return isinstance(x, int) and self.start <= x < self.start + self.n


        class Env:
            def __init__(self):
                self.action_space = Discrete(2)
                self.state = None

            def reset(self, seed=None):
                self.state = 0.0
                return self.state

            def step(self, action):
                assert self.action_space.contains(action), f"{action!r} invalid"
                self.state += 1.0 if action else -1.0
                return {"x": self.state}
    """)
    data, text = run(f"{path}:Env", tmp_path, probe=True)
    assert data["model"]["setup"] == ["reset"] and data["inputs"]["action"]["start"] == 0
    assert "from the model's action or observation space" in text


def test_one_of_several_optional_arguments_is_made_up_when_the_call_needs_it(tmp_path):
    # like fluids' Reynolds(V, D, rho=None, mu=None, nu=None): rho and mu, or nu
    path = write(tmp_path, "flow.py", """
        def reynolds(V, D, rho=None, mu=None, nu=None):
            if rho is not None and mu is not None:
                return V * D * rho / mu
            if nu is not None:
                return V * D / nu
            raise ValueError("Either density and viscosity, or kinematic viscosity is needed")
    """)
    data, text = run(f"{path}:reynolds", tmp_path, probe=True)
    assert data["inputs"]["nu"] == {"start": 1.0} and "rho" not in data["inputs"] and data["outputs"]
    assert "the probe failed without nu" in text


def test_an_enum_argument_with_no_default_starts_at_its_first_member(tmp_path):
    # like pyfluids' Fluid(name: FluidsList)
    path = write(tmp_path, "fluid.py", """
        import enum

        class FluidsList(enum.Enum):
            Water = "water"
            Air = "air"

        def density(name: FluidsList, T: float = 293.15):
            return {"rho": 1000.0 if name.value == "water" else 1.2}
    """)
    data, text = run(f"{path}:density", tmp_path, probe=True)
    assert data["inputs"]["name"]["start"] == "Water" and "FluidsList's first member" in text


def test_a_function_returning_the_model_steps_with_the_method_the_starts_fit(tmp_path):
    # like joblib's load(filename) returning an estimator whose predict(X) runs
    path = write(tmp_path, "store.py", """
        class Estimator:
            def predict(self, X):
                return {"y": 2.0 * X}

            def score(self, X, y):
                return 1.0

        def load(filename):
            return Estimator()
    """)
    data, text = run(f"{path}:load", tmp_path, probe=True, starts={"filename": "model.pkl", "X": 3.0})
    assert data["model"]["call"] == "predict" and data["inputs"]["X"]["start"] == 3.0
    assert "returns the model: X are arguments of its predict()" in text


def test_an_input_the_error_asks_to_be_text_is_made_up_as_text(tmp_path):
    # like sentence-transformers' encode(sentences): "Unsupported input type: float. Expected one of: str, ..."
    path = write(tmp_path, "embed.py", """
        class Encoder:
            def encode(self, sentences):
                if not isinstance(sentences, str):
                    raise ValueError(f"Unsupported input type: {type(sentences).__name__}. Expected one of: str")
                return {"length": float(len(sentences))}
    """)
    data, text = run(f"{path}:Encoder", tmp_path, probe=True, call="encode")
    assert isinstance(data["inputs"]["sentences"]["start"], str) and data["outputs"]
    assert "made up as text" in text


def test_a_callable_without_signature_gets_the_positional_numbers_its_error_asks_for(tmp_path):
    # like sgp4's C method sgp4(jd, fr): "sgp4() takes exactly 2 arguments (0 given)"
    path = write(tmp_path, "orbit.py", """
        class Satellite:
            def propagate(self, *args):
                if len(args) != 2:
                    raise TypeError(f"propagate() takes exactly 2 arguments ({len(args)} given)")
                return {"r": args[0] + args[1]}
    """)
    data, text = run(f"{path}:Satellite", tmp_path, probe=True, call="propagate")
    assert data["inputs"] == {"x0": {"start": 1.0, "to": "pos:0"}, "x1": {"start": 1.0, "to": "pos:1"}}
    assert "takes 2 numbers" in text


def test_a_made_up_input_takes_the_shape_the_error_names(tmp_path):
    # like roboticstoolbox's Panda.fkine(q): "cannot reshape array of size 1 into shape (7)"
    path = write(tmp_path, "arm.py", """
        import numpy

        class Arm:
            def fkine(self, q):
                q = numpy.asarray(q, dtype=float).reshape(5)
                return {"reach": float(q.sum())}
    """)
    data, _ = run(f"{path}:Arm", tmp_path, probe=True, call="fkine", fmi_version=3)
    assert data["inputs"]["q"]["dimensions"] == [5]


def test_a_forecaster_trained_on_a_dated_table_is_probed_on_a_made_up_one(tmp_path):
    # like prophet: fit(df) with columns ds and y, then predict(); the config leaves fit() to the user
    pytest = __import__("pytest")
    pytest.importorskip("pandas")
    path = write(tmp_path, "fc.py", """
        import pandas

        class Forecaster:
            def __init__(self):
                self.history = None

            def fit(self, df):
                if not isinstance(df, pandas.DataFrame) or list(df.columns) != ["ds", "y"]:
                    raise ValueError("Dataframe must have columns 'ds' and 'y' with the dates and values")
                self.history = df

            def predict(self):
                if self.history is None:
                    raise Exception("Model has not been fit.")
                return {"yhat": float(self.history["y"].mean())}
    """)
    data, text = run(f"{path}:Forecaster", tmp_path, probe=True, call="predict")
    assert "setup" not in data["model"] and '# setup = ["fit(...)"]' in text and "ds and y" in text
    assert data["outputs"]["yhat"] == {"from": "return:yhat"}


def test_a_factory_taking_kwargs_returns_the_model_the_leftover_starts_call(tmp_path):
    # like transformers' pipeline(task, model, **kwargs): inputs is an argument of calling what it returns
    path = write(tmp_path, "pipe.py", """
        class Classifier:
            def __init__(self, task):
                self.task = task

            def __call__(self, inputs):
                return {"label": float(len(inputs))}

        def pipeline(task, **kwargs):
            return Classifier(task)
    """)
    data, text = run(f"{path}:pipeline", tmp_path, probe=True, starts={"task": "sentiment", "inputs": "I like it"})
    assert data["model"]["call"] == "__call__" and data["inputs"]["inputs"]["start"] == "I like it"
    assert "returns the model: inputs are arguments of calling it" in text


def test_a_function_returning_a_function_is_called_on_every_step(tmp_path):
    # like sympy's lambdify(args, expr): it makes the function the model is
    path = write(tmp_path, "make.py", """
        def make(gain=2.0):
            def model(x, y):
                return {"z": gain * x + y}
            return model
    """)
    data, text = run(f"{path}:make", tmp_path, probe=True)
    assert data["model"]["call"] == "__call__" and set(data["inputs"]) >= {"x", "y"}
    assert "returns a function" in text


def test_a_set_up_argument_annotated_with_a_class_is_built_from_it(tmp_path):
    # like timesfm's compile(forecast_config: ForecastConfig) before forecast()
    path = write(tmp_path, "tfm.py", """
        class ForecastConfig:
            def __init__(self, horizon=4):
                self.horizon = horizon

        class Model:
            def __init__(self):
                self.config = None

            def compile(self, forecast_config: ForecastConfig):
                self.config = forecast_config

            def forecast(self, x):
                if self.config is None:
                    raise RuntimeError("Model is not compiled. Please call compile() first.")
                return {"y": x * self.config.horizon}
    """)
    data, _ = run(f"{path}:Model", tmp_path, probe=True, call="forecast")
    assert data["model"]["setup"] == ["compile(tfm.ForecastConfig())"]


def test_an_optional_constructor_argument_is_made_up_when_building_fails(tmp_path):
    # like reservoirpy's Reservoir(units=None): "'units' parameter must not be None"
    path = write(tmp_path, "res.py", """
        class Reservoir:
            def __init__(self, units=None, lr=1.0):
                if units is None:
                    raise ValueError("'units' parameter must not be None")
                self.state = [0.0] * units

            def step(self, x):
                self.state = [s + x for s in self.state]
                return {"y": sum(self.state)}
    """)
    data, text = run(f"{path}:Reservoir", tmp_path, probe=True, call="step")
    assert data["parameters"]["units"] == {"start": 1} and "the probe failed without units" in text


OPTION = """
    class Option:
        def __init__(self, S, K):
            self.S, self.K = S, K

        def price(self):
            return {"price": max(self.S - self.K, 0.0)}

        def delta(self):
            return {"delta": 1.0}

        def gamma(self):
            return {"gamma": 0.0}
"""


def test_a_readme_showing_several_queries_leads_with_the_step(tmp_path):
    # like blackscholes' README: price(), then delta(), gamma(); a bash block's fence opens no Python block
    path = write(tmp_path, "opt.py", OPTION)
    write(tmp_path, "README.md", """
        Install it:
        ```bash
        pip install opt
        ```
        Some prose: not code.
        ```python
        from opt import Option
        call = Option(S=55.0, K=50.0)
        call.price()
        call.delta()
        call.gamma()
        ```
    """)
    data, _ = run(f"{path}:Option", tmp_path, probe=True)
    assert data["model"]["call"] == "price" and "setup" not in data["model"]


def test_a_method_iterated_in_a_loop_header_is_the_step(tmp_path):
    # like nashpy's README: for eq in game.support_enumeration(): print(eq)
    path = write(tmp_path, "game.py", """
        class Game:
            def __init__(self, a):
                self.a = a

            def payoffs(self):
                return {"p": self.a}

            def equilibria(self):
                yield {"e": self.a}
    """)
    write(tmp_path, "README.md", """
        ```python
        from game import Game
        game = Game(a=2.0)
        print(game.payoffs())
        for eq in game.equilibria():
            print(eq)
        ```
    """)
    data, _ = run(f"{path}:Game", tmp_path, probe=True)
    assert data["model"]["call"] == "equilibria"


def test_an_argument_no_example_of_the_model_shows_is_taken_from_another_call_of_the_package(tmp_path):
    path = write(tmp_path, "tank.py", TANK + """

    class Pump:
        def __init__(self, area):
            self.area = area
""")
    write(tmp_path, "README.md", """
        ```python
        from tank import Pump
        p = Pump(area=2.5)
        ```
    """)
    data, text = run(f"{path}:Tank", tmp_path, probe=True, call="advance")
    assert data["parameters"]["area"]["start"] == 2.5
    assert "start from the example in README.md" in text


def test_a_dict_argument_is_made_up_from_the_error(tmp_path):
    path = write(tmp_path, "learner.py", """
        class Learner:
            def learn(self, x):
                if not isinstance(x, dict):   # river's check
                    raise TypeError(f"'{type(x).__name__}' object is not an instance of 'dict'")
                self.total = sum(x.values())
                return self.total
    """)
    data, text = run(f"{path}:Learner", tmp_path, probe=True, call="learn")
    assert data["inputs"]["a"]["to"] == "arg:x[a]" and "y" in data["outputs"]


def test_an_object_argument_is_made_up_with_the_attributes_the_errors_name(tmp_path):
    path = write(tmp_path, "patient.py", """
        class Patient:
            def step(self, action):
                return action.carbs * 2 + action.insulin
    """)
    data, text = run(f"{path}:Patient", tmp_path, probe=True, call="step")
    assert "made up as an object with carbs, insulin" in text and "y" in data["outputs"]


def test_keys_the_step_looks_up_in_its_keyword_arguments_are_made_up(tmp_path):
    path = write(tmp_path, "surface.py", """
        class Surface:
            def __call__(self, **params):
                return params["depth"] * 2 + params["width"]
    """)
    data, text = run(f"{path}:Surface", tmp_path, probe=True, call="__call__")
    assert {"depth", "width"} <= set(data["inputs"]) and "looks it up in its keyword arguments" in text


def test_arguments_an_error_asks_for_are_made_up(tmp_path):
    path = write(tmp_path, "ttm.py", """
        class Model:
            def __init__(self, path, **options):
                if "context_length" not in options:
                    raise ValueError("Provide `context_length` when `path` is a hub path")
                self.length = options["context_length"]

            def forward(self, x):
                return x * self.length
    """)
    data, text = run(f"{path}:Model", tmp_path, probe=True, call="forward", starts={"path": "hub/model"})
    assert data["parameters"]["context_length"]["start"] == 1 and "it asks for context_length" in text


def test_a_made_up_number_the_model_refuses_is_tried_larger(tmp_path):
    path = write(tmp_path, "modem.py", """
        class Modem:
            def __init__(self, m):
                if m not in (4, 16, 64):
                    raise OverflowError("cannot convert")
                self.m = m

            def modulate(self, bits):
                return bits * self.m
    """)
    data, text = run(f"{path}:Modem", tmp_path, probe=True, call="modulate")
    assert data["parameters"]["m"]["start"] == 4


def test_a_series_argument_is_made_up_as_a_dated_series(tmp_path):
    path = write(tmp_path, "level.py", """
        import pandas

        class Model:
            def __init__(self, oseries):
                if not isinstance(oseries, pandas.Series):
                    raise ValueError("oseries must be a series")
                self.mean = float(oseries.mean())

            def simulate(self, k):
                return self.mean * k
    """)
    data, text = run(f"{path}:Model", tmp_path, probe=True, call="simulate")
    assert "made up as a dated pandas Series" in text and "y" in data["outputs"]
