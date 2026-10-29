"""init --probe on model shapes found in published libraries: C built-ins without a signature, steps that
are functions taking the object, histories that grow, results nested in objects or tables, functions made
at runtime, attributes an object makes up, and functions writing their results onto an argument."""
import textwrap

import pytest

from fmugen.builder import init, isolated_imports
from fmugen.config import InterfaceError


def write(tmp_path, name, source):
    (tmp_path / name).write_text(textwrap.dedent(source))
    return tmp_path / name


def probe(target, tmp_path, **kwargs):
    with isolated_imports():
        output, data = init(target, output=tmp_path / "fmugen.toml", probe=True, **kwargs)
    return data, output.read_text()


def test_starts_of_a_method_without_signature_go_to_the_step_when_it_takes_them(tmp_path):
    # like ephem's Sun: Sun(date) computes once, Sun().compute(date) on every step; compute is a C
    # built-in, and its position can be read only after a compute
    path = write(tmp_path, "body.py", """
        class Sun:
            def __init__(self, *args):
                self._date = None
                if args:
                    self.compute(*args)

            def compute(self, *args):
                (self._date,) = args

            @property
            def ra(self):
                if self._date is None:
                    raise RuntimeError("field ra undefined until the first compute()")
                return 2.0 * self._date

        Sun.compute.__signature__ = "no signature"   # inspect.signature fails, as for a C built-in
    """)
    data, text = probe(f"{path}:Sun", tmp_path, call="compute", starts={"date": 3.0})
    assert data["inputs"] == {"date": {"start": 3.0, "to": "pos:0"}}
    assert "parameters" not in data
    assert data["outputs"]["ra"] == {}
    assert data["model"]["init_call"] is True
    assert "the probe worked with nothing for the constructor and date for compute()" in text


def test_starts_split_between_a_create_classmethod_and_a_step_without_signatures(tmp_path):
    # like sgp4's Satrec.twoline2rv(line1, line2).sgp4(jd, fr)
    path = write(tmp_path, "sat.py", """
        class Satrec:
            @classmethod
            def twoline2rv(cls, *args):
                line1, line2 = args
                sat = cls()
                sat.n = len(line1) + len(line2)
                return sat

            def sgp4(self, *args):
                jd, fr = args
                return (0, self.n * (jd + fr))

        Satrec.twoline2rv.__func__.__signature__ = "no signature"
        Satrec.sgp4.__signature__ = "no signature"
    """)
    data, _ = probe(f"{path}:Satrec", tmp_path, call="sgp4", create="twoline2rv",
                    starts={"line1": "ab", "line2": "cde", "jd": 1.0, "fr": 0.5})
    assert data["parameters"] == {"line1": {"start": "ab", "to": "pos:0"}, "line2": {"start": "cde", "to": "pos:1"}}
    assert data["inputs"] == {"jd": {"start": 1.0, "to": "pos:0"}, "fr": {"start": 0.5, "to": "pos:1"}}
    assert data["outputs"]["y"] == {"from": "return", "dimensions": [2]}


def test_a_step_that_is_a_function_taking_the_object(tmp_path, monkeypatch):
    # like pandapipes: gas_meshed_delta() makes the network, pipeflow(net) writes its results onto it
    write(tmp_path, "pipes.py", """
        class Net:
            def __init__(self):
                self.p_bar = [0.0, 0.0, 0.0]

        def network():
            return Net()

        def pipeflow(net):
            net.p_bar = [1.0, 0.9, 0.8]
    """)
    monkeypatch.syspath_prepend(str(tmp_path))
    data, _ = probe("pipes:network", tmp_path, call="pipes:pipeflow(self)", fmi_version=3)
    assert data["model"]["call"] == "pipes:pipeflow(self)"
    assert data["outputs"]["p_bar"]["dimensions"] == [3]


def test_c_attributes_the_step_changes_are_outputs(tmp_path):
    # like imufusion's Ahrs: update() returns nothing and changes the quaternion, a C attribute readable before
    path = write(tmp_path, "ahrs.py", """
        class Ahrs:
            __slots__ = ("quaternion", "settings")

            def __init__(self):
                self.quaternion = 1.0
                self.settings = 0.5

            def update(self, gyro=0.1):
                self.quaternion -= gyro
                return self   # for chaining, as imufusion's update_no_magnetometer: no result of its own
    """)
    data, _ = probe(f"{path}:Ahrs", tmp_path, call="update")
    assert data["outputs"] == {"quaternion": {}}   # settings didn't change


def test_columns_of_result_tables_the_step_fills(tmp_path, monkeypatch):
    pytest.importorskip("pandas")
    # like pandapipes: the network is a dict with attribute access; pipeflow(net) fills net.res_junction
    write(tmp_path, "pipes2.py", """
        import pandas

        class Net(dict):
            def __getattr__(self, name):
                try:
                    return self[name]
                except KeyError:
                    raise AttributeError(name) from None

            def __dir__(self):
                return [*super().__dir__(), *self]

        def network():
            return Net(junction=pandas.DataFrame({"pn_bar": [1.0, 1.0]}),
                       res_junction=pandas.DataFrame({"p_bar": [0.0, 0.0]}))

        def pipeflow(net):
            net["res_junction"] = pandas.DataFrame({"p_bar": [0.9, 0.8]})
    """)
    monkeypatch.syspath_prepend(str(tmp_path))
    data, _ = probe("pipes2:network", tmp_path, call="pipes2:pipeflow(self)", fmi_version=3)
    assert data["outputs"] == {"res_junction.p_bar": {"from": "attr:res_junction.p_bar", "dimensions": [2]}}


def test_const_values_are_fixed_and_their_files_copied(tmp_path):
    # like ultralytics' YOLO(model="yolo11n.pt").predict(source="bus.jpg"): the weights are a fixed
    # value (--const), the image an input (--start); both files are copied into the FMU
    (tmp_path / "w.bin").write_text("2.0")
    (tmp_path / "img.txt").write_text("3.0")
    path = write(tmp_path, "det.py", """
        class Detector:
            def __init__(self, model):
                self.k = float(open(model).read())

            def predict(self, source):
                return self.k * float(open(source).read())
    """)
    data, text = probe(f"{path}:Detector", tmp_path, call="predict",
                       starts={"model": {"python": "'w.bin'"}, "source": "img.txt"})
    assert data["model"]["constants"] == {"model": "w.bin"}   # a plain TOML value, not a variable
    assert "parameters" not in data and data["inputs"] == {"source": {"start": "img.txt"}}
    assert sorted(data["model"]["sources"]) == ["img.txt", "w.bin"] and data["model"]["cwd"] == "."
    assert data["outputs"] == {"y": {"from": "return"}}


def test_the_start_names_pick_the_step_method(tmp_path):
    # like pythermalcomfort's JOS3: no step/update method, several public ones; times and dtime fit only simulate
    path = write(tmp_path, "body.py", """
        class Body:
            def __init__(self, height=1.7):
                self.height, self.t = height, 0.0

            def simulate(self, times, dtime=60.0):
                self.t += times * dtime
                return {"t_core": 37.0 + 0.001 * self.t}

            def results(self):
                return {"t": self.t}

            def to_csv(self, path):
                pass
    """)
    data, text = probe(f"{path}:Body", tmp_path, starts={"times": 1, "dtime": 60.0})
    assert data["model"]["call"] == "simulate" and set(data["inputs"]) == {"times", "dtime"}
    assert "simulate() from the --start names" in text
    data, text = probe(f"{path}:Body", tmp_path, force=True)   # no names: simulate is guessed, with a warning
    assert data["model"]["call"] == "simulate" and "simulate() guessed" in text
    assert "candidates, best first: simulate, results, to_csv" in text


def test_the_start_names_win_over_the_only_own_method(tmp_path):
    # like scikit-fem's MeshTri: it defines element_finder; refined(times_or_ix), the step, is inherited
    path = write(tmp_path, "mesh.py", """
        class Mesh:
            def refined(self, times_or_ix=1):
                return {"n": 4 ** times_or_ix}

        class MeshTri(Mesh):
            def element_finder(self, mapping=None):
                return None
    """)
    data, _ = probe(f"{path}:MeshTri", tmp_path, starts={"times_or_ix": 2})
    assert data["model"]["call"] == "refined" and data["outputs"] == {"n": {"from": "return:n"}}


def test_start_names_the_constructor_also_takes_pick_the_method(tmp_path):
    # like ahrs' Madgwick(gyr=None, acc=None, **kwargs): q, gyr, acc fit updateIMU; updateMARG also needs mag
    path = write(tmp_path, "ahrs.py", """
        class Madgwick:
            def __init__(self, gyr=None, acc=None, mag=None, **kwargs):
                self.gain = 0.1

            def updateIMU(self, q, gyr, acc, dt=None):
                return [q[0] - self.gain * gyr[2], q[1], q[2], q[3] + acc[2] * 0.0]

            def updateMARG(self, q, gyr, acc, mag, dt=None):
                return q
    """)
    data, text = probe(f"{path}:Madgwick", tmp_path, fmi_version=3,
                       starts={"q": [1.0, 0.0, 0.0, 0.0], "gyr": [0.0, 0.0, 0.01], "acc": [0.0, 0.0, 9.81]})
    assert data["model"]["call"] == "updateIMU" and set(data["inputs"]) >= {"q", "gyr", "acc"}
    assert "updateIMU() from the --start names" in text


def test_the_method_taking_most_start_names_wins(tmp_path):
    # like pyquaternion's Quaternion(axis=..., angle=...).rotate(vector): vector fits rotate alone
    path = write(tmp_path, "quat.py", """
        class Quaternion:
            def __init__(self, axis=(0.0, 0.0, 1.0), angle=0.0):
                self.angle = angle

            def rotate(self, vector):
                return [v * 2.0 for v in vector]

            def normalise(self):
                return self

            def distance(self, other=None):
                return 0.0
    """)
    data, _ = probe(f"{path}:Quaternion", tmp_path, fmi_version=3, starts={"angle": 0.5, "vector": [1.0, 0.0, 0.0]})
    assert data["model"]["call"] == "rotate"


def test_two_fitting_methods_warn_with_the_candidates(tmp_path, capsys):
    # like pedon's Genuchten: h fits theta(h), s(h), k(h)
    path = write(tmp_path, "soil.py", """
        class Soil:
            def theta(self, h):
                return 0.4 - 0.001 * h

            def k(self, h):
                return 1e-5 / (1 + h)
    """)
    data, text = probe(f"{path}:Soil", tmp_path, starts={"h": 1.0})
    assert data["model"]["call"] == "k" and "candidates, best first: k, theta" in text
    assert "picked k() (candidates, best first: k, theta); pass --call METHOD" in capsys.readouterr().err


def test_a_function_returning_a_callable_model_is_a_factory(tmp_path):
    # like silero_vad's load_silero_vad(): it takes no x or sr; the model it returns does, when called
    path = write(tmp_path, "vad.py", """
        class Model:
            def __init__(self, gain):
                self.gain = gain

            def __call__(self, x, sr=16000):
                return self.gain * x * sr / 16000

            def audio_forward(self, x, sr):
                return x

        def load(gain=2.0):
            return Model(gain)
    """)
    data, text = probe(f"{path}:load", tmp_path, starts={"x": 0.5, "sr": 16000})
    assert data["model"]["call"] == "__call__" and set(data["inputs"]) == {"x", "sr"}
    assert data["parameters"] == {"gain": {"start": 2.0, "variability": "tunable"}}
    assert "load() returns the model: x, sr are arguments of calling it" in text


CRYSTAL = """
    class Atoms:                 # like ase.Atoms: builds with no arguments (an empty structure)
        def __init__(self, symbols=""):
            self.n = len(symbols)

    class Grid:                  # like py-pde's grids: needs its sizes
        def __init__(self, shape):
            self.shape = shape
"""


def test_an_annotated_object_argument_gets_a_default_instance(tmp_path, monkeypatch):
    # like mace / matgl: get_potential_energy(atoms: Atoms); Atoms() is a valid, if trivial, structure
    write(tmp_path, "crystal.py", CRYSTAL)
    path = write(tmp_path, "energy.py", """
        from crystal import Atoms, Grid

        def energy(atoms: Atoms, scale: float = 1.0):
            return {"E": -1.0 * atoms.n * scale}

        def field(grid: Grid, t: float = 0.0):
            return {"n": float(grid.shape)}
    """)
    monkeypatch.syspath_prepend(str(tmp_path))
    data, text = probe(f"{path}:energy", tmp_path)
    assert data["model"]["constants"] == {"atoms": {"call": "crystal:Atoms()"}}
    assert "inputs" in data and set(data["inputs"]) == {"scale"} and data["outputs"] == {"E": {}}
    assert ('atoms = { call = "crystal:Atoms()" }  # no default in the code: Atoms() (its annotation, with no '
            "arguments); give your own with --const atoms=call:...") in text
    with isolated_imports():   # without --probe nothing is built: a commented suggestion
        output, plain = init(f"{path}:energy", output=tmp_path / "plain.toml")
    text = output.read_text()
    assert "constants" not in plain["model"] and "atoms" not in plain.get("inputs", {})
    assert '# atoms = { call = "crystal:Atoms()" }  # no default in the code: uncomment to use Atoms()' in text
    with isolated_imports():   # Grid() fails: no constant, as before
        _, data = init(f"{path}:field", output=tmp_path / "field.toml", probe=True)
    assert "constants" not in data["model"] and "grid" in data["inputs"]


def test_a_class_with_only_properties_is_a_function(tmp_path):
    # like ambiance's Atmosphere(h): built with the inputs, results as properties; no method to run
    path = write(tmp_path, "atm.py", """
        class Atmosphere:
            def __init__(self, h, check_bounds=True):
                self.h = h

            @property
            def temperature(self) -> float:
                return 288.15 - 0.0065 * self.h

            @classmethod
            def from_pressure(cls, p):
                return cls(0.0)

        class Meter:              # a method: still needs --call
            def __init__(self, rate=1.0):
                self.rate = rate

            def integrated(self, data):
                return sum(data)

            def other(self, data):
                return 0.0
    """)
    data, text = probe(f"{path}:Atmosphere", tmp_path, starts={"h": 1000.0})
    assert data["model"]["kind"] == "function" and data["inputs"]["h"] == {"start": 1000.0}
    assert "Atmosphere has no methods to run" in text
    with isolated_imports():
        _, data = init(f"{path}:Meter", output="-")   # a method: guessed, with a warning
    assert data["model"]["call"] == "integrated"


def test_a_library_reading_its_own_sources_warns_about_compile(tmp_path, monkeypatch):
    # like py-pde, which loads pde/backends/*/config.py by path when imported
    import sysconfig
    site = tmp_path / "site"
    (site / "readsrc").mkdir(parents=True)
    (site / "readsrc" / "config.py").write_text("SPEED = 2.0\n")
    (site / "readsrc" / "__init__.py").write_text(textwrap.dedent("""
        import importlib.util, pathlib
        _spec = importlib.util.spec_from_file_location("readsrc_config", pathlib.Path(__file__).parent / "config.py")
        _cfg = importlib.util.module_from_spec(_spec)
        _spec.loader.exec_module(_cfg)

        def run(x=1.0):
            return {"y": _cfg.SPEED * x}
    """))
    monkeypatch.syspath_prepend(str(site))
    paths = sysconfig.get_paths()
    monkeypatch.setattr(sysconfig, "get_paths", lambda *a, **k: {**paths, "purelib": str(site), "platlib": str(site)})
    _, text = probe("readsrc:run", tmp_path)
    assert "# --compile won't work: the model's libraries read their own source files by path (readsrc/config.py)" in text


def test_tunable_parameters_are_checked_by_the_probe(tmp_path):
    # like simple-pid's PID: Kp is read on every step (tunable); sample_count only when built (fixed)
    path = write(tmp_path, "ctl.py", """
        class Ctl:
            def __init__(self, Kp=2.0, n=3):
                self.Kp, self.n = Kp, n
                self.buffer = [0.0] * n

            def step(self, e=1.0):
                return {"u": self.Kp * e + len(self.buffer)}
    """)
    data, text = probe(f"{path}:Ctl", tmp_path)
    assert data["parameters"]["Kp"]["variability"] == "tunable"
    assert "variability" not in data["parameters"]["n"]   # only a hint: zero error would hide a gain too
    assert "changing it in the probe changed nothing, with these start values" in text


def test_an_input_returned_with_a_new_value_is_a_state(tmp_path):
    # like a hand-written integrator: integrate(level, inflow) returns the next level under the same name
    path = write(tmp_path, "tank.py", """
        def integrate(level=1.0, inflow=0.5, dt: float = 1.0):
            return {"level": level + inflow * dt, "overflow": float(level > 10.0)}
    """)
    data, text = probe(f"{path}:integrate", tmp_path)
    assert data["states"] == {"level": {"start": 1.0, "next": "return:level"}}
    assert set(data["inputs"]) == {"inflow"} and "overflow" in data["outputs"]
    assert "the model returns a new level" in text


def test_units_come_from_numpydoc_docstrings(tmp_path):
    # like fluids' Reynolds: "rho : float\n    Density, [kg/m^3]"
    path = write(tmp_path, "flow.py", '''
        def reynolds(V=1.0, D=0.1, rho=1000.0, mu=1e-3):
            """Reynolds number.

            Parameters
            ----------
            V : float
                Velocity, [m/s]
            D : float
                Diameter, [m]
            rho : float
                Density, [kg/m^3]
            mu : float
                Dynamic viscosity of the fluid, [Pa*s]

            Returns
            -------
            Re : float
                Reynolds number, [-]
            """
            return V * D * rho / mu
    ''')
    data, text = probe(f"{path}:reynolds", tmp_path)
    assert {n: i.get("unit") for n, i in data["inputs"].items()} == {"V": "m/s", "D": "m", "rho": "kg/m^3", "mu": "Pa*s"}
    assert "unit" not in data["outputs"]["y"]   # [-]: dimensionless, no unit
    assert 'rho = { start = 1000.0, unit = "kg/m^3" }  # unit from the docstring' in text


def test_inputs_through_kwargs_are_retried_by_position(tmp_path):
    # like neuromancer's Block.forward(*inputs, **kwargs): data only by position
    path = write(tmp_path, "block.py", """
        class Block:
            def forward(self, *inputs, **kwargs):
                if kwargs:
                    raise TypeError(f"Block.forward() got an unexpected keyword argument {next(iter(kwargs))!r}")
                return {"y": 2.0 * sum(inputs[0])}
    """)
    data, text = probe(f"{path}:Block", tmp_path, call="forward", starts={"x": [0.5, -0.25]}, fmi_version=3)
    assert data["inputs"]["x"]["to"] == "pos:0" and data["outputs"] == {"y": {"from": "return:y"}}
    assert "passed by position: the model takes no keyword arguments there" in text


def test_constructor_arrays_are_retried_as_numpy(tmp_path):
    pytest.importorskip("numpy")
    # like GPy's GPRegression(X, Y): the constructor reads X.shape
    path = write(tmp_path, "gp.py", """
        class GP:
            def __init__(self, X, Y):
                self.n = X.shape[0]
                self.mean = float(Y.mean())

            def predict(self, Xnew):
                return {"mu": self.mean + 0.0 * float(Xnew[0])}
    """)
    data, text = probe(f"{path}:GP", tmp_path, call="predict", starts={"X": [1.0, 2.0], "Y": [3.0, 5.0], "Xnew": [0.5]},
                       fmi_version=3)
    assert data["parameters"]["X"]["numpy"] is True and data["parameters"]["Y"]["numpy"] is True
    assert data["outputs"] == {"mu": {"from": "return:mu"}}
    assert "passed as numpy arrays: building the model failed with lists" in text


def test_reset_is_called_when_the_step_needs_it(tmp_path):
    # like gymnasium's environments: step() before reset() raises
    path = write(tmp_path, "env.py", """
        class Env:
            def __init__(self, g=9.81):
                self.state = None
                self.g = g

            def reset(self, seed=None):
                self.state = 0.0
                return self.state, {}

            def step(self, action=0.5):
                if self.state is None:
                    raise RuntimeError("Cannot call env.step() before calling env.reset()")
                self.state += action
                return self.state, 1.0, False, False, {}
    """)
    data, text = probe(f"{path}:Env", tmp_path)
    assert data["model"]["setup"] == ["reset"] and "y0" in data["outputs"]
    assert "reset() before the first step" in text


def test_a_factory_of_a_torch_like_module_reads_forward(tmp_path):
    # like timm's create_model(...)(x): __call__(*args, **kwargs) runs forward(x)
    path = write(tmp_path, "nets.py", """
        class Module:
            def __call__(self, *args, **kwargs):
                return self.forward(*args, **kwargs)

        class Net(Module):
            def __init__(self, scale):
                self.scale = scale

            def forward(self, x):
                return [self.scale * v for v in x]

        def create_model(model_name="net", scale=2.0):
            return Net(scale)
    """)
    data, _ = probe(f"{path}:create_model", tmp_path, starts={"x": [1.0, 2.0]}, fmi_version=3)
    assert data["model"]["call"] == "__call__" and data["inputs"]["x"]["dimensions"] == [2]


def test_a_classmethod_builds_the_object_when_the_constructor_cant(tmp_path):
    # like stable-baselines3's PPO(policy, env): a saved agent comes from PPO.load(path)
    (tmp_path / "agent.txt").write_text("0.5")
    path = write(tmp_path, "agents.py", """
        class PPO:
            def __init__(self, policy, env, learning_rate=3e-4):
                self.k = learning_rate

            @classmethod
            def load(cls, path, device="auto"):
                agent = cls("MlpPolicy", None)
                agent.k = float(open(path).read())
                return agent

            @classmethod
            def describe(cls):
                return "ppo"

            def predict(self, observation=1.0):
                return self.k * observation
    """)
    data, text = probe(f"{path}:PPO", tmp_path, starts={"path": {"python": "'agent.txt'"}, "observation": 2.0})
    assert data["model"]["create"] == "load" and data["model"]["call"] == "predict"
    assert "PPO.load(...) builds it: path not arguments of PPO()" in text   # observation: predict's


def test_the_step_methods_own_names_dont_call_for_a_classmethod(tmp_path):
    # like CFTOC: --start solve_x0 is the step method's copy of the constructor's x0, not a sign __init__ can't build it
    path = write(tmp_path, "ctrl.py", """
        class Ctrl:
            def __init__(self, x0, N=5):
                self.x0, self.N = x0, N

            @classmethod
            def from_dict(cls, d, **kwargs):
                return cls(**d)

            def solve(self, x0):
                return {"u": float(sum(x0)) + self.x0[0]}
    """)
    data, _ = probe(f"{path}:Ctrl", tmp_path, call="solve", fmi_version=3,
                    starts={"x0": [1.0], "solve_x0": [2.0, 3.0]})
    assert "create" not in data["model"] and data["outputs"] == {"u": {"from": "return:u"}}




def test_results_nested_in_returned_objects(tmp_path):
    # like simdkalman's smooth(): result.observations.mean
    path = write(tmp_path, "kf.py", """
        class Gaussian:
            def __init__(self, mean):
                self.mean = mean

        class Result:
            def __init__(self, data):
                self.observations = Gaussian([2 * x for x in data])

        def smooth(data=[1.0, 2.0]):
            return Result(data)
    """)
    data, _ = probe(f"{path}:smooth", tmp_path, fmi_version=3)
    assert data["outputs"]["observations.mean"] == {"dimensions": [2]}   # from = "return:<name>": the default


def test_columns_of_a_returned_table(tmp_path):
    pytest.importorskip("pandas")
    # like neuralforecast's predict(): one row per forecast step, the forecast in a column
    path = write(tmp_path, "fc.py", """
        import pandas

        def predict(h=3):
            return pandas.DataFrame({"unique_id": ["s"] * h, "NHITS": [float(i) for i in range(h)]})
    """)
    data, _ = probe(f"{path}:predict", tmp_path, fmi_version=3)
    assert data["outputs"] == {"NHITS": {"dimensions": [3]}}


def test_a_function_made_at_runtime_gets_its_own_arguments(tmp_path):
    # like sympy's lambdify(args, expr): a generated function of x and y
    path = write(tmp_path, "gen.py", """
        def make(power=2.0):
            def generated(x, y):
                return x ** power + y
            return generated
    """)
    data, _ = probe(f"{path}:make", tmp_path, call="__call__", starts={"x": 3.0, "y": 1.0})
    assert data["inputs"] == {"x": {"start": 3.0}, "y": {"start": 1.0}}
    assert data["outputs"] == {"result": {"from": "return"}}   # not y: that's an input


def test_attributes_the_object_makes_up_are_outputs_when_the_step_changes_them(tmp_path):
    # like tellurium's RoadRunner: rr.S1 for each species, through __getattr__
    path = write(tmp_path, "rr.py", """
        class RoadRunner:
            def __init__(self):
                self._species = {"S1": 10.0, "S2": 0.0}

            def __getattr__(self, name):
                try:
                    return self.__dict__["_species"][name]
                except KeyError:
                    raise AttributeError(name) from None

            def __dir__(self):
                return [*super().__dir__(), *self._species]

            def oneStep(self, currentTime=0.0, stepSize=0.1):
                moved = 0.1 * self._species["S1"]
                self._species["S1"] -= moved
                self._species["S2"] += moved
                return currentTime + stepSize
    """)
    data, _ = probe(f"{path}:RoadRunner", tmp_path, call="oneStep")
    assert data["outputs"]["S1"] == {} and data["outputs"]["S2"] == {}
    assert "init_call" not in data["model"]   # the step changes the object


def test_a_function_writing_onto_its_argument_suggests_the_object_form(tmp_path):
    # like pandapower's runpp(net): returns None, the results are in net's tables
    path = write(tmp_path, "pp.py", """
        class Net:
            pass

        def example():
            return Net()

        def runpp(net):
            net.res_bus = [1.0]
    """)
    _, text = probe(f"{path}:runpp", tmp_path, starts={"net": {"call": "pp:example()"}})
    assert 'make net the model: fmugen init pp:example --call "pp.py:runpp(self)" --probe' in text


def test_a_const_shared_with_the_step_method_goes_to_the_constructor(tmp_path):
    # like CFTOC: --const x0 is the constructor's, --start calculate_optimal_actuation_x0 the step method's
    path = write(tmp_path, "ctrl2.py", """
        class Ctrl:
            def __init__(self, x0, N=5):
                self.x0, self.N = x0, N

            def solve(self, x0, Xm):
                return {"u": float(sum(x0)) + self.x0[0] + Xm[0]}
    """)
    data, text = probe(f"{path}:Ctrl", tmp_path, call="solve", fmi_version=3,
                       starts={"x0": {"call": "builtins:list([1.0])"}, "solve_x0": [2.0], "Xm": [3.0]})
    assert data["model"]["constants"] == {"x0": {"call": "builtins:list([1.0])"}}
    assert "call_constants" not in data["model"]
    assert data["inputs"]["solve_x0"] == {"dimensions": [1], "start": [2.0], "to": "arg:x0"}


def test_an_unclear_step_method_is_asked_in_a_terminal(tmp_path, monkeypatch, capsys):
    path = write(tmp_path, "soil.py", """
        class Soil:
            def theta(self, h):
                return 0.4 - 0.001 * h

            def k(self, h):
                return 1e-5 / (1 + h)
    """)
    import sys
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True, raising=False)
    monkeypatch.setattr(sys.stderr, "isatty", lambda: True, raising=False)
    answers = iter(["7", "2"])   # out of range, then theta
    monkeypatch.setattr("builtins.input", lambda: next(answers))
    data, text = probe(f"{path}:Soil", tmp_path, starts={"h": 1.0})
    err = capsys.readouterr().err
    assert "  1. k\n  2. theta" in err and "not one of the candidates" in err
    assert data["model"]["call"] == "theta" and "theta() chosen at the prompt" in text


def test_ranking_set_up_names_and_variants():
    from fmugen.interface import _rank_methods

    class Fitter:             # like lifelines' KaplanMeierFitter: fit runs it; no predict
        def fit(self, durations): ...
        def fit_left_censoring(self, durations): ...
        def cumulative_density_at_times(self, times): ...
        def plot(self): ...

    class Solver:             # like osqp's OSQP: update changes settings, solve runs it
        def setup(self, P): ...
        def update(self, q): ...
        def solve(self): ...

    class Finder:             # like timezonefinder and radioactivedecay: the plain name, not its variants
        def certain_timezone_at(self, lng, lat): ...
        def timezone_at(self, lng, lat): ...
        def cumulative_decays(self, t): ...
        def decay(self, t): ...

    assert _rank_methods(Fitter, {}, Fitter.__init__)[:2] == ["fit", "cumulative_density_at_times"]
    assert _rank_methods(Fitter, {}, Fitter.__init__)[-1] == "plot"
    assert _rank_methods(Solver, {}, Solver.__init__)[0] == "solve"
    assert _rank_methods(Solver, {}, Solver.__init__)[-1] == "update"
    assert _rank_methods(Finder, {}, Finder.__init__)[:2] == ["decay", "timezone_at"]


def test_update_is_the_step_only_without_a_run_like_method():
    from fmugen.interface import _pick_method

    class Solver:             # like osqp's OSQP and simdkalman's KalmanFilter: update sets it up, solve runs it
        def setup(self, P): ...
        def update(self, q): ...
        def solve(self): ...

    class Filter:             # like filterpy's GHFilter: update is the step
        def update(self, z): ...
        def batch_filter(self, data): ...

    assert _pick_method(Solver)[0] == "solve"
    assert _pick_method(Filter)[0] == "update"


def test_ranking_run_like_prefixes_and_clean_up_names():
    from fmugen.interface import _rank_methods

    class Zone:               # like RC_BuildingSimulator's Zone: solve_energy runs it, calc_* are its parts
        def calc_energy_demand(self, t): ...
        def calc_t_air(self, t): ...
        def solve_energy(self, t): ...

    class Sim:                # like pyswmm's Simulation: close is clean-up, never the step
        def close(self): ...
        def report(self): ...

    assert _rank_methods(Zone, {}, Zone.__init__)[0] == "solve_energy"
    assert _rank_methods(Sim, {}, Sim.__init__) == ["report", "close"]


def test_ranking_a_partial_set_of_start_names_points_at_the_step():
    from fmugen.interface import _rank_methods

    class Model:              # --start a only: solve takes it, though b is left to be made up
        def summary(self): ...
        def plot(self): ...
        def solve(self, a, b): ...

    assert _rank_methods(Model, {"a": 1.0}, Model.__init__)[0] == "solve"


def test_a_documented_array_without_default_is_made_up_as_ones(tmp_path):
    # like ahrs' Madgwick.updateIMU(q, gyr, acc): numpydoc says q is an array of shape (4,)
    path = write(tmp_path, "filt.py", '''
        import numpy as np

        class Filter:
            def update(self, q, gyr):
                """One step.

                Parameters
                ----------
                q : numpy.ndarray, shape (4,)
                    Quaternion.
                gyr : array_like
                    Angular rates.
                """
                q, gyr = np.asarray(q), np.asarray(gyr)
                if q.shape != (4,):
                    raise TypeError("q must be given as an array")
                return float(q.sum() + gyr.sum())
    ''')
    data, text = probe(f"{path}:Filter", tmp_path)
    assert data["inputs"]["q"] == {"dimensions": [4], "start": [1.0] * 4, "numpy": True}
    assert data["inputs"]["gyr"]["dimensions"] == [3]
    assert data["outputs"]


def test_made_up_inputs_start_at_one_then_are_retried_as_arrays(tmp_path):
    # like blackscholes (S must be > 0) and scikit-learn's predict (X must be 2-D); nothing documented
    path = write(tmp_path, "pricing.py", """
        import numpy as np

        class Call:
            def __init__(self, S):
                assert S > 0, "S needs to be larger than 0"
                self.S = S

            def predict(self, X):
                X = np.asarray(X)
                if X.ndim != 2:
                    raise ValueError("Expected 2D array, got scalar array instead")
                return float(self.S * X.sum())
    """)
    data, text = probe(f"{path}:Call", tmp_path)
    assert data["parameters"]["S"]["start"] == 1.0 and "made up, as" not in text.split("[inputs]")[0]
    assert data["inputs"]["X"]["dimensions"] == [1, 1] and "arrays of ones, shape (1, 1)" in text
    assert data["outputs"]


def test_made_up_parameters_are_retried_as_zero_when_the_step_fails(tmp_path):
    # like quantecon's LinearStateSpace: A = 1.0 builds, but the step solves (1 - A) x = 1
    path = write(tmp_path, "lss.py", """
        class LSS:
            def __init__(self, A):
                self.A = A

            def stationary(self):
                return 1.0 / (1.0 - self.A)
    """)
    data, text = probe(f"{path}:LSS", tmp_path, call="stationary")
    assert data["parameters"]["A"]["start"] == 0.0 and "made up, as 0.0 (the step failed with 1.0)" in text
    assert data["outputs"]


def test_made_up_parameters_are_not_retried_as_zero_when_building_fails_then(tmp_path):
    # like pedon's Genuchten: 1.0 everywhere divides by theta_s - theta_r = 0 in the step, and n = 0.0
    # fails the constructor; the parameters stay as made up, not changed by the constructor's retries
    path = write(tmp_path, "soil.py", """
        class Soil:
            def __init__(self, theta_r, theta_s, n):
                self.m = 1 - 1 / n
                self.theta_r, self.theta_s = theta_r, theta_s

            def theta(self, h):
                return self.theta_r + (self.theta_s - self.theta_r) / (1 + h) ** self.m

            def se(self, theta):
                return (theta - self.theta_r) * (1.0 / (self.theta_s - self.theta_r))
    """)
    data, text = probe(f"{path}:Soil", tmp_path, call="se")
    assert all(data["parameters"][n] == {"start": 1.0} for n in ("theta_r", "theta_s", "n"))
    assert "made up" not in text.partition("[parameters]")[2].partition("[inputs]")[0]


def test_a_set_up_the_step_needs_is_found(tmp_path):
    # like osqp's OSQP: solve() fails until setup() makes the solver; setup takes the problem
    path = write(tmp_path, "qp.py", """
        class _Solver:
            def __init__(self, P, q):
                self.x = q / P

            def solve(self):
                return self.x

        class QP:
            def __init__(self):
                self._solver = None

            def setup(self, P, q):
                self._solver = _Solver(P, q)

            def update(self, q):
                self._solver.x = q

            def solve(self):
                return self._solver.solve()
    """)
    data, text = probe(f"{path}:QP", tmp_path, call="solve")
    assert data["model"]["setup"] == ["setup(1.0, 1.0)"] and "made up as 1.0: check them" in text
    assert data["outputs"]


def test_a_set_up_the_error_names_is_found(tmp_path):
    path = write(tmp_path, "dev.py", """
        class Device:
            def __init__(self):
                self.ready = False

            def connect(self):
                self.ready = True

            def read(self):
                if not self.ready:
                    raise RuntimeError("call connect() before read()")
                return 1.5
    """)
    data, text = probe(f"{path}:Device", tmp_path, call="read")
    assert data["model"]["setup"] == ["connect()"] and data["outputs"]


def test_training_on_made_up_data_is_only_for_the_probe(tmp_path):
    # like scikit-learn: predict before fit; the probe fits made-up data to find the outputs, the config
    # leaves fit() to the user's data
    path = write(tmp_path, "reg.py", """
        class Regressor:
            def __init__(self):
                self.coef_ = None

            def fit(self, X, y):
                self.coef_ = 2.0

            def fit_weighted(self, X, y):
                self.coef_ = 3.0

            def predict(self, X):
                return self.coef_ * X
    """)
    data, text = probe(f"{path}:Regressor", tmp_path, call="predict")
    assert "setup" not in data["model"] and '# setup = ["fit(...)"]  # predict() needs fit() first' in text
    assert data["outputs"] and "probe call failed" not in text


def test_a_method_the_error_names_otherwise_is_no_set_up(tmp_path):
    # like torch modules: __call__ runs forward, whose error names it; float() is a method too
    path = write(tmp_path, "net.py", """
        class Net:
            def float(self):
                return self

            def forward(self, x, graphs):
                return x

            def __call__(self, x):
                return self.forward(x)
    """)
    data, text = probe(f"{path}:Net", tmp_path)
    assert "setup" not in data["model"] and "seems to need" not in text


def test_docstring_array_shapes():
    from fmugen.interface import docstring_arrays

    def f(q, X, x, opts):
        """f.

        Parameters
        ----------
        q : array_like, shape (4,)
            A quaternion.
        X : ndarray of shape (n_samples, n_features)
            Data.
        x : array_like
            Values.
        opts : dict
            A list of options.
        """
    assert docstring_arrays(f) == {"q": [4], "X": [3, 3], "x": None}


def test_a_documented_array_that_takes_a_number_stays_a_number(tmp_path):
    # like gsw's rho(SA, t, p): array_like in the docstring, but numbers work, and give numbers
    path = write(tmp_path, "sea.py", '''
        def rho(SA, CT):
            """Density.

            Parameters
            ----------
            SA : array_like
                Absolute salinity.
            CT : array_like
                Conservative temperature.
            """
            return 1000.0 + 0.8 * SA - 0.2 * CT
    ''')
    data, text = probe(f"{path}:rho", tmp_path)
    assert data["inputs"]["SA"] == {"start": 1.0} and data["inputs"]["CT"] == {"start": 1.0}


def test_a_fitted_model_gets_its_number_of_features(tmp_path):
    # like catboost and tabnet: fitted by --setup on 4 features, predict(X) needs X with 4 columns
    path = write(tmp_path, "reg.py", """
        import numpy as np

        class Regressor:
            def fit(self, X, y):
                self.n_features_in_ = np.asarray(X).shape[1]

            def predict(self, X):
                X = np.asarray(X, dtype=float)
                if X.ndim != 2 or X.shape[1] != self.n_features_in_:
                    raise ValueError(f"Feature count mismatch: expecting {self.n_features_in_} features")
                return X.sum(axis=1)
    """)
    data, text = probe(f"{path}:Regressor", tmp_path, call="predict",
                       setup=["fit(numpy.ones((5, 4)), numpy.ones(5))"])
    assert data["inputs"]["X"]["dimensions"] == [1, 4] and data["outputs"]
