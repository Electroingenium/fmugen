"""init --probe on model shapes found in published libraries: C built-ins without a signature, steps that
are functions taking the object, histories that grow, results nested in objects or tables, functions made
at runtime, attributes an object makes up, and functions writing their results onto an argument."""
import textwrap

import pytest

from fmugen.builder import init, isolated_imports


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


def test_a_history_that_grows_each_step_gives_its_last_entry(tmp_path):
    # like seirsplus' numS: one more entry per run_epoch()
    path = write(tmp_path, "epi.py", """
        class Model:
            def __init__(self, N=100.0):
                self.numS = [N]

            def run_epoch(self, runtime=1.0):
                self.numS = self.numS + [self.numS[-1] * 0.9]
    """)
    data, text = probe(f"{path}:Model", tmp_path, call="run_epoch", fmi_version=3)
    assert data["outputs"]["numS"] == {"from": "attr:numS.-1"}
    assert "its last entry" in text


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
