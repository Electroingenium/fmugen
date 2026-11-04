"""init infers options the 2026-11-04 open models needed by hand: which of equally fitting methods steps (by the
package's own calls), a classmethod that loads files, a class whose constructor computes the result, a factory's
run method, a dict passed as pandas or to the constructor, result getters after a status, and generate()."""
import textwrap

from fmugen.builder import init, isolated_imports


def write(tmp_path, name, source):
    (tmp_path / name).write_text(textwrap.dedent(source))
    return tmp_path / name


def run(target, tmp_path, **kwargs):
    with isolated_imports():
        output, data = init(target, output=tmp_path / "fmugen.toml", force=True, **kwargs)
    return data, output.read_text()


def test_equally_fitting_methods_go_by_how_often_the_package_calls_them(tmp_path):
    # like thermopack's cubic: guess_phase and two_phase_tpflash both take (temp, press, z)
    path = write(tmp_path, "flash.py", '''
        class Eos:
            def guess_phase(self, temp, press):
                return 1

            def two_phase_tpflash(self, temp, press):
                return {"beta": temp / press}


        def _demo(eos):
            eos.two_phase_tpflash(1.0, 2.0)
            eos.two_phase_tpflash(3.0, 4.0)
    ''')
    data, _ = run(f"{path}:Eos", tmp_path, starts={"temp": 250.0, "press": 2e6})
    assert data["model"]["call"] == "two_phase_tpflash"


def test_an_inherited_method_the_package_calls_far_more_beats_one_the_class_defines():
    from fmugen import interface

    class Vector:             # like skyfield's VectorFunction.at(t), inherited by EarthSatellite
        def at(self, t): ...

    class Satellite(Vector):
        def position_error(self, t): ...

    saved = interface._package_calls
    interface._package_calls = lambda cls: {"at": 283}
    try:
        assert interface._rank_methods(Satellite, {"t": 1.0}, Satellite.__init__)[0] == "at"
        interface._package_calls = lambda cls: {"at": 2, "position_error": 1}
        assert interface._rank_methods(Satellite, {"t": 1.0}, Satellite.__init__)[0] == "position_error"
    finally:
        interface._package_calls = saved


def test_a_classmethod_that_loads_an_existing_folder_builds_it(tmp_path):
    # like flopy's MFSimulation: the constructor makes an empty simulation, load() reads the one in sim_ws
    (tmp_path / "ws").mkdir()
    path = write(tmp_path, "sim.py", '''
        import os


        class Simulation:
            def __init__(self, sim_ws="."):
                self.files = 0

            @classmethod
            def load(cls, sim_ws=".", strict=True):
                sim = cls(sim_ws)
                sim.files = len(os.listdir(sim_ws)) + 1
                return sim

            def run_simulation(self, silent=False):
                return {"heads": float(self.files)}
    ''')
    data, text = run(f"{path}:Simulation", tmp_path, starts={"sim_ws": str(tmp_path / "ws")}, probe=True)
    assert data["model"]["create"] == "load" and data["model"]["call"] == "run_simulation"
    assert "a path that exists" in text


def test_a_class_whose_methods_need_more_is_built_every_step(tmp_path):
    # like geopy's geodesic(a, b): the properties are the distance; destination(point, bearing) is for other points
    path = write(tmp_path, "dist.py", '''
        class Distance:
            def __init__(self, a, b):
                self.km = abs(b - a) * 111.0

            @property
            def miles(self):
                return self.km / 1.609344

            def destination(self, point, bearing):
                return point + bearing
    ''')
    data, text = run(f"{path}:Distance", tmp_path, starts={"a": 1.0, "b": 3.0}, probe=True)
    assert data["model"]["kind"] == "function" and "call" not in data["model"]
    assert {"km", "miles"} <= set(data["outputs"]) and "built with the inputs every step" in text


def test_a_factory_result_runs_by_its_optimize(tmp_path):
    # like cobra's load_model('textbook').optimize()
    path = write(tmp_path, "fba.py", '''
        class Solution:
            def __init__(self, value):
                self.objective_value = value


        class Model:
            def __init__(self, scale):
                self.scale = scale

            def optimize(self, objective_sense=None):
                return Solution(0.87 * self.scale)


        def load_model(model_id, scale=1.0):
            return Model(scale)
    ''')
    data, text = run(f"{path}:load_model", tmp_path, starts={"model_id": "textbook"}, probe=True)
    assert data["model"]["call"] == "optimize" and "objective_value" in str(data["outputs"])
    assert "its optimize() runs it" in text


def test_a_dict_is_passed_as_the_pandas_series_the_error_asks_for(tmp_path):
    # like pylife's WoehlerCurve(pandas_obj), a pandas accessor
    path = write(tmp_path, "woehler.py", '''
        import pandas


        class Curve:
            def __init__(self, pandas_obj):
                if not isinstance(pandas_obj, pandas.Series):
                    raise AttributeError("An accessor object needs to be either a pandas.Series or a pandas.DataFrame")
                self.k = float(pandas_obj["k"])

            def cycles(self, load):
                return 1e6 * load ** -self.k
    ''')
    data, text = run(f"{path}:Curve", tmp_path, starts={"pandas_obj": {"k": 2.0}, "load": 3.0}, probe=True)
    assert data["model"]["constants"]["pandas_obj"] == {"call": "pandas:Series({'k': 2.0})"}
    assert data["model"]["call"] == "cycles" and "passed as a pandas Series" in text


def test_a_dict_the_step_refuses_goes_to_the_constructor(tmp_path):
    # like rrmpg's GR4J(params=None).simulate(prec, params=None): the step wants a structured array
    path = write(tmp_path, "gr4j.py", '''
        class GR4J:
            def __init__(self, params=None):
                self.x1 = params["x1"] if params else 1.0

            def simulate(self, prec, params=None):
                if params is not None:
                    params.dtype
                return prec * self.x1
    ''')
    data, text = run(f"{path}:GR4J", tmp_path, starts={"params": {"x1": 3.0}, "prec": 2.0}, probe=True)
    assert data["model"]["constants"]["params"] == {"python": "{'x1': 3.0}"}
    assert "passed to the constructor" in text


def test_a_step_returning_a_status_is_read_through_its_result_methods(tmp_path):
    # like vosk's KaldiRecognizer: AcceptWaveform says whether an utterance ended, FinalResult gives the text
    path = write(tmp_path, "asr.py", '''
        class Recognizer:
            def __init__(self, rate):
                self.text = ""

            def AcceptWaveform(self, data):
                self.text = data.upper()
                return 0

            def FinalResult(self):
                return self.text
    ''')
    data, _ = run(f"{path}:Recognizer", tmp_path, starts={"rate": 16000, "data": "and so"}, probe=True)
    assert data["outputs"]["FinalResult"] == {"from": "get:FinalResult", "type": "String"}


def test_a_generative_model_steps_with_generate_when_forward_fails(tmp_path):
    # like transformers' VisionEncoderDecoderModel (TrOCR): forward() needs decoder ids that generate() makes
    path = write(tmp_path, "ocr.py", '''
        class Model:
            @classmethod
            def from_pretrained(cls, name, **kwargs):
                return cls()

            def forward(self, pixel_values=None, decoder_input_ids=None):
                if decoder_input_ids is None:
                    raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")
                return pixel_values

            def generate(self, inputs=None, **kwargs):
                return [len(kwargs["pixel_values"]), 7]
    ''')
    data, text = run(f"{path}:Model", tmp_path, starts={"name": "trocr", "pixel_values": [0.5, 0.25]}, probe=True)
    assert data["model"]["create"] == "from_pretrained" and data["model"]["call"] == "generate"
    assert "pixel_values" in data["inputs"]
