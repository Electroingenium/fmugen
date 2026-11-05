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
    data, _text = run(f"{path}:Model", tmp_path, starts={"name": "trocr", "pixel_values": [0.5, 0.25]}, probe=True)
    assert data["model"]["create"] == "from_pretrained" and data["model"]["call"] == "generate"
    assert "pixel_values" in data["inputs"]


def test_a_data_argument_left_out_is_warned_about(tmp_path, capsys):
    # like biosppy's ecg(signal=None, sampling_rate=1000.0): the probe fails without the signal
    path = write(tmp_path, "ecg.py", '''
        def ecg(signal=None, sampling_rate=1000.0, show=True):
            if signal is None:
                raise TypeError("Please specify an input signal.")
            return {"heart_rate": sum(signal) / sampling_rate}
    ''')
    run(f"{path}:ecg", tmp_path, starts={"sampling_rate": 1000.0}, probe=True)
    err = capsys.readouterr().err
    assert "signal: looks like data the model runs on, and none was given (left out: its default is None" in err
    assert "the probe failed without it (Please specify an input signal.)" in err
    assert "--const \"signal=call:numpy:loadtxt('signal.txt')\"" in err
    assert "sampling_rate: looks like data" not in err


def test_data_given_or_a_made_up_number_is_not_warned_about(tmp_path, capsys):
    path = write(tmp_path, "ecg.py", '''
        def ecg(signal=None, sampling_rate=1000.0):
            return {"n": len(signal)}


        def square(x):
            return x * x
    ''')
    run(f"{path}:ecg", tmp_path, starts={"signal": [1.0, 2.0, 3.0]}, probe=True)
    run(f"{path}:square", tmp_path, probe=True)
    assert "looks like data" not in capsys.readouterr().err


def test_the_data_of_a_model_a_factory_returns_is_warned_about(tmp_path, capsys):
    # like open_clip's create_model(...): the CLIP model it returns takes the image, which the FMU never passes
    path = write(tmp_path, "clip.py", '''
        class Clip:
            def __init__(self):
                self.context_length = 77

            def __call__(self, image=None):
                return [0.5 * v for v in image]


        def create_model(model_name):
            return Clip()
    ''')
    run(f"{path}:create_model", tmp_path, starts={"model_name": "ViT-B-32"}, probe=True)
    err = capsys.readouterr().err
    assert "image: looks like data the model runs on, and none was given (not passed: create_model() returns a " \
           "Clip, whose __call__() takes it" in err
    assert "--const \"image=call:numpy:load('image.npy')\"" in err


def test_data_templates_follow_the_type():
    from fmugen.interface import _data_template, _looks_like_data
    assert _data_template("pixel_values", "", True) == "--const \"pixel_values=call:torch:load('pixel_values.pt')\""
    assert _data_template("img", "PIL.Image.Image", False) == "--const \"img=call:PIL.Image:open('img.png')\""
    assert _data_template("df", "pandas.DataFrame", False) == "--const \"df=call:pandas:read_csv('df.csv')\""
    assert _data_template("audio_path", "", False) == "--start audio_path=path/to/file"
    assert _data_template("text", "str", False) == '--start "text=your text"'
    assert _looks_like_data("pixel_values") and _looks_like_data("inputFrame") and _looks_like_data("q", "torch.Tensor")
    assert not _looks_like_data("sampling_rate") and not _looks_like_data("data_format") and not _looks_like_data("tol")
    assert not _looks_like_data("p", "array_like")   # pastas' parameters


def test_a_class_whose_step_gives_no_number_is_built_every_step(tmp_path):
    # like iapws' IAPWS97(T=..., P=...): __call__(**kwargs) recomputes and returns nothing; h, rho are attributes
    path = write(tmp_path, "steam.py", '''
        class Steam:
            def __init__(self, **kwargs):
                self.status = "ok"
                self.__call__(**kwargs)

            def __call__(self, **kwargs):
                if "T" in kwargs and "P" in kwargs:
                    self.h = 4.18 * kwargs["T"]
                    self.rho = 1000.0 / (1.0 + kwargs["P"])
    ''')
    data, text = run(f"{path}:Steam", tmp_path, starts={"T": 400.0, "P": 1.0}, probe=True)
    assert data["model"]["kind"] == "function" and {"h", "rho"} <= set(data["outputs"])
    assert "stepping it gives no numeric result" in text


def test_the_classmethod_requiring_the_most_names_builds_it(tmp_path):
    # like hapsira's Orbit: from_classical(attractor, a, ecc, ...) before heliosynchronous(attractor, a=None, ...)
    path = write(tmp_path, "orbit.py", '''
        class Orbit:
            def __init__(self, state, epoch):
                self.a = state

            @classmethod
            def from_classical(cls, attractor, a, ecc, epoch=0.0):
                return cls(a * (1 - ecc), epoch)

            @classmethod
            def heliosynchronous(cls, attractor, a=None, ecc=None):
                return cls(7000.0, 0.0)

            @classmethod
            def stationary(cls, attractor):
                return cls(42164.0, 0.0)

            def propagate(self, value):
                return {"r": self.a + value}
    ''')
    data, _ = run(f"{path}:Orbit", tmp_path, starts={"attractor": "Earth", "a": 7000.0, "ecc": 0.01, "value": 1.0},
                  probe=True)
    assert data["model"]["create"] == "from_classical" and data["model"]["call"] == "propagate"


def test_a_name_kwargs_would_swallow_goes_to_the_classmethod_that_requires_it(tmp_path):
    # like timesfm's TimesFM_2p5_200M_torch(torch_compile=True, config=None, **kwargs) and from_pretrained(repo id)
    path = write(tmp_path, "fm.py", '''
        class Forecaster:
            def __init__(self, torch_compile=True, config=None, **kwargs):
                self.scale = 1.0

            @classmethod
            def from_pretrained(cls, pretrained_model_name_or_path, *, force_download=False, **model_kwargs):
                model = cls()
                model.scale = float(len(pretrained_model_name_or_path))
                return model

            def forecast(self, horizon, inputs):
                return [self.scale * x for x in inputs[:horizon]]
    ''')
    data, _ = run(f"{path}:Forecaster", tmp_path,
                  starts={"pretrained_model_name_or_path": "org/model", "horizon": 2, "inputs": [1.0, 2.0, 3.0]},
                  probe=True)
    assert data["model"]["create"] == "from_pretrained"


def test_an_environment_is_reset_before_its_first_step(tmp_path):
    # like a gym distillation column whose step() works without reset(), from a state that isn't the start
    path = write(tmp_path, "env.py", '''
        class Space:
            pass


        class Column:
            action_space = Space()
            observation_space = Space()

            def __init__(self):
                self.level = 0.0

            def reset(self, seed=None):
                self.level = 1.0
                self.purity = 0.5
                return self.level, {}

            def step(self, action):
                self.level += action
                return self.level, 0.0, False, {}
    ''')
    data, text = run(f"{path}:Column", tmp_path, starts={"action": 0.1}, probe=True)
    assert data["model"]["setup"] == ["reset(seed=0)"] and "it's an environment" in text


def test_an_onnx_session_feed_is_made_up_from_its_inputs(tmp_path):
    # like onnxruntime's InferenceSession.run(output_names, input_feed): the inputs it declares, as zeros
    path = write(tmp_path, "session.py", '''
        import numpy


        class NodeArg:
            name, shape, type = "Input3", ["N", 1, 2, 2], "tensor(float)"


        class Session:
            def __init__(self, path_or_bytes):
                self.path = path_or_bytes

            def get_inputs(self):
                return [NodeArg()]

            def get_outputs(self):
                return []

            def run(self, output_names, input_feed, run_options=None):
                return [numpy.asarray(input_feed["Input3"]).sum(axis=(2, 3)) + 1.0]
    ''')
    data, text = run(f"{path}:Session", tmp_path, starts={"path_or_bytes": "m.onnx"}, call="run", probe=True)
    assert data["model"]["call_constants"]["input_feed"] == \
        {"call": "builtins:dict({'Input3': numpy.zeros((1, 1, 2, 2), dtype='float32')})"}
    assert "made up from the session's inputs" in text and data["outputs"]


def test_a_multi_agent_environment_gets_one_action_input_per_agent(tmp_path):
    # like PettingZoo's parallel_env: step(actions), a dict of one action per agent
    path = write(tmp_path, "agents.py", '''
        class Discrete:
            def __init__(self, n):
                self.n, self.start = n, 0


        class Env:
            possible_agents = ["agent_0", "agent_1"]

            def action_space(self, agent):
                return Discrete(5)

            def step(self, actions):
                return {agent: float(a) for agent, a in actions.items()}
    ''')
    data, _ = run(f"{path}:Env", tmp_path, call="step", probe=True)
    assert data["inputs"]["agent_0"]["to"] == "arg:actions[agent_0]" and "agent_1" in data["inputs"]


def test_an_unannotated_argument_is_built_from_the_record_class_it_is_named_after(tmp_path):
    # like simglucose's T1DPatient.step(action), whose module defines Action = namedtuple(...)
    path = write(tmp_path, "patient.py", '''
        from collections import namedtuple

        Action = namedtuple("patient_action", ["CHO", "insulin"])


        class Patient:
            def __init__(self):
                self.glucose = 150.0

            def step(self, action):
                self.glucose += action.CHO - 10.0 * action.insulin
    ''')
    data, _ = run(f"{path}:Patient", tmp_path, starts={"action.CHO": 0.0, "action.insulin": 0.02}, probe=True)
    assert data["inputs"]["action_CHO"]["to"] == "arg:action.CHO" and "action_insulin" in data["inputs"]


def test_the_runtime_rebuilds_the_record_class_an_argument_is_named_after(tmp_path, monkeypatch):
    import importlib
    import sys

    from fmugen.templates.fmugen_runtime._helpers import build_object
    write(tmp_path, "patient_rt.py", '''
        from collections import namedtuple

        Action = namedtuple("patient_action", ["CHO", "insulin"])


        def step(action):
            return action.CHO
    ''')
    monkeypatch.syspath_prepend(str(tmp_path))
    module = importlib.import_module("patient_rt")
    try:
        built = build_object(module.step, "action", {"CHO": 1.0, "insulin": 0.5})
        assert isinstance(built, module.Action) and built.insulin == 0.5
    finally:
        sys.modules.pop("patient_rt", None)


def test_an_attribute_held_outside_the_object_dict_as_an_array_is_not_a_scalar_local(tmp_path):
    # like botorch's SingleTaskGP: train_targets, a tensor torch keeps as a buffer, not in __dict__
    path = write(tmp_path, "gp.py", '''
        import numpy


        class Model:
            def __init__(self, train_y):
                object.__setattr__(self, "_buffers", {})
                self.train_targets = numpy.asarray(train_y, dtype=float)

            def __setattr__(self, name, value):
                if isinstance(value, numpy.ndarray):
                    self._buffers[name] = value
                else:
                    object.__setattr__(self, name, value)

            def __getattr__(self, name):
                buffers = object.__getattribute__(self, "_buffers")
                if name in buffers:
                    return buffers[name]
                raise AttributeError(name)

            def posterior(self, x):
                return {"mean": float(self.train_targets.mean()) * x}
    ''')
    data, _ = run(f"{path}:Model", tmp_path, starts={"train_y": [1.0, 2.0, 3.0], "x": 2.0}, call="posterior",
                  probe=True)
    assert "train_targets" not in data.get("locals", {}) and "mean" in data["outputs"]


def test_properties_a_model_adds_to_its_class_at_run_time_are_outputs(tmp_path):
    # like roadrunner, which adds a property per species (rr.S1) to its class when it loads a model
    path = write(tmp_path, "rr.py", '''
        class Runner:
            def __init__(self, model):
                self._values = {"S1": 10.0, "S2": 0.0}
                for name in self._values:
                    setattr(Runner, name, property(lambda self, n=name: self._values[n]))

            def oneStep(self, currentTime, stepSize):
                moved = 0.1 * stepSize * self._values["S1"]
                self._values["S1"] -= moved
                self._values["S2"] += moved
                return currentTime + stepSize
    ''')
    data, _ = run(f"{path}:Runner", tmp_path, starts={"model": "S1 -> S2", "currentTime": 0.0, "stepSize": 0.1},
                  call="oneStep", probe=True)
    assert {"S1", "S2"} <= set(data["outputs"])


def test_the_runtime_file_keeps_every_import_made_inside_a_function(tmp_path):
    # an optional import (numpy) inside a function of one runtime module isn't redundant with the same line in a
    # function of another: dropping it broke every numpy = true array input
    import ast

    from fmugen.builder import RUNTIME_SOURCES, _concatenate_runtime

    def local_imports(text):
        tree = ast.parse(text)
        top = {id(n) for n in tree.body}
        return sum(isinstance(n, (ast.Import, ast.ImportFrom)) and id(n) not in top
                   and not (isinstance(n, ast.ImportFrom) and (n.module or "").startswith("fmugen."))
                   for n in ast.walk(tree))

    _concatenate_runtime(tmp_path / "fmugen_runtime.py")
    expected = sum(local_imports(p.read_text(encoding="utf-8")) for p in RUNTIME_SOURCES)
    assert local_imports((tmp_path / "fmugen_runtime.py").read_text(encoding="utf-8")) == expected


def test_no_init_call_when_a_second_step_advances_a_state_the_probe_cant_compare(tmp_path):
    # like pybamm's Simulation: it can't be pickled, and step() advances a solution object, not a number
    path = write(tmp_path, "sim.py", '''
        import threading


        class Solution:
            def __init__(self, t):
                self.t = t


        class Simulation:
            def __init__(self):
                self.lock = threading.Lock()
                self.solution = Solution(0.0)

            def step(self, dt=100.0):
                self.solution = Solution(self.solution.t + dt)
                return self.solution


        class Lookup:
            def __init__(self):
                self.lock = threading.Lock()

            def step(self, x=2.0):
                return {"y": x * 3}
    ''')
    data, text = run(f"{path}:Simulation", tmp_path, probe=True, reads={"t": "return:t"})
    assert "init_call" not in data["model"]
    assert "a second step() with the same inputs changed t" in text
    data, _ = run(f"{path}:Lookup", tmp_path, probe=True)
    assert data["model"]["init_call"] is True   # a step that changes nothing still runs at initialization
