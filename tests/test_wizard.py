"""The questions of fmugen init, answered with real key presses through prompt_toolkit's pipe input."""
import textwrap

import pytest
from prompt_toolkit.application import create_app_session
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output import DummyOutput

from fmugen import wizard
from fmugen.__main__ import main

ENTER, DOWN, CTRL_C = "\r", "\x1b[B", "\x03"

MODEL = """
    def gain(x, k: float = 2.0):
        return {"y": k * x}

    def offset(x: float = 0.0):
        return {"y": x + 1.0}
"""


@pytest.fixture
def run(tmp_path, monkeypatch):
    """Run ``fmugen init`` in tmp_path as in a terminal, with the keys given."""
    (tmp_path / "plant.py").write_text(textwrap.dedent(MODEL))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(wizard, "interactive", lambda: not wizard._assume_yes)
    monkeypatch.setattr(wizard, "_output", DummyOutput)

    def run_init(argv, keys, plain=False):
        with create_pipe_input() as pipe, create_app_session(input=pipe, output=DummyOutput()):
            pipe.send_text(keys)
            main(["init", *([] if plain else ["--wizard"]), *argv])
    yield run_init
    wizard.assume_yes(False)


def test_every_question_with_its_default(run, tmp_path, capsys):
    # the model file, the first entry, FMI 2, the path, no probe, the made-up start of x, nothing else
    run([], ENTER * 4 + "n" + ENTER * 2)
    text = (tmp_path / "fmugen.toml").read_text()
    assert 'plant.py:gain"' in text and "fmi_version = 2" in text and "start = 1.0" in text
    assert "fmugen init plant.py:gain --fmi 2" in capsys.readouterr().err


def test_answers_become_options(run, tmp_path, capsys):
    # the second entry, FMI 3, the path, probe; in offset, x has a default, so no start value is asked
    run(["plant.py"], DOWN + ENTER + DOWN + ENTER + ENTER + "y" + ENTER)
    text = (tmp_path / "fmugen.toml").read_text()
    assert 'plant.py:offset"' in text and "fmi_version = 3" in text
    assert "fmugen init plant.py:offset --probe --fmi 3" in capsys.readouterr().err


def test_a_start_value_typed_is_written(run, tmp_path, capsys):
    # the prompt's text starts as 1.0: erased, then 5.5
    run(["plant.py:gain", "--fmi", "2", "-o", "fmugen.toml"], "n" + "\x7f" * 3 + "5.5" + ENTER * 2)
    assert "start = 5.5" in (tmp_path / "fmugen.toml").read_text()
    assert "--start x=5.5" in capsys.readouterr().err


def test_an_entry_can_be_written(run, tmp_path):
    (tmp_path / "plant.py").write_text("from math import sqrt\n" + textwrap.dedent(MODEL))
    # the entries, then the last choice, "Write my own": sqrt is imported, so it isn't listed
    run(["plant.py", "--fmi", "2", "-o", "fmugen.toml"], DOWN * 2 + ENTER + "offset" + ENTER + "n" + ENTER)
    assert 'plant.py:offset"' in (tmp_path / "fmugen.toml").read_text()


CLASS = """
    class Soil:
        def theta(self, h: float = 1.0):
            return 0.4 - 0.001 * h

        def k(self, h: float = 1.0):
            return 1e-5 / (1 + h)
"""


def test_step_methods_found_are_listed(run, tmp_path, capsys):
    (tmp_path / "plant.py").write_text(textwrap.dedent(CLASS))
    run(["plant.py", "--fmi", "2", "-o", "fmugen.toml"], "n" + DOWN + ENTER * 2)
    assert 'call = "theta"' in (tmp_path / "fmugen.toml").read_text()
    assert "--call theta" in capsys.readouterr().err


def test_a_step_method_can_be_written(run, tmp_path):
    (tmp_path / "plant.py").write_text(textwrap.dedent(CLASS))
    run(["plant.py", "--fmi", "2", "-o", "fmugen.toml"], "n" + DOWN * 2 + ENTER + "theta" + ENTER * 2)
    assert 'call = "theta"' in (tmp_path / "fmugen.toml").read_text()


def test_data_arguments_are_asked(run, tmp_path, capsys):
    (tmp_path / "plant.py").write_text("def f(signal=None, k: float = 1.0):\n    return {'y': k}\n")
    run(["plant.py", "--fmi", "2", "-o", "fmugen.toml"], "n" + "call:numpy:loadtxt('s.txt')" + ENTER * 2)
    assert "numpy:loadtxt('s.txt')" in (tmp_path / "fmugen.toml").read_text()
    assert "--const" in capsys.readouterr().err


NEEDS_SIGNAL = """
    def f(k: float = 1.0, signal=None):
        if signal is None:
            raise ValueError("Please specify an input signal.")
        return {"y": k * sum(signal)}
"""


def test_what_only_the_probe_finds_is_asked(run, tmp_path, capsys):
    (tmp_path / "plant.py").write_text(textwrap.dedent(NEEDS_SIGNAL))
    run(["plant.py", "--fmi", "2", "-o", "fmugen.toml"], "y" + "call:builtins:list([1.0, 2.0])" + ENTER * 2)
    assert "builtins:list([1.0, 2.0])" in (tmp_path / "fmugen.toml").read_text()
    err = capsys.readouterr().err
    assert "probe call failed" in err and "looks like data" not in err


def test_without_the_probe_it_isnt_called(run, tmp_path):
    (tmp_path / "plant.py").write_text(textwrap.dedent(NEEDS_SIGNAL))
    run(["plant.py", "--fmi", "2", "-o", "fmugen.toml"], "n" + ENTER)   # no question about signal: nothing shows it
    assert "probe" not in (tmp_path / "fmugen.toml").read_text()


def test_a_start_from_an_example_is_asked(run, tmp_path, capsys):
    (tmp_path / "plant.py").write_text('def f(x):\n    """Double.\n\n    >>> f(2.5)\n    """\n    return {"y": 2.0 * x}\n')
    run(["plant.py", "--fmi", "2", "-o", "fmugen.toml"], "n" + "\x7f" * 3 + "7.5" + ENTER * 2)
    assert "start = 7.5" in (tmp_path / "fmugen.toml").read_text()
    assert "--start x=7.5" in capsys.readouterr().err


OUTPUTS = 'def f(x: float = 1.0):\n    return {"a": 1, "b": 2, "c": 2.5}\n'


def test_outputs_are_chosen_after_the_probe(run, tmp_path, capsys):
    (tmp_path / "plant.py").write_text(OUTPUTS)
    # c unchecked; then of a and b, the ints: a is a count
    run(["plant.py", "--fmi", "2", "-o", "fmugen.toml"], "y" + ENTER + DOWN * 2 + " " + ENTER + " " + ENTER)
    text = (tmp_path / "fmugen.toml").read_text()
    assert "\nc = " not in text and 'type = "Integer"' in text.split("\na = ")[1].splitlines()[0]
    assert text.split("\nb = ")[1].startswith("{}")
    assert "--keep a --keep b --integer a" in capsys.readouterr().err


def test_outputs_arent_asked_without_the_probe(run, tmp_path):
    (tmp_path / "plant.py").write_text(OUTPUTS)
    run(["plant.py", "--fmi", "2", "-o", "fmugen.toml"], "n" + ENTER)
    assert "\nc = " in (tmp_path / "fmugen.toml").read_text()


def test_keep_and_integer_are_options(run, tmp_path):
    (tmp_path / "plant.py").write_text(OUTPUTS)
    run(["plant.py", "--probe", "--keep", "a", "--integer", "a"], "", plain=True)
    text = (tmp_path / "fmugen.toml").read_text()
    assert "\nb = " not in text and 'type = "Integer"' in text


def test_a_computed_start_value_becomes_a_constant(run, tmp_path, capsys):
    run(["plant.py:gain", "--fmi", "2", "-o", "fmugen.toml"], "n" + "\x7f" * 3 + "call:builtins:float(3)" + ENTER * 2)
    assert 'x = { call = "builtins:float(3)" }' in (tmp_path / "fmugen.toml").read_text()
    assert "--const 'x=call:builtins:float(3)'" in capsys.readouterr().err


def test_a_variable_bound_under_another_name_isnt_asked():
    data = {"inputs": {"b": {"start": 2.0, "to": "arg:x[b]"}, "c": {"start": 1.0}}}
    comments = {("inputs", "b"): "made up", ("inputs", "c"): "made up"}
    assert list(wizard.guessed_starts(data, comments)) == ["c"]


def test_building_the_object_every_step_is_a_choice_among_the_methods(monkeypatch):
    shown = {}

    def choose(kind, message, choices, **options):
        shown.update(choices=choices, options=options)
        return choices[0]

    monkeypatch.setattr(wizard, "_ask", choose)
    assert wizard._ask_method(None, ["apply"], "Fit") is None
    assert shown["choices"][1:] == ["apply", "Write my own"] and not shown["options"]
    assert wizard._ask_method("m0", [f"m{i}" for i in range(20)], "Fit") == "m0"
    assert shown["options"]["use_search_filter"]


FACTORY = """
    class Engine:
        def run(self, x: float = 1.0):
            return {"y": x}

        def stop(self):
            pass

    def make(n: int = 1) -> Engine:
        return Engine()
"""


def test_the_method_of_what_a_factory_returns_is_asked(run, tmp_path, capsys):
    (tmp_path / "plant.py").write_text(textwrap.dedent(FACTORY))
    run(["plant.py:make", "--fmi", "2", "-o", "fmugen.toml"], "n" + ENTER * 2)   # listed first: None, or run
    assert "--call" not in capsys.readouterr().err
    run(["plant.py:make", "--fmi", "2", "-o", "fmugen.toml", "--force"], "n" + DOWN + ENTER * 2)
    assert 'call = "run"' in (tmp_path / "fmugen.toml").read_text()


def test_a_local_package_is_found_in_the_current_folder(run, tmp_path):
    (tmp_path / "localpkg").mkdir()
    (tmp_path / "localpkg" / "__init__.py").write_text("")
    (tmp_path / "localpkg" / "core.py").write_text("def f(x: float = 1.0):\n    return {'y': x}\n")
    run(["localpkg.core:f"], "", plain=True)
    assert 'entry = "localpkg.core:f"' in (tmp_path / "fmugen.toml").read_text()


def test_installed_modules_stay_loaded_with_their_folder_added():
    import sys
    import sysconfig

    from fmugen.builder import isolated_imports
    purelib = sysconfig.get_path("purelib")
    with isolated_imports():
        sys.path.append(purelib)
        sys.modules["fake_installed"] = type(textwrap)("fake_installed")
        sys.modules["fake_installed"].__file__ = str(__import__("pathlib").Path(purelib) / "fake_installed.py")
    assert "fake_installed" in sys.modules
    del sys.modules["fake_installed"]


def test_a_config_with_no_variables_warns():
    from fmugen.builder import _init_warnings
    assert any("no FMU variables" in w for w in _init_warnings({"model": {}}, {}, True))


def test_what_an_object_holds_inside_starts_unticked(run, tmp_path, monkeypatch):
    seen = {}
    real = wizard._ask

    def spy(kind, message, **options):
        if kind == "checkbox":
            seen[message] = {c.value: c.checked for c in options["choices"]}
        return real(kind, message, **options)

    monkeypatch.setattr(wizard, "_ask", spy)
    args = type("A", (), {"keep": [], "integer": []})()
    data = {"outputs": {"y": {"from": "return"}, "inner.x": {"from": "attr:inner.x"}, "r.x": {"from": "return:r.x"}}}
    with create_pipe_input() as pipe, create_app_session(input=pipe, output=DummyOutput()):
        pipe.send_text(ENTER)
        wizard._ask_outputs(args, data, {}, True)
    assert next(iter(seen.values())) == {"y": True, "inner.x": False, "r.x": True}
    assert args.keep == ["y", "r.x"]


NONE_RESULT ='def f(x: float = 1.0):\n    return {"a": 1.5, "b": 2.5, "maybe": None}\n'


def test_outputs_left_commented_out_can_be_chosen(run, tmp_path, capsys):
    (tmp_path / "plant.py").write_text(NONE_RESULT)
    # a, b, then maybe, unticked: ticked; its type is unknown: the second, Integer
    run(["plant.py", "--fmi", "2", "-o", "fmugen.toml"], "y" + ENTER + DOWN * 2 + " " + ENTER + DOWN + ENTER)
    text = (tmp_path / "fmugen.toml").read_text()
    assert '\nmaybe = { from = "return:maybe", type = "Integer" }' in text
    err = capsys.readouterr().err
    assert "--keep a --keep b --keep maybe=Integer" in err and "None in the probe" not in err


def test_outputs_left_commented_out_and_not_chosen_are_gone(run, tmp_path, capsys):
    (tmp_path / "plant.py").write_text(NONE_RESULT)
    run(["plant.py", "--fmi", "2", "-o", "fmugen.toml"], "y" + ENTER * 2)
    assert "maybe" not in (tmp_path / "fmugen.toml").read_text()
    assert "maybe" not in capsys.readouterr().err


def test_without_the_flag_they_stay_commented_out(run, tmp_path):
    (tmp_path / "plant.py").write_text(NONE_RESULT)
    run(["plant.py", "--probe"], "", plain=True)
    assert "# maybe = " in (tmp_path / "fmugen.toml").read_text()


def test_keep_asks_for_what_a_commented_output_needs(run, tmp_path, capsys):
    (tmp_path / "plant.py").write_text(NONE_RESULT)
    with pytest.raises(SystemExit) as stop:
        run(["plant.py", "--probe", "--keep", "maybe"], "", plain=True)
    assert stop.value.code == 2 and "needs a type" in capsys.readouterr().err


def test_keep_names_an_output_found(run, tmp_path, capsys):
    (tmp_path / "plant.py").write_text(OUTPUTS)
    with pytest.raises(SystemExit) as stop:
        run(["plant.py", "--probe", "--keep", "nope"], "", plain=True)
    assert stop.value.code == 2 and "nope: no such output" in capsys.readouterr().err
    assert not (tmp_path / "fmugen.toml").exists()


def test_a_data_argument_can_be_skipped(run, tmp_path, capsys):
    (tmp_path / "plant.py").write_text("def f(signal=None, k: float = 1.0):\n    return {'y': k}\n")
    run(["plant.py", "--fmi", "2", "-o", "fmugen.toml"], "n" + ENTER * 2)
    assert "looks like data" in capsys.readouterr().err


def test_options_given_are_not_asked(run, tmp_path):
    run(["plant.py:offset", "-o", "fmugen.toml", "--fmi", "3", "--probe"], ENTER)
    assert "fmi_version = 3" in (tmp_path / "fmugen.toml").read_text()


def test_an_existing_config_is_kept_unless_confirmed(run, tmp_path):
    (tmp_path / "fmugen.toml").write_text("mine")
    run(["plant.py:offset", "--fmi", "3", "--probe"], ENTER + "y" + ENTER)
    assert "fmi_version = 3" in (tmp_path / "fmugen.toml").read_text()


def test_ctrl_c_writes_nothing(run, tmp_path, capsys):
    with pytest.raises(SystemExit) as stop:
        run(["plant.py"], CTRL_C)
    assert stop.value.code == 130 and "nothing written" in capsys.readouterr().err
    assert not (tmp_path / "fmugen.toml").exists()


def test_without_the_flag_nothing_is_asked(run, tmp_path):
    run(["plant.py:offset"], "", plain=True)
    assert 'plant.py:offset"' in (tmp_path / "fmugen.toml").read_text()


def test_no_model_outside_a_terminal(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit) as stop:
        main(["init"])
    assert stop.value.code == 2 and "no model given" in capsys.readouterr().err


def test_entries_read_without_importing(tmp_path):
    path = tmp_path / "m.py"
    path.write_text("raise SystemExit\nclass Mode(Enum): ...\nclass Bad(ValueError): ...\nclass Oops(MyError): ...\n"
                    "def _private(): ...\ndef f(): ...\nclass C: ...\n")
    assert wizard.entries_in_file(path) == ["f", "C"]


def test_a_rename_waits_while_the_path_is_held_open(tmp_path):
    from fmugen.builder import _retrying
    calls = []

    def held_twice(source, target):
        calls.append(source)
        if len(calls) < 3:
            raise PermissionError
        return "moved"

    assert _retrying(held_twice, "a", "b") == "moved" and len(calls) == 3
    with pytest.raises(PermissionError):
        _retrying(lambda s, t: (_ for _ in ()).throw(PermissionError()), "a", "b", tries=2)


def test_the_wizard_needs_a_terminal(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit) as stop:
        main(["init", "--wizard", "x.py"])
    assert stop.value.code == 2 and "needs a terminal" in capsys.readouterr().err


def test_a_wrong_model_is_asked_again(run, tmp_path, capsys):
    (tmp_path / "broken.py").write_text("raise ValueError('nope')\n")
    # broken.py, FMI 2, the path, no probe: reading it fails. Then plant.py, its first entry, its path
    run([], ENTER + ENTER * 2 + "n" + DOWN + ENTER + ENTER * 2 + ENTER * 2)
    assert 'plant.py:gain"' in (tmp_path / "fmugen.toml").read_text()
    assert "nope" in capsys.readouterr().err


def test_a_wrong_option_given_stops(run, tmp_path, capsys):
    with pytest.raises(SystemExit) as stop:
        (tmp_path / "plant.py").write_text(textwrap.dedent(CLASS))
        run(["plant.py", "--fmi", "2", "-o", "fmugen.toml", "--call", "nope"], "n")
    assert stop.value.code == 2


def test_a_wrong_step_method_is_asked_again(run, tmp_path, capsys):
    (tmp_path / "plant.py").write_text(textwrap.dedent(CLASS))
    run(["plant.py", "--fmi", "2", "-o", "fmugen.toml"], "n" + DOWN * 2 + ENTER + "nope" + ENTER + DOWN + ENTER * 2)
    assert 'call = "theta"' in (tmp_path / "fmugen.toml").read_text()
    assert "nope" in capsys.readouterr().err


def test_an_existing_config_given_is_asked_about_before_the_rest(run, tmp_path):
    (tmp_path / "fmugen.toml").write_text("mine")
    # not overwritten: another path. Then no probe, the start of x, nothing else
    run(["plant.py:gain", "--fmi", "2", "-o", "fmugen.toml"], "n" + "\x7f" * 11 + "other.toml" + ENTER + "n" + ENTER * 2)
    assert (tmp_path / "fmugen.toml").read_text() == "mine" and (tmp_path / "other.toml").exists()


def test_a_folder_as_the_path_means_the_config_in_it(run, tmp_path):
    (tmp_path / "sub").mkdir()
    (tmp_path / "up.py").write_text("def f(x: float = 1.0):\n    return {'y': x}\n")
    (tmp_path / "sub" / "m.py").write_text("def f(x: float = 1.0):\n    return {'y': x}\n")
    run(["sub/m.py", "--fmi", "2", "-o", "."], "n" + ENTER)
    assert (tmp_path / "fmugen.toml").exists()


def test_a_path_the_model_cant_be_named_from_is_asked_again(run, tmp_path, capsys):
    (tmp_path / "sub").mkdir()
    run(["plant.py:offset", "--fmi", "2", "-o", "sub/c.toml"], "\x7f" * 11 + "c.toml" + ENTER + "n" + ENTER)
    assert (tmp_path / "c.toml").exists() and "must be in the folder" in capsys.readouterr().err


def test_arrays_are_read_with_the_fmi_version_chosen(run, tmp_path):
    (tmp_path / "plant.py").write_text("def f(xs=[1.0, 2.0]):\n    return {'y': sum(xs)}\n")
    seen = []
    real = wizard._draft
    wizard._draft = lambda args, values: seen.append(args.fmi) or real(args, values)
    try:
        run(["plant.py", "-o", "fmugen.toml"], DOWN + ENTER + "n" + ENTER)
    finally:
        wizard._draft = real
    assert seen == [3]


def test_a_typo_in_a_number_doesnt_become_a_text():
    assert wizard._check_value("1.0.0", 1.0) != True
    assert wizard._check_value("2.5", 1.0) is True
    assert wizard._check_value("hello", "abc") is True
    assert wizard._check_value("call:nothing", 1.0) != True
    assert wizard._check_value("call:numpy:eye(3)", 1.0) is True


def test_a_whole_number_for_a_real_is_asked_about(run, tmp_path, capsys):
    run(["plant.py:gain", "--fmi", "2", "-o", "fmugen.toml"], "n" + "\x7f" * 3 + "5" + ENTER * 3)
    assert "start = 5.0" in (tmp_path / "fmugen.toml").read_text()
    assert "--start x=5.0" in capsys.readouterr().err


def test_a_count_for_a_real_can_be_an_integer(run, tmp_path):
    run(["plant.py:gain", "--fmi", "2", "-o", "fmugen.toml", "--force"], "n" + "\x7f" * 3 + "5" + ENTER + DOWN + ENTER * 2)
    assert "x = { start = 5 }" in (tmp_path / "fmugen.toml").read_text()


BUILDERS = """
    class Tank:
        def __init__(self, area: float = 1.0):
            self.area = area

        @classmethod
        def from_file(cls, path: str = "t.txt"):
            return cls()

        def step(self, q: float = 1.0):
            return {"level": q}

        def reset(self):
            pass
"""


def test_the_classmethod_that_builds_it_is_asked(run, tmp_path, capsys):
    (tmp_path / "plant.py").write_text(textwrap.dedent(BUILDERS))
    run(["plant.py", "--fmi", "2", "-o", "fmugen.toml"], "n" + DOWN + ENTER + ENTER * 2)
    assert 'create = "from_file"' in (tmp_path / "fmugen.toml").read_text()
    assert "--create from_file" in capsys.readouterr().err


def test_building_the_object_every_step_can_be_chosen_over_a_method(run, tmp_path, capsys):
    (tmp_path / "plant.py").write_text(textwrap.dedent(CLASS))
    run(["plant.py", "--fmi", "2", "-o", "fmugen.toml"], "n" + DOWN * 3 + ENTER + ENTER + "n")
    assert 'kind = "function"' in (tmp_path / "fmugen.toml").read_text()
    assert "--kind function" in capsys.readouterr().err


def test_a_class_with_nothing_to_choose_isnt_asked(run, tmp_path):
    (tmp_path / "plant.py").write_text("class C:\n    def __call__(self, u: float = 1.0):\n        return u\n")
    run(["plant.py", "--fmi", "2", "-o", "fmugen.toml"], "n" + ENTER)
    assert 'entry = "plant.py:C"' in (tmp_path / "fmugen.toml").read_text()


def test_other_options_are_offered(run, tmp_path, capsys):
    run(["plant.py:offset", "--fmi", "2", "-o", "fmugen.toml"], "n" + DOWN + ENTER + "x=2.5" + ENTER + ENTER)
    assert "start = 2.5" in (tmp_path / "fmugen.toml").read_text()
    assert "--start x=2.5" in capsys.readouterr().err


def test_an_option_that_fails_on_the_model_is_left_out(run, tmp_path, capsys):
    (tmp_path / "plant.py").write_text("def f(x: float = 1.0):\n    return {'y': x}\n")
    # --read of nothing, with the probe: refused, then the menu again
    run(["plant.py", "--fmi", "2", "-o", "fmugen.toml", "--probe"],
        DOWN * 4 + ENTER + "z=return:nope" + ENTER + ENTER)
    err = capsys.readouterr().err
    assert "left out" in err and "--read" not in err.split("To repeat")[1].splitlines()[0]


def test_no_outputs_offers_the_probe(run, tmp_path, capsys):
    (tmp_path / "plant.py").write_text('def f(x: float = 1.0):\n    print(x)\n')
    run(["plant.py", "--fmi", "2", "-o", "fmugen.toml"], "n" + ENTER + "y")
    assert "--probe" in capsys.readouterr().err


def test_a_module_without_a_name_lists_what_it_has(run, tmp_path):
    (tmp_path / "localpkg").mkdir()
    (tmp_path / "localpkg" / "__init__.py").write_text("")
    (tmp_path / "localpkg" / "core2.py").write_text(
        "def f(x: float = 1.0):\n    return {'y': x}\ndef g(x: float = 1.0):\n    return {'z': x}\n")
    run(["localpkg.core2", "--fmi", "2", "-o", "fmugen.toml"], DOWN + ENTER + "n" + ENTER)
    assert 'entry = "localpkg.core2:g"' in (tmp_path / "fmugen.toml").read_text()


def test_targets_are_checked_when_typed(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "m.py").write_text("")
    assert wizard._check_target('"m.py"') is True
    assert wizard._check_target("no.py") != True
    assert wizard._check_target("nosuchmodule_x.mod:f") != True
    assert wizard._check_target("json:loads") is True


def test_the_command_keeps_the_order_of_the_values():
    args = type("A", (), dict(model="m.py", start=["a=1", "c=3"], const=["b=2"], value_order=["a", "b", "c"], probe=True,
                              probe_timeout=5.0, create=None, kind=None, call="step", setup=[], read=[], convert=[],
                              keep=[], integer=[], fmi=2, output="x.toml", force=True))()
    assert wizard._command(args) == ["m.py", "--start", "a=1", "--const", "b=2", "--start", "c=3", "--probe",
                                     "--probe-timeout", "5", "--call", "step", "--fmi", "2", "-o", "x.toml", "--force"]


def test_end_of_input_cancels(run, tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(wizard, "_ask_model", lambda: (_ for _ in ()).throw(EOFError()))
    with pytest.raises(SystemExit) as stop:
        run([], "")
    assert stop.value.code == 130
