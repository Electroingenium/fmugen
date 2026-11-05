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

    def run_init(argv, keys):
        with create_pipe_input() as pipe, create_app_session(input=pipe, output=DummyOutput()):
            pipe.send_text(keys)
            main(["init", *argv])
    yield run_init
    wizard.assume_yes(False)


def test_every_question_with_its_default(run, tmp_path, capsys):
    # the model file, no probe, the first entry, the made-up start of x, FMI 2, the path
    run([], ENTER + "n" + ENTER * 4)
    text = (tmp_path / "fmugen.toml").read_text()
    assert 'plant.py:gain"' in text and "fmi_version = 2" in text and "start = 1.0" in text
    assert "fmugen init plant.py:gain --fmi 2 --yes" in capsys.readouterr().err


def test_answers_become_options(run, tmp_path, capsys):
    # probe, the second entry; in offset, x has a default, so no start value is asked. FMI 3, the path
    run(["plant.py"], "y" + DOWN + ENTER + DOWN + ENTER + ENTER)
    text = (tmp_path / "fmugen.toml").read_text()
    assert 'plant.py:offset"' in text and "fmi_version = 3" in text
    assert "fmugen init plant.py:offset --probe --fmi 3 --yes" in capsys.readouterr().err


def test_a_start_value_typed_is_written(run, tmp_path, capsys):
    # the prompt's text starts as 1.0: erased, then 5.5
    run(["plant.py:gain", "--fmi", "2", "-o", "fmugen.toml"], "n" + "\x7f" * 3 + "5.5" + ENTER)
    assert "start = 5.5" in (tmp_path / "fmugen.toml").read_text()
    assert "--start x=5.5" in capsys.readouterr().err


def test_an_entry_can_be_written(run, tmp_path):
    (tmp_path / "plant.py").write_text("from math import sqrt\n" + textwrap.dedent(MODEL))
    # the entries, then the last choice, "Write my own": sqrt is imported, so it isn't listed
    run(["plant.py", "--fmi", "2", "-o", "fmugen.toml"], "n" + DOWN * 2 + ENTER + "offset" + ENTER)
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
    run(["plant.py", "--fmi", "2", "-o", "fmugen.toml"], "n" + DOWN + ENTER)
    assert 'call = "theta"' in (tmp_path / "fmugen.toml").read_text()
    assert "--call theta" in capsys.readouterr().err


def test_a_step_method_can_be_written(run, tmp_path):
    (tmp_path / "plant.py").write_text(textwrap.dedent(CLASS))
    run(["plant.py", "--fmi", "2", "-o", "fmugen.toml"], "n" + DOWN * 2 + ENTER + "theta" + ENTER)
    assert 'call = "theta"' in (tmp_path / "fmugen.toml").read_text()


def test_data_arguments_are_asked(run, tmp_path, capsys):
    (tmp_path / "plant.py").write_text("def f(signal=None, k: float = 1.0):\n    return {'y': k}\n")
    run(["plant.py", "--fmi", "2", "-o", "fmugen.toml"], "n" + "call:numpy:loadtxt('s.txt')" + ENTER)
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
    run(["plant.py", "--fmi", "2", "-o", "fmugen.toml"], "y" + "call:builtins:list([1.0, 2.0])" + ENTER)
    assert "builtins:list([1.0, 2.0])" in (tmp_path / "fmugen.toml").read_text()
    err = capsys.readouterr().err
    assert "probe call failed" in err and "looks like data" not in err


def test_without_the_probe_it_isnt_called(run, tmp_path):
    (tmp_path / "plant.py").write_text(textwrap.dedent(NEEDS_SIGNAL))
    run(["plant.py", "--fmi", "2", "-o", "fmugen.toml"], "n")   # no question about signal: nothing shows it
    assert "probe" not in (tmp_path / "fmugen.toml").read_text()


def test_a_data_argument_can_be_skipped(run, tmp_path, capsys):
    (tmp_path / "plant.py").write_text("def f(signal=None, k: float = 1.0):\n    return {'y': k}\n")
    run(["plant.py", "--fmi", "2", "-o", "fmugen.toml"], "n" + ENTER)
    assert "looks like data" in capsys.readouterr().err


def test_options_given_are_not_asked(run, tmp_path):
    run(["plant.py:offset", "-o", "fmugen.toml", "--fmi", "3", "--probe"], "")
    assert "fmi_version = 3" in (tmp_path / "fmugen.toml").read_text()


def test_an_existing_config_is_kept_unless_confirmed(run, tmp_path):
    (tmp_path / "fmugen.toml").write_text("mine")
    run(["plant.py:offset", "--fmi", "3", "--probe"], ENTER + "y")
    assert "fmi_version = 3" in (tmp_path / "fmugen.toml").read_text()


def test_ctrl_c_writes_nothing(run, tmp_path, capsys):
    with pytest.raises(SystemExit) as stop:
        run(["plant.py"], CTRL_C)
    assert stop.value.code == 130 and "nothing written" in capsys.readouterr().err
    assert not (tmp_path / "fmugen.toml").exists()


def test_yes_asks_nothing(run, tmp_path):
    run(["plant.py:offset", "--yes"], "")
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
