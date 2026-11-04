"""init --read: outputs at paths the probe doesn't search (OpenHENS' private attributes, flopy's method chain,
pybamm's string keys on an object that isn't a Mapping)."""
import textwrap

import pytest

from fmugen.__main__ import _parse_reads
from fmugen.builder import init, isolated_imports
from fmugen.config import InterfaceError

MODEL = '''
    class Solution:
        def __init__(self, voltage):
            self._entries = [voltage - 0.1, voltage]

        def __getitem__(self, key):   # takes keys, but isn't a Mapping (pybamm's Solution)
            if key != "Voltage [V]":
                raise KeyError(key)
            return self

        @property
        def entries(self):
            return self._entries


    class Cell:
        def __init__(self):
            self._best = {"case": {"TAC": 0.0}}

        def step(self, current=1.0):
            self._best["case"]["TAC"] = current * 2
            return Solution(3.7 - current)


    def run(current=1.0):
        return Solution(3.7 - current)
'''


def run(tmp_path, target, **kwargs):
    (tmp_path / "cell.py").write_text(textwrap.dedent(MODEL))
    with isolated_imports():
        output, data = init(f"{tmp_path / 'cell.py'}:{target}", output=tmp_path / "fmugen.toml", force=True, **kwargs)
    return data, output.read_text()


def test_probe_checks_and_types_the_paths_of_a_class(tmp_path):
    data, text = run(tmp_path, "Cell", probe=True,
                     reads={"V": "return:Voltage [V].entries.-1", "tac": "attr:_best.case.TAC"})
    assert data["outputs"]["V"] == {"from": "return:Voltage [V].entries.-1"}
    assert data["outputs"]["tac"] == {"from": "attr:_best.case.TAC"}
    assert "--read; 2.7" in text and "--read; 2.0" in text


def test_return_path_of_a_function(tmp_path):
    data, _ = run(tmp_path, "run", probe=True, reads={"V": "return:Voltage [V].entries.-1"})
    assert data["outputs"]["V"] == {"from": "return:Voltage [V].entries.-1"}


def test_a_path_that_fails_in_the_probe_stops_init(tmp_path):
    with pytest.raises(InterfaceError, match=r"--read tac=attr:_best.case.missing: .*KeyError"):
        run(tmp_path, "Cell", probe=True, reads={"tac": "attr:_best.case.missing"})
    with pytest.raises(InterfaceError, match="called as a function"):
        run(tmp_path, "run", probe=True, reads={"x": "attr:anything"})
    with pytest.raises(InterfaceError, match="gives Solution"):
        run(tmp_path, "Cell", probe=True, reads={"v": "return:Voltage [V]"})


def test_without_probe_the_path_is_written_unchecked(tmp_path):
    data, text = run(tmp_path, "Cell", reads={"tac": "attr:_best.case.TAC"})
    assert data["outputs"]["tac"] == {"from": "attr:_best.case.TAC"}
    assert "type not checked" in text


def test_parse_reads():
    assert _parse_reads(["y.a=return", "V=return:Voltage [V].entries.-1", "h=attr:get_model().head()"]) == {
        "y.a": "return", "V": "return:Voltage [V].entries.-1", "h": "attr:get_model().head()"}
    for bad in ("y", "y=attr:", "y=foo:bar", "1y=return", "y=returned"):
        with pytest.raises(InterfaceError, match="--read"):
            _parse_reads([bad])


def test_the_built_fmu_reads_the_paths(tmp_path, make_fmu, adapter):
    run(tmp_path, "Cell", probe=True, reads={"V": "return:Voltage [V].entries.-1", "tac": "attr:_best.case.TAC"})
    fmu = adapter(make_fmu(tmp_path / "fmugen.toml"))
    fmu.initialize()
    assert fmu.set("current", 0.5) == 0
    assert fmu.fmi2DoStep(0.0, 1.0, False) == 0
    assert fmu.get("V", "tac") == pytest.approx([3.2, 1.0])
