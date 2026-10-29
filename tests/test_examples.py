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
        output, data = init(target, output=tmp_path / "fmugen.toml", **kwargs)
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
