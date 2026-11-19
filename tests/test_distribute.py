"""FMUs for other machines: vendored wheels (--vendor) and frozen executables (--compile)."""
import importlib.util

import pytest
from fmpy import simulate_fmu

from fmugen.builder import build


def test_compile_finds_the_modules_of_computed_constants():
    from fmugen.distribute import _modules
    interface = {"entry": {"module": "joblib"}, "constants": {
        "filename": {"call": "huggingface_hub:hf_hub_download(repo_id='x', grid=numpy.linspace(0, 1, 3))"}}}
    assert {"joblib", "huggingface_hub", "numpy"} <= set(_modules(interface))


def test_compile_collects_the_dependencies_of_the_model_packages():
    from fmugen.distribute import _packages
    direct, packages, distributions, imports = _packages(["fmpy.simulation", "cloudpickle"])
    assert direct == ["cloudpickle", "fmpy"]   # their submodules are collected
    assert {"fmpy", "numpy", "lxml"} <= set(packages)   # data files and binaries of the whole tree
    assert {"fmpy", "numpy"} <= set(distributions)
    assert {"numpy", "lxml"} <= set(imports)   # fmpy's own dependencies, imported by name
    assert "fmugen" not in packages and "os" not in direct
    assert _packages(["os", "not_installed_anywhere"]) == ([], [], [], [])


def test_nuitka_gets_package_options_only_for_packages():
    # matplotlib's pylab.py and six.py are single modules: Nuitka stops on package options for them
    from fmugen.distribute import _nuitka_package_args
    if not (importlib.util.find_spec("matplotlib") and importlib.util.find_spec("six")):
        pytest.skip("matplotlib and six are needed")
    expected = ["--include-package-data=matplotlib", "--include-package=matplotlib", "--include-module=six"]
    packages, direct = ["matplotlib", "pylab", "six", "not_installed_anywhere"], ["matplotlib", "six"]
    assert _nuitka_package_args(packages, direct) == expected
    import six  # once imported, six's spec has __path__ = [] (for six.moves), as during a build
    assert six.__spec__.submodule_search_locations == []
    assert _nuitka_package_args(packages, direct) == expected


def test_hugging_face_model_ids_are_found(tmp_path):
    from fmugen.distribute import hf_models
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "x.csv").write_text("1")
    interface = {
        "variables": [{"start": "amazon/chronos-bolt-tiny"}, {"start": "data/x.csv"}, {"start": "male"},
                      {"start": 1.0}],
        "constants": {"repo": {"python": "'sb3/ppo-CartPole-v1'"}, "n": {"python": "3"}},
    }
    assert hf_models(interface, tmp_path) == ["amazon/chronos-bolt-tiny", "sb3/ppo-CartPole-v1"]


def test_file_paths_are_not_taken_for_hugging_face_models(tmp_path):
    from fmugen.distribute import hf_models
    (tmp_path / "run" / "data").mkdir(parents=True)
    (tmp_path / "run" / "data" / "weather.bin").write_text("1")
    interface = {
        "cwd": "fmugen_model/run",   # [model] cwd = "run": data/weather.bin is opened from there
        "variables": [{"start": "data/weather.bin"}, {"start": "results/out.csv"}, {"start": "sb3/ppo-x"}],
    }
    assert hf_models(interface, tmp_path) == ["sb3/ppo-x"]


def test_compiled_launcher_restores_the_execute_bits_of_the_whole_bundle():
    # unzipping drops them; gekko runs its solver from inside the bundle (dist/main/_internal/gekko/bin)
    from fmugen.distribute import compiled_launch_command
    assert compiled_launch_command("linux") == ["sh", "-c", "chmod -R u+x dist/main && ./dist/main/main"]
    assert compiled_launch_command("windows")[0] == "powershell"


@pytest.mark.parametrize("meipass", [True, False])   # PyInstaller, Nuitka
def test_data_kept_outside_a_compiled_bundle_is_found(tmp_path, monkeypatch, meipass):
    from fmugen.templates.fmugen_runtime import _helpers
    resources = tmp_path / "resources"
    exe_dir = resources / "dist" / "main"
    bundle = exe_dir / "_internal" if meipass else exe_dir
    (bundle / "parts" / "pv").mkdir(parents=True)
    assert _helpers.outside_bundle(bundle) == bundle   # not compiled: unchanged
    monkeypatch.setattr(_helpers.sys, "frozen", True, raising=False)
    monkeypatch.setattr(_helpers.sys, "executable", str(exe_dir / "main"))
    if meipass:
        monkeypatch.setattr(_helpers.sys, "_MEIPASS", str(bundle), raising=False)
    else:
        monkeypatch.delattr(_helpers.sys, "_MEIPASS", raising=False)
    assert _helpers.outside_bundle(bundle) == resources.resolve()
    assert _helpers.outside_bundle(bundle / "parts" / "pv") == (resources / "parts" / "pv").resolve()
    assert _helpers.outside_bundle(tmp_path) == tmp_path   # not in the bundle


@pytest.mark.slow
def test_compiled_fmu_reads_the_data_files_of_its_working_folder(tmp_path):
    # like NHES and CGE: [model] cwd, and a file the model reads by a relative path
    if not importlib.util.find_spec("PyInstaller"):
        pytest.skip("PyInstaller is not installed")
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "load.csv").write_text("12.5\n")
    (tmp_path / "grid.py").write_text(
        "def load(scale: float = 2.0):\n"
        "    with open('data/load.csv') as f:\n"
        "        return scale * float(f.read())\n")
    (tmp_path / "fmugen.toml").write_text(
        '[model]\nentry = "grid.py:load"\ncwd = "."\nsources = ["data/load.csv"]\n'
        '[inputs]\nscale = { start = 2.0 }\n[outputs]\ny = { from = "return" }\n')
    fmu = build(tmp_path / "fmugen.toml", tmp_path / "grid.fmu", compiler="pyinstaller")[0]
    assert simulate_fmu(str(fmu), stop_time=1, output=["y"])["y"][-1] == 25.0


def test_packages_that_import_their_own_submodules_are_collected_whole(monkeypatch):
    # pyomo.environ imports its plugins by name; pyomo is only a dependency of OpenHENS
    from fmugen import distribute
    monkeypatch.setattr(distribute, "IMPORTS_OWN_SUBMODULES", {"numpy"})
    direct, _packages, _, imports = distribute._packages(["fmpy.simulation"])
    assert "numpy" in direct and "numpy" not in imports   # a dependency of fmpy, collected with its submodules


def test_libraries_loaded_by_path_and_package_folders_are_kept(tmp_path, monkeypatch):
    # torchvision loads _C_stable.so with torch.ops.load_library (not an import, not lib*.so);
    # chgnet opens chgnet/model/../pretrained/x.pth.tar, so the code-only folder model/ must exist
    from fmugen.distribute import _add_package_folders, _loaded_libraries
    pkg = tmp_path / "site" / "visionlike"
    (pkg / "model").mkdir(parents=True)
    (pkg / "pretrained").mkdir()
    for f in ("__init__.py", "model/__init__.py", "_C_stable.so", "libtorch_like.so", "pretrained/w.pth.tar"):
        (pkg / f).write_text("")
    monkeypatch.syspath_prepend(str(tmp_path / "site"))
    assert _loaded_libraries(["visionlike"]) == [(pkg / "_C_stable.so", "visionlike")]
    internal = tmp_path / "_internal"
    _add_package_folders(["visionlike"], internal)
    # a file in it: the FMU is a zip, which keeps no empty folders
    assert [p.name for p in (internal / "visionlike" / "model").iterdir()] == [".keep"]
    # a module imported without a spec makes find_spec raise ValueError (humanfriendly, floris)
    import sys
    import types
    monkeypatch.setitem(sys.modules, "nospec_like", types.ModuleType("nospec_like"))
    sys.modules["nospec_like"].__spec__ = None
    assert _loaded_libraries(["nospec_like", "visionlike"]) == [(pkg / "_C_stable.so", "visionlike")]


def test_libraries_loaded_by_path_go_in_a_hook_not_the_command_line(tmp_path):
    # 194 --add-binary paths (scikit-learn, scipy, numpy) made a 44 165-character PyInstaller command,
    # over Windows' 32 767 limit (WinError 206)
    import ast

    from fmugen.distribute import _binaries_hook
    assert _binaries_hook([], tmp_path / "hooks") is None
    libraries = [(tmp_path / "site" / "pkg" / f"_ext{i}.pyd", "pkg/sub") for i in range(300)]
    hooks = _binaries_hook(libraries, tmp_path / "hooks")
    tree = ast.parse((hooks / "hook-fmugen_runtime.py").read_text(encoding="utf-8"))
    binaries = ast.literal_eval(tree.body[0].value)
    assert binaries == [(str(src), dest) for src, dest in libraries]

def test_libraries_built_on_torch_ship_their_sources(tmp_path, monkeypatch):
    # TorchScript compiles functions when mace or norse is imported, and reads their source to do it
    from fmugen import distribute
    monkeypatch.setattr(distribute, "packages_distributions", lambda: {"torchlib": ["torchlib"]})
    monkeypatch.setattr(distribute, "_hook_dirs", list)

    class Dist:
        requires = ("torch>=2", "numpy")

    import importlib.metadata as md
    monkeypatch.setattr(md, "distribution", lambda name: Dist())
    hooks = distribute._source_hooks(["torchlib"], tmp_path / "hooks")
    assert (hooks / "hook-torchlib.py").read_text() == "module_collection_mode = 'pyz+py'\n"


def test_editable_installs_are_put_on_the_freezer_path(tmp_path, monkeypatch):
    # pip install -e: an import hook finds the project, which isn't on sys.path (the debutanizer project)
    import importlib.machinery
    import sys

    from fmugen.distribute import _editable_paths
    project = tmp_path / "project"
    (project / "debutanizer_like").mkdir(parents=True)
    (project / "debutanizer_like" / "__init__.py").write_text("")

    class EditableFinder:   # what setuptools' __editable___*_finder installs
        @classmethod
        def find_spec(cls, name, path=None, target=None):
            if name == "debutanizer_like":
                return importlib.machinery.PathFinder.find_spec(name, [str(project)])
            return None

    monkeypatch.setattr(sys, "meta_path", [EditableFinder, *sys.meta_path])
    assert _editable_paths(["debutanizer_like", "pytest", "json", "not_installed_anywhere"]) == [project.resolve()]


def test_wheels_are_stored_in_the_fmu_not_deflated_again(tmp_path):
    # torch's 120 MB wheel took 6 s more to deflate again, for 3 MB
    import zipfile

    from fmugen.builder import _write_zip
    folder = tmp_path / "fmu"
    (folder / "resources" / "wheels").mkdir(parents=True)
    (folder / "resources" / "wheels" / "torch-2.15.0-cp313-win_amd64.whl").write_bytes(b"PK" + b"x" * 1000)
    (folder / "resources" / "model.py").write_text("x = 1\n" * 200)
    _write_zip(folder, tmp_path / "m.fmu")
    types = {i.filename: i.compress_type for i in zipfile.ZipFile(tmp_path / "m.fmu").infolist()}
    assert types == {"resources/model.py": zipfile.ZIP_DEFLATED,
                     "resources/wheels/torch-2.15.0-cp313-win_amd64.whl": zipfile.ZIP_STORED}


def test_the_unifmu_template_is_kept_between_builds(tmp_path, monkeypatch):
    # each build ran `unifmu generate` (0.8 s) into a new folder, and Windows Defender scanned its four
    # new binaries on their first read (0.9 s); a kept folder, hard-linked into each FMU, is scanned once
    import os
    import shutil
    from pathlib import Path

    from fmugen import builder
    monkeypatch.setenv(builder.TEMPLATE_DIR_ENV, str(tmp_path / "cache"))
    monkeypatch.setattr(builder, "_templates", {})
    builder.generate_boilerplate(2, tmp_path / "a")
    [kept] = (tmp_path / "cache").iterdir()
    assert kept.name.startswith("python-out-fmi2-")

    # a later build (a new process: nothing cached in memory) doesn't run unifmu generate again
    run = builder.subprocess.run
    def no_generate(cmd, *args, **kwargs):
        assert "generate" not in cmd, cmd
        return run(cmd, *args, **kwargs)
    monkeypatch.setattr(builder, "_templates", {})
    monkeypatch.setattr(builder.subprocess, "run", no_generate)
    builder.generate_boilerplate(2, tmp_path / "b")
    dll = Path("binaries", "win64", "unifmu.dll")
    assert os.path.samefile(tmp_path / "b" / dll, kept / "out" / dll)   # hard-linked, not copied
    monkeypatch.setattr(builder.subprocess, "run", run)

    # a kept folder that is cut short is made again
    shutil.rmtree(kept / "out" / "resources")
    monkeypatch.setattr(builder, "_templates", {})
    builder.generate_boilerplate(2, tmp_path / "c")
    assert (tmp_path / "c" / "resources").is_dir() and (kept / "out" / "resources").is_dir()
