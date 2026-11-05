"""fmugen FMU launcher for vendored FMUs (fmugen build --vendor).

UniFMU runs this file (see launch.toml) instead of main.py. The first time the FMU
runs on a machine, it creates a virtual environment from the wheels in
resources/wheels/, without a network connection, and caches it. Then it runs
UniFMU's main.py with that environment's Python. Later runs, and other FMUs with
the same wheels, reuse the environment.

The cache is in FMUGEN_ENV_DIR if set, else the user cache folder
(%LOCALAPPDATA%\\fmugen\\envs, ~/Library/Caches/fmugen/envs, ~/.cache/fmugen/envs).

Only the standard library is used, so that any Python 3.8+ can run it.
"""
import hashlib
import os
import shutil
import subprocess
import sys
import tempfile

#: The FMU's resources folder (this file is copied there).
HERE = os.path.dirname(os.path.abspath(__file__))
#: The requirements the vendored wheels satisfy, written by fmugen build --vendor.
REQUIREMENTS = os.path.join(HERE, "requirements.txt")
#: Folder of the vendored wheels, installed offline.
WHEELS = os.path.join(HERE, "wheels")


def say(message):
    """Print a progress or error message on stderr (UniFMU shows the backend's console).

    :param message: the text, prefixed with ``[fmugen]``.
    :type message: str
    """
    sys.stderr.write("[fmugen] " + message + "\n")
    sys.stderr.flush()


def cache_dir():
    """The folder holding the cached environments.

    ``FMUGEN_ENV_DIR`` if set, else the platform's user cache folder + ``fmugen/envs``.

    :return: the folder path (may not exist yet).
    :rtype: str
    """
    if os.environ.get("FMUGEN_ENV_DIR"):
        return os.environ["FMUGEN_ENV_DIR"]
    if sys.platform == "win32":
        base = os.environ.get("LOCALAPPDATA") or os.path.expanduser("~\\AppData\\Local")
    elif sys.platform == "darwin":
        base = os.path.expanduser("~/Library/Caches")
    else:
        base = os.environ.get("XDG_CACHE_HOME") or os.path.expanduser("~/.cache")
    return os.path.join(base, "fmugen", "envs")


def env_key():
    """Same requirements, wheels and interpreter -> same environment.

    Hashing these lets FMUs with the same wheels share one environment, while a different Python
    gets its own (wheels are built per interpreter).

    :return: 16 hex characters naming the environment folder.
    :rtype: str
    """
    h = hashlib.sha256()
    with open(REQUIREMENTS, "rb") as f:
        h.update(f.read())
    h.update("\n".join(sorted(os.listdir(WHEELS))).encode())
    h.update(sys.version.encode())
    h.update(os.path.realpath(sys.executable).encode())
    return h.hexdigest()[:16]


def env_python(env):
    """The Python executable of a virtual environment.

    :param env: the environment folder.
    :type env: str
    :return: its ``Scripts/python.exe`` (Windows) or ``bin/python``.
    :rtype: str
    """
    if sys.platform == "win32":
        return os.path.join(env, "Scripts", "python.exe")
    return os.path.join(env, "bin", "python")


def run(cmd):
    """Run a command, raising with its output when it fails.

    :param cmd: the command and its arguments.
    :type cmd: list[str]
    :raises RuntimeError: a non-zero exit code; the message holds the combined output.
    """
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError("command failed: " + " ".join(cmd) + "\n" + result.stdout)


def create(env):
    """Create a virtual environment and install the vendored wheels into it, offline.

    Uses ``uv`` when it is on PATH (much faster), else ``venv`` and pip. ``--no-index`` with
    ``--find-links`` makes the install use only the FMU's wheels, never the network.

    :param env: the folder to create.
    :type env: str
    :raises RuntimeError: environment creation or installation failed.
    """
    offline = ["--no-index", "--find-links", WHEELS, "-r", REQUIREMENTS]
    uv = shutil.which("uv")
    if uv:
        run([uv, "venv", "--quiet", "--python", sys.executable, env])
        run([uv, "pip", "install", "--quiet", "--python", env_python(env)] + offline)
    else:
        run([sys.executable, "-m", "venv", env])
        run([env_python(env), "-m", "pip", "install", "--quiet", "--disable-pip-version-check"] + offline)


def ensure_env():
    """Return the cached environment for this FMU, creating it on the first run.

    The environment is built in a temporary folder and renamed into place, with a
    ``.complete`` marker, so FMU instances started at the same time never see a half-installed
    one; when another instance wins the race, its environment is used.

    :return: the environment folder.
    :rtype: str
    :raises RuntimeError: the packages could not be installed.
    """
    root = cache_dir()
    env = os.path.join(root, env_key())
    if os.path.exists(os.path.join(env, ".complete")):
        return env
    os.makedirs(root, exist_ok=True)
    say("installing this FMU's packages into " + env + " (first run only)")
    # Build in a private folder and rename it into place, so FMU instances started at
    # the same time never see a half-installed environment.
    tmp = tempfile.mkdtemp(prefix="tmp-", dir=root)
    try:
        create(tmp)
        open(os.path.join(tmp, ".complete"), "w").close()
        try:
            os.rename(tmp, env)
        except OSError:
            if not os.path.exists(os.path.join(env, ".complete")):
                raise
            # another instance finished first; use its environment
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return env


def main():
    """Ensure the environment, then run UniFMU's ``main.py`` with its Python.

    Command-line arguments are passed on (a distributed FMU's private side takes the proxy's port).

    :return: the exit code of ``main.py``, or 1 when the environment can't be installed.
    :rtype: int
    """
    try:
        env = ensure_env()
    except Exception as e:
        say("cannot install this FMU's packages: " + str(e))
        say("the FMU may have no wheels for this platform or Python version; rebuild it with "
            "fmugen build --vendor --platform ... --python-version ...")
        return 1
    # arguments go on to main.py: a distributed FMU's private side takes the proxy's port
    return subprocess.call([env_python(env), os.path.join(HERE, "main.py"), *sys.argv[1:]], cwd=HERE)


if __name__ == "__main__":
    sys.exit(main())
