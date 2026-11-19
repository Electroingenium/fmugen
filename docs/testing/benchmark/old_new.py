"""Old vs new fmugen, interleaved: each case runs with OLD's src, then NEW's, ROUNDS times (one process per run).

Usage (from the folder holding the configs): old_new.py OLD_SRC NEW_SRC ROUNDS
"""
import os
import statistics
import subprocess
import sys
import time

old, new, rounds = sys.argv[1], sys.argv[2], int(sys.argv[3])
cases = [
    ("init forest", ["init", "sklearn.ensemble:RandomForestRegressor", "--probe", "-o", "-"]),
    ("init ridge", ["init", "sklearn.linear_model:Ridge", "--probe", "-o", "-"]),
    ("init cartpole", ["init", "gymnasium.envs.classic_control.cartpole:CartPoleEnv", "--probe", "-o", "-"]),
    ("init linear", ["init", "torch.nn:Linear", "--start", "in_features=4", "--start", "out_features=2", "--probe", "-o", "-"]),
    ("build tank", ["build", "tank/fmugen.toml", "-o", "out/tank_on.fmu"]),
    ("build linear --vendor", ["build", "linear/fmugen.toml", "--vendor", "-o", "out/linear_on.fmu"]),
]
times = {(c, w): [] for c, _ in cases for w in ("old", "new")}
fails = {}
for _ in range(rounds):
    for name, args in cases:
        for which, src in (("old", old), ("new", new)):
            env = {**os.environ, "PYTHONPATH": src, "PYTHONWARNINGS": "ignore"}
            t = time.perf_counter(); r = subprocess.run([sys.executable, "-m", "fmugen", *args], capture_output=True, text=True, env=env, check=False)
            times[(name, which)].append(time.perf_counter() - t)
            if r.returncode: fails[(name, which)] = (r.stderr.strip().splitlines() or ["?"])[-1][:110]
def show(name, which):
    """The median and the runs of one case with one code, or its error."""
    if (name, which) in fails:
        return "failed: " + fails[(name, which)]
    runs = times[(name, which)]
    return f"{statistics.median(runs):6.1f}s ({' '.join(f'{x:.1f}' for x in runs)})"


for name, _ in cases:
    print(f"{name:22} old {show(name, 'old'):38} new {show(name, 'new')}")
