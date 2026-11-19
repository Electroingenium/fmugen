import subprocess, sys, time, statistics, os
PY = sys.argv[1]
cases = [
  ("fn.py", ["models/fn.py"]),
  ("tank.py", ["models/tank.py"]),
  ("GHFilter", ["filterpy.gh:GHFilter", "--call", "update"]),
  ("Q_white_noise", ["filterpy.common:Q_discrete_white_noise"]),
  ("KalmanFilter", ["filterpy.kalman:KalmanFilter", "--call", "predict"]),
]
for name, args in cases:
    row = [name]
    for probe in (False, True):
        ts = []
        for _ in range(3):
            cmd = [PY, "-m", "fmugen", "init", *args, "-o", "-"] + (["--probe"] if probe else [])
            t = time.perf_counter(); r = subprocess.run(cmd, capture_output=True, text=True); ts.append(time.perf_counter() - t)
        row.append(f"{statistics.median(ts):6.2f}s rc={r.returncode}")
        if r.returncode: row.append(r.stderr.strip().splitlines()[-1][:120])
    print(" | ".join(row), flush=True)
t = time.perf_counter(); subprocess.run([PY, "-c", "import fmugen.interface, fmugen.probe"]); print("import fmugen.interface:", round(time.perf_counter()-t, 2))
t = time.perf_counter(); subprocess.run([PY, "-c", "pass"]); print("bare python:", round(time.perf_counter()-t, 2))
