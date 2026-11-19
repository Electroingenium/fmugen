import subprocess, sys, time, statistics, shutil, os
PY = sys.argv[1]
cases = [
  ("tank toml", ["models/tank/fmugen.toml"]),
  ("tank toml fmi3", ["models/tank/fmugen.toml", "--fmi", "3"]),
  ("tank toml folder", ["models/tank/fmugen.toml", "--format", "folder"]),
  ("tank.py", ["models/tank/tank.py"]),
  ("tank.py --probe", ["models/tank/tank.py", "--probe"]),
  ("KF toml", ["models/kf/fmugen.toml"]),
  ("tank distributed", ["models/tank/fmugen.toml", "--distributed"]),
]
os.makedirs("out", exist_ok=True)
for name, args in cases:
    ts = []
    for i in range(3):
        out = f"out/{name.replace(' ','_')}{'' if '--format' in args else '.fmu'}"
        shutil.rmtree(out, ignore_errors=True); shutil.rmtree(out.replace('.fmu','') + '_private', ignore_errors=True)
        if os.path.isfile(out): os.remove(out)
        t = time.perf_counter(); r = subprocess.run([PY, "-m", "fmugen", "build", *args, "-o", out], capture_output=True, text=True); ts.append(time.perf_counter() - t)
    size = os.path.getsize(out) if os.path.isfile(out) else sum(os.path.getsize(os.path.join(d, f)) for d, _, fs in os.walk(out) for f in fs)
    print(f"{name:20} median {statistics.median(ts):5.2f}s  runs {' '.join(f'{x:.2f}' for x in ts)}  rc={r.returncode}  size={size/1024:.0f} KB", (r.stderr.strip().splitlines() or [''])[-1][:100] if r.returncode else "", flush=True)
