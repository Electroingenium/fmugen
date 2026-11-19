import time, hashlib, shutil, subprocess, tempfile, sys
from pathlib import Path
d = Path(tempfile.mkdtemp())
subprocess.run([shutil.which("unifmu"), "generate", "python", "out", "fmi2"], cwd=d, capture_output=True)
for rnd in (1, 2):
    for p in sorted((d/"out"/"binaries").rglob("*")):
        if p.is_file():
            s = time.perf_counter(); f = open(p, "rb"); o = time.perf_counter(); hashlib.file_digest(f, "sha256"); f.close(); e = time.perf_counter()
            print(f"read {rnd}  {p.parent.name}/{p.name:14} {p.stat().st_size>>10:6} KB  open {o-s:.3f}s  read+hash {e-o:.3f}s")
# a copy of an already-scanned file in a persistent folder
keep = Path(sys.argv[1]); keep.mkdir(exist_ok=True)
for p in (d/"out"/"binaries").rglob("*"):
    if p.is_file(): shutil.copy2(p, keep / f"{p.parent.name}-{p.name}")
