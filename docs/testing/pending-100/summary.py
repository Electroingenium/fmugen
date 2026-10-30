"""One line per model from results.json: each config's verdict and the compiled FMU's."""
import json
from pathlib import Path

rows = json.loads((Path(__file__).parent / "results.json").read_text())
for r in rows:
    cfgs = {c["config"]: c for c in r["configs"]}
    v = lambda k: (cfgs.get(k) or {}).get("verdict", "-")
    cp = r["compiled"]
    vers = ", ".join(f"{k} {x}" for k, x in r["versions"].items())
    print(f"{r['model']:18} fmi{r['fmi']} probe={v('probe'):15} plain={v('plain'):15} hand={v('hand'):10} "
          f"seeded={v('seeded'):8} comp={cp.get('verdict')} {cp.get('config')} {cp.get('size_mb')}MB "
          f"n={cp.get('compared')} | {vers}")
