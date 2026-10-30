"""Print, per model, the init result and each config's verdict with its error (for fixing starts)."""
import json
import sys
from pathlib import Path

W = Path.home() / "cmp/w100/m"
names = sys.argv[1:] or sorted(p.name for p in W.iterdir())


def tail(p, n=3):
    try:
        ls = [l for l in p.read_text(errors="replace").splitlines() if l.strip()]
        return " | ".join(ls[-n:])[-400:]
    except OSError:
        return ""


for n in names:
    d = W / n
    comp = json.loads((d / "compiled.json").read_text()) if (d / "compiled.json").exists() else {}
    if comp.get("verdict") == "match" and "-a" not in sys.argv:
        continue
    print(f"== {n}  compiled: {comp.get('verdict')} {comp.get('config') or ''} {(comp.get('notes') or '')[:200]}")
    recs = {r["config"]: r for r in json.loads((d / "results.json").read_text())} if (d / "results.json").exists() else {}
    for kind in ("probe", "plain", *[k for k in recs if k.startswith("hand")]):
        r = recs.get(kind, {})
        if r.get("verdict") in (None, "no config"):
            print(f"  {kind}: init failed: {tail(d / f'init_{kind}.log')}")
        else:
            outs = r.get("fmu_outputs") or []
            print(f"  {kind}: {r.get('verdict')} outs={outs[:6]}{'…' if len(outs) > 6 else ''} {(r.get('notes') or '')[:300]} {('direct: ' + r['direct_error'][:200]) if r.get('direct_error') else ''}")
