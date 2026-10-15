"""Eigene Scans durch den ganzen Ablauf schicken und zusammenfassen (ohne GUI).

python tests/test_real.py scan1.stl [scan2.stl ...] [--out=ordner] [--step]

Je Scan: Laufzeit je Schritt, Features nach Art, Bohrungen/Drehkonturen, Ergebnis der
STEP-Vorschau (geschlossen?, Volumen gegen das Netz), Abgleich Features ↔ STEP, optional STEP-Datei.
"""
import os
import sys
import time
import warnings

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "app"))
warnings.filterwarnings("ignore")

from core import autopilot as ap  # noqa: E402
from core import preprocess as pp  # noqa: E402
from core.document import Document  # noqa: E402


def run(path, out=None, step=False):
    t0 = time.time()
    doc = Document(pp.load_mesh(path), mesh_path=path)
    print(f"\n=== {os.path.basename(path)}: {doc.model.n_faces} Dreiecke ===", flush=True)
    times = {}
    for key in ap.STEP_KEYS:
        t = time.time()
        d2 = doc.clone()
        try:
            lines = ap.run_step(key, d2)
        except Exception as e:  # noqa: BLE001
            print(f"FAIL {key}: {e}", flush=True)
            return False
        doc = d2
        times[key] = time.time() - t
        if key in ("noise", "holes", "ports", "cleanup", "preview"):
            for l in lines[:12]:
                print(f"   [{key}] {l}", flush=True)
    kinds = {}
    for f in doc.features:
        kinds[f.kind] = kinds.get(f.kind, 0) + 1
    print("   Features: " + ", ".join(f"{k} {v}" for k, v in sorted(kinds.items())))
    print("   Zeiten: " + ", ".join(f"{k} {v:.1f}s" for k, v in times.items() if v > 0.5))
    res = doc.preview or {}
    ok = bool(res.get("solid"))
    from core.preprocess import vf_to_pd
    vm = vf_to_pd(doc.model.V, doc.model.F).volume
    vb = None
    if res.get("items") and ok:
        try:
            from core.brep_seg import _volume
            vb = _volume(res["items"][0][1])
        except Exception:  # noqa: BLE001
            vb = (res.get("stats") or {}).get("volume")
    print(("OK   " if ok else "FAIL ") + f"geschlossen={ok}  Netzvolumen {vm:.1f}" + (f", Körper {vb:.1f}" if vb else "")
          + f"  ({time.time() - t0:.0f} s)", flush=True)
    if res.get("items"):
        try:
            from core import fidelity
            fr = fidelity.check(doc.model, doc.features, res["items"][0][1], doc.fit_tol)
            for l in fr["lines"][:1] + [l for l in fr["lines"] if "Scanfläche" in l or "scan area" in l]:
                print("   " + l, flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"   Abgleich fehlgeschlagen: {e}")
    if step and res.get("items"):
        from core import occ_build as ob
        out = out or os.path.dirname(os.path.abspath(path))
        p = os.path.join(out, os.path.splitext(os.path.basename(path))[0] + ".step")
        ob.write_step(res["items"], p)
        print(f"   STEP: {p}")
    return ok


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    opts = dict(a[2:].split("=", 1) if "=" in a else (a[2:], "1") for a in sys.argv[1:] if a.startswith("--"))
    if not args:
        print(__doc__)
        sys.exit(2)
    if opts.get("out"):
        os.makedirs(opts["out"], exist_ok=True)
    results = [run(p, opts.get("out"), "step" in opts) for p in args]
    print(f"\n{sum(results)}/{len(results)} geschlossen")
    sys.exit(0 if all(results) else 1)
