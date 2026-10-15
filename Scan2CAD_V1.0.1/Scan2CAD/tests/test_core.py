"""Kern-Test ohne GUI: 'Auf gut Glück' vom rohen Scan bis zum geschlossenen STEP, dann typische
Anwender-Korrekturen (Bohrungen, Bodenabstand per Bedingung), Volumenvergleich mit dem CAD-Original,
Symmetrien, Projekt speichern/laden, STEP-Export.

python tests/test_core.py [ausgabe_ordner]       (erzeugt die Scans selbst mit make_scan_like.py)
python tests/test_core.py ordner scan.stl ...    (vorhandene Scans; scan.stl.json = Sollwerte)
"""
import json
import os
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "app"))
import warnings  # noqa: E402

warnings.filterwarnings("ignore")
import numpy as np  # noqa: E402

from core import autopilot as ap  # noqa: E402
from core import constraints as cs  # noqa: E402
from core import export as exp  # noqa: E402
from core import preprocess as pp  # noqa: E402
from core.bores import hole_face_ids  # noqa: E402
from core.brep_seg import _volume  # noqa: E402
from core.document import Document  # noqa: E402
from core.project import load_project, save_project  # noqa: E402

out = sys.argv[1] if len(sys.argv) > 1 else tempfile.mkdtemp(prefix="s2c_test_")
os.makedirs(out, exist_ok=True)
scans = sys.argv[2:]
if not scans:
    for name, extra in (("testteil", []), ("flansch_offen", ["--part=flansch", "--open-bottom"])):
        p = os.path.join(out, name + ".stl")
        if not os.path.exists(p):
            subprocess.check_call([sys.executable, os.path.join(HERE, "make_scan_like.py"), p, "0.25", "0.015"] + extra)
        scans.append(p)
errors = []


def check(cond, text):
    print(("OK   " if cond else "FAIL ") + text, flush=True)
    if not cond:
        errors.append(text)


for stl in scans:
    truth = json.load(open(stl + ".json"))
    print(f"\n=== {os.path.basename(stl)} ({truth['part']}, offener Boden: {truth.get('open_bottom')}) ===")
    doc = Document(pp.load_mesh(stl), mesh_path=stl)
    t0 = time.time()
    for key in ap.STEP_KEYS:
        d = doc.clone()
        lines = ap.run_step(key, d)
        doc = d
        print(f"[{ap.step_title(key)}] " + " | ".join(lines)[:300])
    print(f"Auf gut Glück: {time.time() - t0:.0f} s")
    check(doc.preview is not None and doc.preview["solid"], "geschlossener Volumenkörper nach 'Auf gut Glück'")
    holes = [f for f in doc.features if f.kind == "hole"]
    check(len(holes) == len(truth["holes"]), f"{len(holes)} Bohrungen (Soll {len(truth['holes'])})")

    # ------------------------------------------------ Anwender-Entscheidungen (was der Scan nicht weiß)
    for f in holes:
        p = dict(f.params)
        th = min(truth["holes"], key=lambda h: abs(h["dia"] - p["d"]))
        check(abs(th["dia"] - p["d"]) < 0.05, f"{f.name}: Ø {p['d']:.3f} (Soll {th['dia']})")
        if th.get("through"):
            p["bottom"], p["depth"] = "through", 40.0
        else:
            p["bottom"], p["depth"], p["point_angle"] = "drill", th["depth"], th.get("point", 118.0)
            if th.get("thread"):
                p["thread"] = {"size": th["thread"], "pitch": 0.8, "length": th["depth"] - 1.0, "display": "cosmetic"}
        if th.get("csk"):
            check(p.get("head") == "csk" and abs(p["head_d"] - th["csk"][0]) < 0.2, f"{f.name}: Senkung {p.get('head')} Ø{p.get('head_d', 0):.2f}")
        f.params = p
        f.face_ids = hole_face_ids(doc.model, p, tol=doc.fit_tol)
        f.invalidate()
    floor = next((f for f in doc.features if f.params.get("fixed")), None)
    check((floor is not None) == bool(truth.get("open_bottom")), "festgelegter Boden genau bei offenem Boden")
    if floor is not None:
        fz = float(floor.params["origin"][2])
        tops = [f for f in doc.features if f.kind == "plane" and not f.params.get("fixed") and f.params["normal"][2] > 0.99]
        top = min(tops, key=lambda f: abs(float(f.params["origin"][2]) - fz - 10.0))
        con = [cs.new_constraint("distance", [f"f:{top.id}", f"f:{floor.id}"], 10.0),
               cs.new_constraint("parallel", [f"f:{top.id}", f"f:{floor.id}"])]
        doc.constraints += con
        res = cs.solve(doc.model, doc.features, doc.symmetries, doc.constraints, tol=doc.fit_tol)
        cs.apply_solution(doc.model, doc.features, doc.symmetries, res, doc.fit_tol)
        print("\n".join(res["report"]))
        dz = abs(float(top.params["origin"][2]) - float(floor.params["origin"][2]))
        check(abs(dz - 10.0) < 1e-3, f"Bedingung Abstand Deckfläche–Boden = 10 (ist {dz:.5f})")
    syms = doc.symmetries
    check(any(s["type"] == "mirror" and s["scope"] == "global" for s in syms), f"globale Spiegelebene gefunden ({len(syms)} Symmetrien)")

    # ------------------------------------------------ Ergebnis
    r = exp.build_scan_topology(doc.model, doc.features, tol=doc.fit_tol)
    check(r["solid"], "Ergebnis geschlossen")
    if r["solid"]:
        vol = _volume(r["items"][0][1])
        err = abs(vol - truth["volume"]) / truth["volume"]
        check(err < 0.01, f"Volumen {vol:.1f} mm³ vs. CAD {truth['volume']:.1f} mm³ ({100 * err:.2f} %)")
    step = os.path.join(out, os.path.basename(stl) + ".step")
    log = exp.export_step(step, doc.features, mode="scan", model=doc.model, tol=doc.fit_tol)
    check(os.path.getsize(step) > 1000, "STEP geschrieben: " + step)
    proj = os.path.join(out, os.path.basename(stl) + ".s2c")
    save_project(proj, doc.model, doc.features, None, {"tol": doc.tol}, doc.symmetries, doc.constraints)
    back = load_project(proj)
    check(len(back["features"]) == len(doc.features) and len(back["model"].fills) == len(doc.model.fills)
          and len(back["symmetries"]) == len(doc.symmetries) and len(back["constraints"]) == len(doc.constraints),
          "Projekt speichern/laden vollständig")

print(f"\n{len(errors)} Fehler")
for e in errors:
    print("  -", e)
sys.exit(1 if errors else 0)
