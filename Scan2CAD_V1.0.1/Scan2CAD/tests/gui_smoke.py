"""GUI-Rauchtest v3: fährt 'Auf gut Glück' und die Menüs programmatisch ab, speichert Screenshots.

xvfb-run -a python tests/gui_smoke.py scan.stl ausgabe_ordner [--en]
(scan.stl aus tests/make_scan_like.py; scan.stl.json enthält die Sollwerte)
"""
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "app"))
os.environ.setdefault("QT_API", "pyside6")
import warnings  # noqa: E402

warnings.filterwarnings("ignore")
import numpy as np  # noqa: E402
from qtpy.QtWidgets import QApplication, QFileDialog, QMessageBox  # noqa: E402

from core.i18n import set_lang  # noqa: E402

EN = "--en" in sys.argv
set_lang("en" if EN else "de")
app = QApplication(sys.argv[:1])
app.setStyle("Fusion")
from gui.main_window import MainWindow  # noqa: E402

args = [a for a in sys.argv[1:] if not a.startswith("--")]
stl, out = args[0], args[1]
os.makedirs(out, exist_ok=True)
truth = json.load(open(stl + ".json")) if os.path.exists(stl + ".json") else {}
w = MainWindow()
w.resize(1720, 1020)
w.show()
errors = []
w._error = lambda msg: (errors.append(msg), print("GUI-FEHLER:", msg[:3000]))
QMessageBox.information = staticmethod(lambda *a, **k: None)
QMessageBox.question = staticmethod(lambda *a, **k: QMessageBox.Yes)


def wait(timeout=1800):
    t0 = time.time()
    while True:
        app.processEvents()
        if not w.runner.busy:
            for _ in range(8):
                app.processEvents()
                time.sleep(0.01)
            if not w.runner.busy:
                return
        if time.time() - t0 > timeout:
            raise RuntimeError("timeout")
        time.sleep(0.02)


def shot(name):
    for _ in range(5):
        app.processEvents()
    w.viewer.plotter.render()
    w.grab().save(os.path.join(out, name + ".png"))


def check(cond, text):
    print(("OK   " if cond else "FAIL ") + text, flush=True)
    if not cond:
        errors.append(text)


t0 = time.time()
w.open_mesh(stl)
wait()
check(w.model is not None, f"Netz geladen ({w.model.n_faces} Dreiecke)")
shot("01_geladen")
w.goto_page("scan")
w.do_check()
wait()
check(bool(w.doc.tol), f"Rauschen geschätzt: σ={w.doc.tol.get('sigma', 0):.4f}")
w.chk_rims.setChecked(True)
shot("02_pruefen_raender")

# ----------------------------------------------------------------- Auf gut Glück, Schritt für Schritt
w.lucky_start()                    # mit Pause: führt genau einen Schritt aus
wait()
step = 1
names = {}
while w._lucky_i < w.lucky_list.count():
    key = w.lucky_list.item(w._lucky_i).data(256)  # Qt.UserRole
    w.lucky_continue()
    wait()
    step += 1
    if key in ("base", "detect", "holes", "symmetry", "preview"):
        shot(f"03_gluck_{step:02d}_{key}")
    names[key] = True
    if errors:
        break
check(w.doc.preview is not None and w.doc.preview.get("solid"), "Auf gut Glück: geschlossener Volumenkörper")
print("Bericht Auf gut Glück:\n" + w.lucky_log.toPlainText(), flush=True)
nholes = sum(1 for f in w.features if f.kind == "hole")
check(nholes >= 1, f"{nholes} Bohrungen erkannt")
check(any(f.params.get("fixed") for f in w.features) == bool(truth.get("open_bottom")),
      "Boden als festgelegte Ebene (nur bei offenem Boden)")

# ----------------------------------------------------------------- Bohrungen-Menü
w.act_step_prev.setChecked(False)
w.goto_page("bores")
w.refresh_bores()
check(w.bore_list.count() == nholes, "Bohrungsliste vollständig")
shot("04_bohrungen")
# Anwenderentscheidungen: Durchgangslöcher durchgehend, kleine Kernlöcher M5 8 tief
for i in range(w.bore_list.count()):
    w.bore_list.setCurrentRow(i)
    f = w._bore()
    if f.params["d"] > 5.0:
        w.b_bottom.setCurrentIndex(w.b_bottom.findData("through"))
        w.b_depth.setValue(40.0)
    else:
        w.b_depth.setValue(8.0)
        w.b_bottom.setCurrentIndex(w.b_bottom.findData("drill"))
        w.bore_suggest_thread()
    w.bore_apply()
w.bore_snap_all()
check(all(f.params.get("bottom_seen", True) for f in w.features if f.kind == "hole"), "Bohrungen entschieden")
w.bore_list.setCurrentRow(0)
w.bore_center()
shot("05_bohrung_editor")

# ----------------------------------------------------------------- Symmetrie & Bedingungen
w.goto_page("symcon")
w.refresh_symcon()
check(w.sym_list.count() >= 1, f"{w.sym_list.count()} Symmetrien gelistet")
shot("06_symmetrie")
floor = next((f for f in w.features if f.params.get("fixed")), None)
top = None
if floor is not None:
    fz0 = float(floor.params["origin"][2])
    cands = [f for f in w.features if f.kind == "plane" and not f.params.get("fixed") and f.params["normal"][2] > 0.99
             and abs(f.params["origin"][2] - fz0 - 10.0) < 2.5]
    top = min(cands, key=lambda f: abs(f.params["origin"][2] - fz0 - 10.0)) if cands else None
if floor is not None and top is not None:
    w.con_type.setCurrentIndex(w.con_type.findData("distance"))
    w.con_a.setCurrentIndex(w.con_a.findData(f"f:{top.id}"))
    w.con_b.setCurrentIndex(w.con_b.findData(f"f:{floor.id}"))
    w.con_val.setValue(10.0)
    w.con_add()
    w.con_type.setCurrentIndex(w.con_type.findData("parallel"))
    w.con_add()
    w.con_solve()
    wait()
    fz = float(w.doc.feature(floor.id).params["origin"][2])
    tz = float(w.doc.feature(top.id).params["origin"][2])
    check(abs(abs(tz - fz) - 10.0) < 1e-3, f"Bedingung Abstand Deckfläche–Boden = 10 (ist {abs(tz - fz):.5f})")
    print(w.con_report.toPlainText())
w.con_suggest()
check(w.sug_list.count() > 0, f"{w.sug_list.count()} Vorschläge")
shot("07_bedingungen")

# ----------------------------------------------------------------- Messen
planes = [f for f in w.features if f.kind == "plane" and not f.params.get("fixed")]
if len(planes) >= 2:
    w.right_tabs.setCurrentIndex(1)
    w.refresh_measure_refs()
    w.meas_fa.setCurrentIndex(w.meas_fa.findData(planes[0].id))
    w.meas_fb.setCurrentIndex(w.meas_fb.findData(planes[1].id))
    w.measure_features()
    check("°" in w.meas_out.toPlainText(), "Messen: Winkel angezeigt")
    shot("08_messen")

# ----------------------------------------------------------------- Export (Scan-Topologie)
w.goto_page("export")
w.act_step_prev.setChecked(True)
wait()
check(w.doc.preview is not None and w.doc.preview.get("solid"), "Vorschau nach Bohrungs-Korrektur geschlossen")
shot("09_vorschau")
step_path = os.path.join(out, "ergebnis.step")
QFileDialog.getSaveFileName = staticmethod(lambda *a, **k: (step_path, ""))
w.export_step()
wait()
check(os.path.exists(step_path) and os.path.getsize(step_path) > 1000, "STEP geschrieben")
from core.brep_seg import _volume  # noqa: E402
vol = None
if w.doc.preview is not None and w.doc.preview.get("solid"):
    vol = _volume(w.doc.preview["items"][0][1])
    if truth.get("volume"):
        err = abs(vol - truth["volume"]) / truth["volume"]
        check(err < 0.01, f"Volumen {vol:.1f} vs. Soll {truth['volume']:.1f} ({100 * err:.2f} %)")

# ----------------------------------------------------------------- Rückgängig / Projekt
w.act_step_prev.setChecked(False)
n_before = len(w.features)
w.undo()
w.redo()
check(len(w.features) == n_before, "Undo/Redo")
proj = os.path.join(out, "projekt.s2c")
QFileDialog.getSaveFileName = staticmethod(lambda *a, **k: (proj, ""))
w.save_project()
nf, ns, nc = len(w.features), len(w.doc.symmetries), len(w.doc.constraints)
w.doc.mesh_path = None
w.load_project(proj)
wait()
check(w.doc.mesh_path is not None and len(w.features) == nf and len(w.doc.symmetries) == ns and len(w.doc.constraints) == nc,
      "Projekt speichern/laden (Features, Symmetrien, Bedingungen)")
check(len(w.model.fills) > 0, "Füllungen im Projekt erhalten")
w.goto_page("scan")
w.refresh_fills()
shot("10_fuellungen")
print(f"Gesamt {time.time() - t0:.0f} s; Fehler: {len(errors)}")
for e in errors:
    print("  -", e[:300])
sys.exit(1 if errors else 0)
