"""GUI-Test der Einzelwerkzeuge (ergänzt gui_smoke.py): Boden-Varianten, Füllungen bearbeiten,
Korrigieren, Ausrichten an Symmetrie, Optimieren, Bohrung per Klick, Messen per Klick,
Symmetrie anlegen/erzwingen, Export-Modi.

xvfb-run -a python tests/gui_tools.py scan_mit_offenem_boden.stl ausgabe_ordner [--en]
"""
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

set_lang("en" if "--en" in sys.argv else "de")
app = QApplication(sys.argv[:1])
app.setStyle("Fusion")
from gui.main_window import MainWindow  # noqa: E402

args = [a for a in sys.argv[1:] if not a.startswith("--")]
stl, out = args[0], args[1]
os.makedirs(out, exist_ok=True)
w = MainWindow()
w.resize(1720, 1020)
w.show()
errors = []
w._error = lambda msg: (errors.append(msg), print("GUI-FEHLER:", msg[:2000]))
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


def ok_since(n0):
    """keine neuen GUI-Fehler seit n0 (Folgefehler nicht doppelt zählen)"""
    return len(errors) == n0


def check(cond, text):
    print(("OK   " if cond else "FAIL ") + text, flush=True)
    if not cond:
        errors.append(text)


w.open_mesh(stl)
wait()
w.do_check()
wait()
# Boden schließen mit "z = Wert" nach grobem Ausrichten
w.goto_page("scan")
w._holes = None
w._refresh_holes()
w.hole_list.setCurrentRow(0)                       # größter Rand
w.base_mode.setCurrentIndex(w.base_mode.findData("lowest"))
w.close_base_ui()
wait()
floor = next((f for f in w.features if f.params.get("fixed")), None)
check(floor is not None, "Boden schließen → festgelegte Ebene")
n_fills = len(w.model.fills)
check(n_fills >= 1, f"{n_fills} Füllung(en) protokolliert")
w.fill_holes_upto()
wait()
check(w.model.boundary_edge_count == 0 or len(w.model.fills) > n_fills, "restliche Löcher gefüllt")
shot("t01_boden")
# Füllung als Ebene erzwingen (eine Lochfüllung)
w.refresh_fills()
row = next((i for i, fl in enumerate(w.model.fills) if fl.get("kind") == "hole"), None)
if row is not None:
    w.fill_list.setCurrentRow(row)
    nfeat = len(w.features)
    w.force_plane_ui()
    wait()
    check(len(w.features) == nfeat + 1 and w.features[-1].params.get("fixed"), "Füllung als Ebene erzwungen → festgelegte Ebene")
    w.refresh_fills()
    w.fill_list.setCurrentRow(row)
    w.refill_mode.setCurrentIndex(w.refill_mode.findData("smooth"))
    n0 = len(errors)
    w.refill_ui()
    wait()
    check(ok_since(n0), "Füllung neu gefüllt")
    w.refresh_fills()
    w.fill_list.setCurrentRow(row)
    n0 = len(errors)
    w.fair_fill_ui()
    wait()
    check(ok_since(n0), "Füllung gefairt")
# Boden-Ebene ändern (Abstand 1 mm zur aktuellen Bodenebene → Lage z−1)
w.refresh_fills()
w.base_mode.setCurrentIndex(w.base_mode.findData("offset"))
w.base_ref.setCurrentIndex(w.base_ref.findData(floor.id))
w.base_value.setValue(1.0)
o_before = np.asarray(w.doc.feature(floor.id).params["origin"]).copy()
n_before = np.asarray(w.doc.feature(floor.id).params["normal"]).copy()
w.change_base_plane()
wait()
f2 = w.doc.feature(floor.id)
moved = float((np.asarray(f2.params["origin"]) - o_before) @ n_before)
check(abs(abs(moved) - 1.0) < 1e-6, f"Boden-Ebene geändert (verschoben um {moved:+.4f} mm)")
w.undo()
check(abs(float((np.asarray(w.doc.feature(floor.id).params["origin"]) - o_before) @ n_before)) < 1e-9, "Undo Boden-Ebene")
# Korrigieren: Auswahl glätten
w.sel[:] = False
w.sel[np.arange(0, w.model.n_faces, 50)] = True
w.sel = w.model.grow(w.sel, 1)
n0 = len(errors)
w.region_op("taubin")
wait()
check(ok_since(n0), "Auswahl glätten")
w.sel[:] = False
# Erkennen + Bohrungen + Symmetrien
w.goto_page("detect")
w.do_detect()
wait()
w.do_merge()
wait()
w.optimize_align_ui()
wait()
w.do_detect_holes()
wait()
holes = [f for f in w.features if f.kind == "hole"]
check(len(holes) >= 1, f"{len(holes)} Bohrungen")
# Bohrung per Klick (K): Wand der ersten Bohrung löschen und neu anklicken
if holes:
    h = holes[0]
    ids = np.asarray(h.face_ids)
    P = w.model.centroids[ids] - h.params["point"]
    a = h.params["axis"]
    z = P @ a
    rho = np.linalg.norm(P - np.outer(z, a), axis=1)
    wall = ids[(np.abs(rho - 0.5 * h.params["d"]) < 0.1) & (z > 0.5) & (z < 2.0) & ~w.model.synthetic[ids]]
    w.doc.features = [f for f in w.features if f.id != h.id]
    w.refresh_all()
    if len(wall):
        w._set_tool("bore")
        w.bore_from_click(int(wall[0]))
        wait()
        nh = [f for f in w.features if f.kind == "hole"]
        check(len(nh) == len(holes), "Bohrung per Klick (K) neu angelegt")
        f = nh[-1]
        check(abs(f.params["d"] - h.params["d"]) < 0.1, f"Ø per Klick {f.params['d']:.3f} ≈ {h.params['d']:.3f}")
        shot("t02_bohrung_klick")
# Bohrungsumgebung als Drehprofil (Ringnuten/Fasen) – einzeln und für alle
w.goto_page("bores")
w.refresh_bores()
n0 = len(errors)
w.bore_port()
wait()
w.bore_port_all()
wait()
check(ok_since(n0) and any(f.kind == "hole" for f in w.features), "Bohrungsumgebung als Drehprofil")
# Messen per Klick auf zwei Ebenen
planes = [f for f in w.features if f.kind == "plane" and len(f.face_ids) > 20 and not f.params.get("fixed")]
if len(planes) >= 2:
    w._set_tool("measure")
    w.measure_click(int(planes[0].face_ids[0]), w.model.centroids[planes[0].face_ids[0]])
    w.measure_click(int(planes[1].face_ids[0]), w.model.centroids[planes[1].face_ids[0]])
    txt = w.meas_out.toPlainText()
    check("°" in txt, "Messen per Klick: " + txt.splitlines()[0] if txt else "Messen per Klick")
    shot("t03_messen")
w._set_tool("navigate")
# Symmetrien: suchen, aus Mittelebene anlegen, erzwingen, ausrichten
w.goto_page("symcon")
w.sym_detect()
wait()
n_sym = len(w.doc.symmetries)
check(n_sym >= 1, f"{n_sym} Symmetrien")
w.refresh_symcon()
mid = next((i for i in range(w.sym_src.count()) if str(w.sym_src.itemData(i)).startswith("mid:")), None)
if mid is not None:
    w.sym_src.setCurrentIndex(mid)
    w.sym_add("mirror")
    check(len(w.doc.symmetries) == n_sym + 1, "Spiegelebene aus Mittelebene angelegt")
w.sym_list.setCurrentRow(0)
nc = len(w.doc.constraints)
w.sym_enforce()
check(len(w.doc.constraints) > nc, f"Erzwingen: {len(w.doc.constraints) - nc} Bedingungen")
n0 = len(errors)
w.con_solve()
wait()
check(ok_since(n0), "Symmetrie-Bedingungen gelöst")
print(w.con_report.toPlainText())
shot("t04_symmetrie")
# Ausrichten an Symmetrie (3-2-1: Z aus Symmetrie)
w.goto_page("align")
w._refresh_align_combos()
i = next((k for k in range(w.al_z.count()) if str(w.al_z.itemData(k)).startswith("sym:")), None)
if i is not None:
    w.al_z.setCurrentIndex(i)
    w.al_o.setCurrentIndex(w.al_o.findData("keep"))
    n0 = len(errors)
    w.align_custom()
    wait()
    check(ok_since(n0), "3-2-1 an Symmetrie ausgerichtet")
    w.undo()
# Export-Modi
w.goto_page("export")
for mode in ("bool", "trim", "scan"):
    w.ex_modes[mode].setChecked(True)
    p = os.path.join(out, f"export_{mode}.step")
    QFileDialog.getSaveFileName = staticmethod(lambda *a, _p=p, **k: (_p, ""))
    w.export_step()
    wait()
    check(os.path.exists(p) and os.path.getsize(p) > 500, f"Export-Modus {mode}")
w.ex_fillet.setChecked(True)
w.ex_fillet_r.setValue(0.0)
w.ex_color.setCurrentIndex(w.ex_color.findData("dev"))
n0 = len(errors)
w.act_step_prev.setChecked(True)
wait()
_t = w.export_log.toPlainText()
check(ok_since(n0) and ("Abweichung" in _t or "deviation" in _t),
      "STEP-Vorschau mit Verrundung erkannter Kantenrundungen und Abweichungsfarben")
shot("t05_vorschau")
w.act_step_prev.setChecked(False)
w.ex_color.setCurrentIndex(w.ex_color.findData("src"))
n0 = len(errors)
w.act_step_prev.setChecked(True)
wait()
_t = w.export_log.toPlainText()
check(ok_since(n0) and ("Abgleich Features" in _t or "check features" in _t) and "Features" in w.lbl_closed.text(),
      "STEP-Vorschau mit Abgleich Features ↔ STEP und Herkunftsfarben")
print(_t[-1500:])
shot("t06_herkunft")
# Über / Zitieren
from gui.about_dialog import AboutDialog  # noqa: E402
_d = AboutDialog(w)
_d.show()
app.processEvents()
_d.copy("bibtex")
_clip = QApplication.clipboard().text()
check("1.0.1" in _clip and "Claude Opus 5.5" in _clip and "Lanz, Amon P." in _clip and w.btn_about.isVisible(),
      "Über/Zitieren: Knopf in der Ecke, BibTeX kopiert")
shot_w = _d.grab()
shot_w.save(os.path.join(out, "t07_about.png"))
_d.close()
print(f"Fehler: {len(errors)}")
for e in errors:
    print("  -", e[:300])
sys.exit(1 if errors else 0)
