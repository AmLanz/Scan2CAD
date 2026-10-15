"""Seite 3 · Erkennen: Formen (RANSAC), Ähnliche zusammenführen, Bohrungen, Restflächen → Freiform."""
from __future__ import annotations

import numpy as np
from qtpy.QtCore import Qt
from qtpy.QtWidgets import QCheckBox, QFormLayout, QHBoxLayout, QSlider, QVBoxLayout, QWidget

from core.analysis import merge_similar
from core.autopatch import auto_freeform
from core.bores import detect_holes, hole_summary, snap_hole
from core.features import Feature, _stats, trim_primitive
from core.fitting import KIND_LABELS
from core.i18n import tr
from core.ransac import detect_primitives

from .common import Section, button, combo, dspin, hint, ispin


class DetectPage:
    def _page_detect(self):
        w = QWidget()
        v = QVBoxLayout(w)
        s = Section(tr("Formen erkennen (RANSAC)", "Detect shapes (RANSAC)"))
        h = QHBoxLayout()
        self.det_kinds = {}
        for k in ("plane", "cylinder", "sphere", "cone", "torus"):
            c = QCheckBox(KIND_LABELS[k].split("/")[0])
            c.setChecked(True)
            self.det_kinds[k] = c
            h.addWidget(c)
        s.add(h)
        f = QFormLayout()
        self.det_eps = dspin(0.1, 0.001, 10, 0.01, 3, "mm")
        self.det_ang = dspin(15, 1, 60, 1, 1, "°")
        self.det_min = dspin(0.2, 0.01, 50, 0.1, 2, tr("% Fläche", "% area"))
        self.det_max = ispin(80, 1, 500)
        self.det_only_free = QCheckBox(tr("nur noch nicht zugeordnete Flächen", "only unassigned areas"))
        self.det_only_free.setChecked(True)
        f.addRow(tr("Abstandstoleranz", "Distance tolerance"), self.det_eps)
        f.addRow(tr("Normalentoleranz", "Normal tolerance"), self.det_ang)
        f.addRow(tr("min. Größe", "min. size"), self.det_min)
        f.addRow(tr("max. Anzahl", "max. count"), self.det_max)
        f.addRow(self.det_only_free)
        s.add(f)
        s.add(button(tr("Erkennen", "Detect"), self.do_detect, bold=True))
        s.add(hint(tr("Abstandstoleranz ≈ 4× Rauschen (wird beim Prüfen gesetzt). Künstliche Dreiecke (Füllungen) "
                      "dienen nie als Startpunkt.", "Distance tolerance ≈ 4× noise (set by Check). Synthetic triangles "
                      "(fills) are never used as seeds.")))
        v.addWidget(s)

        s = Section(tr("Aufräumen", "Clean up"))
        s.add(button(tr("Ähnliche Features zusammenführen", "Merge similar features"), self.do_merge,
                     tr("Gleiche Ebenen (z. B. Riffel einer Füllung), Hälften einer Bohrung usw. gemeinsam neu fitten",
                        "Refit equal planes (e.g. ripples of a fill), halves of a hole etc. together")))
        s.add(button(tr("Bohrungen erkennen (+ Werte glätten)", "Detect holes (+ smooth values)"), self.do_detect_holes,
                     tr("Konkave Zylinder → Bohrungs-Features mit Senkung/Boden; Details im Bohrungen-Menü",
                        "Concave cylinders → hole features with countersink/bottom; details in the holes menu")))
        v.addWidget(s)

        s = Section(tr("Restflächen → Freiform", "Remaining areas → freeform"), expanded=False)
        f = QFormLayout()
        self.ap_spacing = dspin(3.0, 0.2, 100, 0.5, 2, "mm")
        self.ap_smooth = QSlider(Qt.Horizontal)
        self.ap_smooth.setRange(0, 100)
        self.ap_smooth.setValue(35)
        self.ap_angle = dspin(65, 20, 85, 5, 0, "°")
        self.ap_min = ispin(40, 5, 100000)
        self.ap_scope = combo([("free", tr("alle nicht zugeordneten Flächen", "all unassigned areas")),
                               ("sel", tr("nur aktuelle Auswahl", "current selection only"))])
        f.addRow(tr("Bereich", "Scope"), self.ap_scope)
        f.addRow(tr("Kontrollpunkt-Abstand", "Control point spacing"), self.ap_spacing)
        f.addRow(tr("Glättung", "Smoothing"), self.ap_smooth)
        f.addRow(tr("max. Krümmung je Patch", "max. curvature per patch"), self.ap_angle)
        f.addRow(tr("min. Dreiecke je Patch", "min. triangles per patch"), self.ap_min)
        s.add(f)
        s.add(button(tr("Restflächen als Freiform-Patches übersetzen", "Translate remaining areas into freeform patches"),
                     self.do_autopatch))
        s.add(hint(tr("Für die Scan-Topologie nicht nötig: schmale freie Streifen werden dort automatisch den "
                      "Nachbarflächen zugeschlagen. Sinnvoll für echte Freiformbereiche.",
                      "Not required for the scan topology: narrow free strips are assigned to neighbouring surfaces "
                      "automatically. Useful for genuine freeform areas.")))
        v.addWidget(s)
        v.addStretch(1)
        return w

    def do_detect(self):
        kinds = tuple(k for k, c in self.det_kinds.items() if c.isChecked())
        if not kinds:
            return
        only_free = self.det_only_free.isChecked()
        args = dict(eps=self.det_eps.value(), angle_deg=self.det_ang.value(),
                    min_area_frac=self.det_min.value() / 100, kinds=kinds, max_shapes=self.det_max.value())

        def fn(d, rep):
            excl = (d.assigned() > 0) if only_free else None
            res = detect_primitives(d.model, exclude_mask=excl,
                                    progress=lambda n, k: rep(tr(f"Erkennung: {n} Formen", f"Detection: {n} shapes")), **args)
            cnt = {}
            for kind, fr, ids in res:
                try:
                    pr = trim_primitive(kind, fr.params, d.model, ids, {"outline_tol": max(0.05, 2 * args["eps"])})
                    d.features.append(Feature(kind, pr, "surface", ids, stats=_stats(fr.residuals, fr.weights), auto=True))
                    cnt[kind] = cnt.get(kind, 0) + 1
                except Exception:
                    continue
            return [tr("Erkannt: ", "Detected: ") + (", ".join(f"{v}× {KIND_LABELS[k]}" for k, v in cnt.items()) or "–")]
        self.run_doc(tr("Formen erkennen", "Detect shapes"), fn)

    def do_merge(self):
        tol = self.fit_tol.value()

        def fn(d, rep):
            res = merge_similar(d.model, d.features, tol)
            if not res:
                return [tr("Nichts zusammenzuführen.", "Nothing to merge.")]
            removed, first = set(), {}
            for nf, old in res:
                removed |= set(old)
                first[old[0]] = nf
            d.features = [first.get(f.id, f) for f in d.features if f.id in first or f.id not in removed]
            return [tr(f"{len(removed)} Features zu {len(res)} zusammengeführt", f"{len(removed)} features merged into {len(res)}")]
        self.run_doc(tr("Zusammenführen", "Merge"), fn)

    def do_detect_holes(self):
        tol = self.fit_tol.value()

        def fn(d, rep):
            res = detect_holes(d.model, d.features, tol)
            consumed = set()
            out = []
            for r in res:
                p, notes = snap_hole(r["params"], axis_dirs=np.eye(3),
                                    meas=min(self.fit_tol.value(), max(2.0 * float(r.get("rms", 1.0)), 0.005)))
                f = Feature("hole", p, "sub", r["face_ids"], auto=True)
                f.name = tr("Bohrung", "Hole") + f" {f.id}"
                d.features.append(f)
                consumed |= set(r["consumed"])
                out.append(f"{f.name}: {hole_summary(p)}")
            d.features = [f for f in d.features if f.id not in consumed]
            return out or [tr("Keine Bohrungen gefunden.", "No holes found.")]
        self.run_doc(tr("Bohrungen erkennen", "Detect holes"), fn, after=lambda: self.goto_page("bores"))

    def do_autopatch(self):
        if not self._need_mesh():
            return
        if self.ap_scope.currentData() == "sel":
            ids = np.nonzero(self.sel & (self.assigned == 0))[0]
        else:
            ids = np.nonzero(self.assigned == 0)[0]
        if len(ids) < self.ap_min.value():
            self._msg(tr("Keine (ausreichend großen) freien Flächen.", "No (sufficiently large) free areas."))
            return
        kw = dict(spacing=self.ap_spacing.value(), smooth=self.ap_smooth.value() / 100.0,
                  tol=self.fit_tol.value(), max_angle_deg=self.ap_angle.value(), min_faces=self.ap_min.value())

        def fn(d, rep):
            feats, rest = auto_freeform(d.model, ids, progress=lambda k, n: rep(f"{k}/{n}"), **kw)
            d.features += feats
            return [tr(f"{len(feats)} Freiform-Patches; {len(rest)} Dreiecke bleiben frei",
                       f"{len(feats)} freeform patches; {len(rest)} triangles stay free")]
        self.run_doc(tr("Restflächen → Freiform", "Remaining → freeform"), fn)
