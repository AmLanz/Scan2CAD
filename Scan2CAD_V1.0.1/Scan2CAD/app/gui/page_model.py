"""Seite 4 · Modellieren: Bereiche anmalen, Typ wählen, fitten, übernehmen."""
from __future__ import annotations

import numpy as np
from qtpy.QtCore import Qt, QTimer
from qtpy.QtWidgets import (QButtonGroup, QCheckBox, QFormLayout, QGridLayout, QHBoxLayout, QLabel,
                            QPlainTextEdit, QRadioButton, QSlider, QStackedWidget, QVBoxLayout, QWidget)

from core import occ_build as ob
from core import preprocess as pp
from core.features import fit_feature, grow_selection_to_fit
from core.fitting import KIND_LABELS
from core.i18n import tr

from .common import Section, button, combo, dspin, hint, ispin, row


def fit_kinds():
    return [("auto", tr("Auto (Primitiv wählen)", "Auto (choose primitive)")), ("plane", tr("Ebene", "Plane")),
            ("cylinder", tr("Zylinder", "Cylinder")), ("sphere", tr("Kugel", "Sphere")), ("cone", tr("Kegel", "Cone")),
            ("torus", tr("Torus / Kantenrundung", "Torus / edge fillet")), ("ellipsoid", tr("Ellipsoid", "Ellipsoid")),
            ("freeform", tr("Freiform (B-Spline)", "Freeform (B-spline)")),
            ("extrusion", tr("Extrusion (Profil × Höhe)", "Extrusion (profile × height)")),
            ("revolve", tr("Rotationskörper", "Revolved body"))]


class ModelPage:
    def _page_model(self):
        w = QWidget()
        v = QVBoxLayout(w)
        s = Section(tr("Werkzeug && Auswahl", "Tool && selection"))
        grid = QGridLayout()
        self.tool_group = QButtonGroup(self)
        self.tool_buttons = {}
        for i, (key, label) in enumerate((("navigate", tr("Navigieren (N)", "Navigate (N)")),
                                          ("brush", tr("Pinsel (B)", "Brush (B)")),
                                          ("erase", tr("Radierer (E)", "Eraser (E)")),
                                          ("fill", tr("Füllen bis Kante (F)", "Fill to edge (F)")))):
            rb = QRadioButton(label)
            self.tool_group.addButton(rb)
            self.tool_buttons[key] = rb
            rb.toggled.connect(lambda on, k=key: on and self._set_tool(k, from_ui=True))
            grid.addWidget(rb, i // 2, i % 2)
        s.add(grid)
        f = QFormLayout()
        self.brush_slider = QSlider(Qt.Horizontal)
        self.brush_slider.setRange(1, 400)
        self.brush_spin = dspin(2.0, 0.05, 200, 0.25, 2, "mm")
        self.brush_slider.valueChanged.connect(lambda v_: self.brush_spin.setValue(self._slider_to_r(v_)))
        self.brush_spin.valueChanged.connect(self._brush_changed)
        f.addRow(tr("Pinselradius", "Brush radius"), row(self.brush_slider, self.brush_spin))
        self.brush_normal = QCheckBox(tr("nur ähnliche Normalen", "similar normals only"))
        self.brush_normal_ang = dspin(30, 1, 90, 1, 0, "°")
        f.addRow(self.brush_normal, self.brush_normal_ang)
        self.brush_protect = QCheckBox(tr("zugeordnete Flächen schützen", "protect assigned areas"))
        self.brush_protect.setChecked(True)
        f.addRow(self.brush_protect)
        self.fill_ang = dspin(20, 1, 90, 1, 0, "°")
        f.addRow(tr("Füllen stoppt an Knick >", "Fill stops at crease >"), self.fill_ang)
        s.add(f)
        grid = QGridLayout()
        for i, (label, fn) in enumerate(((tr("Leeren (C)", "Clear (C)"), self.sel_clear),
                                         (tr("Invertieren", "Invert"), self.sel_invert),
                                         (tr("+1 Ring", "+1 ring"), lambda: self.sel_grow(1)),
                                         (tr("−1 Ring", "−1 ring"), lambda: self.sel_grow(-1)),
                                         (tr("Freie Flächen", "Free areas"), self.sel_unassigned),
                                         (tr("Künstliche (Füllungen)", "Synthetic (fills)"), self.sel_synthetic))):
            grid.addWidget(button(label, fn), i // 2, i % 2)
        s.add(grid)
        self.lbl_sel = QLabel(tr("Auswahl: 0 Dreiecke", "Selection: 0 triangles"))
        s.add(self.lbl_sel)
        v.addWidget(s)

        s = Section(tr("Fitten", "Fit"))
        f = QFormLayout()
        self.fit_kind = combo(fit_kinds())
        self.fit_kind.currentIndexChanged.connect(self._fit_kind_changed)
        f.addRow(tr("Typ", "Type"), self.fit_kind)
        self.fit_tol = dspin(0.05, 0.001, 5, 0.005, 3, "mm")
        f.addRow(tr("Toleranz", "Tolerance"), self.fit_tol)
        self.fit_robust = QCheckBox(tr("robust (Ausreißer abschwächen)", "robust (down-weight outliers)"))
        self.fit_robust.setChecked(True)
        f.addRow(self.fit_robust)
        self.fit_snap = dspin(0.0, 0, 10, 0.01, 3, "mm")
        self.fit_snap.setSpecialValueText(tr("aus", "off"))
        f.addRow(tr("Maße runden auf", "Round dimensions to"), self.fit_snap)
        self.fit_to_planes = QCheckBox(tr("Höhe/Länge bis an parallele Ebenen-Features", "height/length up to parallel plane features"))
        self.fit_to_planes.setChecked(True)
        f.addRow(self.fit_to_planes)
        s.add(f)
        self.fit_stack = QStackedWidget()
        self._fit_pages = {}
        for key, _ in fit_kinds():
            self._fit_pages[key] = self.fit_stack.addWidget(self._fit_options_page(key))
        s.add(self.fit_stack)
        self.btn_fit = button(tr("Fitten (Leertaste)", "Fit (space)"), self.do_fit)
        self.btn_grow = button(tr("Auswahl auf Passendes erweitern (G)", "Grow selection to fit (G)"), self.grow_fit)
        s.add(self.btn_fit)
        s.add(self.btn_grow)
        self.btn_accept = button(tr("✓ Übernehmen (Enter)", "✓ Accept (Enter)"), self.accept_fit, bold=True)
        self.btn_discard = button(tr("✗ Verwerfen", "✗ Discard"), self.discard_fit)
        s.add(row(self.btn_accept, self.btn_discard))
        self.chk_fit_dev = QCheckBox(tr("Abweichungskarte der Vorschau zeigen", "Show deviation map of the preview"))
        self.chk_fit_dev.setChecked(True)
        self.chk_fit_dev.toggled.connect(self._show_preview_dev)
        s.add(self.chk_fit_dev)
        self.fit_result = QPlainTextEdit()
        self.fit_result.setReadOnly(True)
        self.fit_result.setMaximumHeight(150)
        s.add(self.fit_result)
        v.addWidget(s)
        v.addStretch(1)
        return w

    def _fit_options_page(self, key):
        w = QWidget()
        f = QFormLayout(w)
        f.setContentsMargins(0, 0, 0, 0)
        if key in ("auto", "plane"):
            if not hasattr(self, "pl_outline"):
                self.pl_outline = combo([("fit", tr("Kontur fitten (Linien/Radien/Kreise)", "fit contour (lines/radii/circles)")),
                                         ("boundary", tr("Rand als Polygon", "rim as polygon")),
                                         ("hull", tr("konvexe Hülle", "convex hull")),
                                         ("rect", tr("min. Rechteck", "min. rectangle"))])
                self.pl_outline_tol = dspin(0.2, 0.001, 10, 0.05, 3, "mm")
            if key == "plane":
                f.addRow(tr("Umriss (Ebene)", "Outline (plane)"), self.pl_outline)
                f.addRow(tr("Umriss-Toleranz", "Outline tolerance"), self.pl_outline_tol)
            else:
                f.addRow(hint(tr("Wählt das einfachste Primitiv, dessen RMS nahe am besten liegt.",
                                 "Chooses the simplest primitive whose RMS is close to the best.")))
        elif key == "freeform":
            self.ff_ctrl = ispin(12, 4, 80)
            self.ff_smooth = QSlider(Qt.Horizontal)
            self.ff_smooth.setRange(0, 100)
            self.ff_smooth.setValue(30)
            self.ff_mode = combo([("auto", tr("automatisch", "automatic")),
                                  ("planar", tr("Projektion + Trimmung", "projection + trimming")),
                                  ("harmonic", tr("harmonisch (Patch → Rechteck)", "harmonic (patch → rectangle)"))])
            f.addRow(tr("Kontrollpunkte (lange Seite)", "Control points (long side)"), self.ff_ctrl)
            f.addRow(tr("Glättung (fein ↔ grob)", "Smoothing (fine ↔ coarse)"), self.ff_smooth)
            f.addRow(tr("Parametrisierung", "Parametrisation"), self.ff_mode)
        elif key == "extrusion":
            self.ex_profile = combo([("auto", tr("automatisch", "automatic")), ("circle", tr("Kreis", "circle")),
                                     ("rect", tr("Rechteck (± Radien)", "rectangle (± radii)")),
                                     ("polygon", tr("Polygon (± Radien)", "polygon (± radii)")),
                                     ("spline", tr("Freie Kontur (Spline)", "free contour (spline)"))])
            self.ex_fillets = combo([("auto", tr("Eckenradien erkennen", "detect corner radii")),
                                     ("none", tr("scharfe Ecken", "sharp corners"))])
            self.ex_smooth = QSlider(Qt.Horizontal)
            self.ex_smooth.setRange(0, 100)
            self.ex_smooth.setValue(50)
            self.ex_step = dspin(2.0, 0.1, 50, 0.25, 2, "mm")
            self.ex_ext0 = dspin(0.0, -100, 100, 0.1, 3, "mm")
            self.ex_ext1 = dspin(0.0, -100, 100, 0.1, 3, "mm")
            self.ex_dir = combo([("auto", tr("automatisch (aus Normalen)", "automatic (from normals)")),
                                 ("z", tr("Z-Achse", "Z axis")), ("x", tr("X-Achse", "X axis")), ("y", tr("Y-Achse", "Y axis"))])
            f.addRow(tr("Richtung", "Direction"), self.ex_dir)
            f.addRow(tr("Profil", "Profile"), self.ex_profile)
            f.addRow(tr("Ecken", "Corners"), self.ex_fillets)
            f.addRow(tr("Spline-Glättung", "Spline smoothing"), self.ex_smooth)
            f.addRow(tr("Spline-Stützpunktabstand", "Spline knot spacing"), self.ex_step)
            f.addRow(tr("Verlängern Anfang", "Extend start"), self.ex_ext0)
            f.addRow(tr("Verlängern Ende", "Extend end"), self.ex_ext1)
            f.addRow(hint(tr("Mantelflächen markieren (Deckel werden ignoriert).", "Mark the side walls (caps are ignored).")))
        elif key == "revolve":
            self.rv_profile = combo([("auto", tr("automatisch", "automatic")),
                                     ("polyline", tr("Geraden + Radien", "lines + radii")), ("spline", "Spline")])
            self.rv_smooth = QSlider(Qt.Horizontal)
            self.rv_smooth.setRange(0, 100)
            self.rv_smooth.setValue(40)
            self.rv_axis = combo([("auto", tr("automatisch (Normalenlinien)", "automatic (normal lines)")),
                                  ("z", tr("parallel Z", "parallel Z")), ("x", tr("parallel X", "parallel X")),
                                  ("y", tr("parallel Y", "parallel Y"))])
            f.addRow(tr("Achsrichtung", "Axis direction"), self.rv_axis)
            f.addRow(tr("Profil", "Profile"), self.rv_profile)
            f.addRow(tr("Spline-Glättung", "Spline smoothing"), self.rv_smooth)
        else:
            f.addRow(hint({"cylinder": tr("Achse aus Normalen, Radius per Kreisfit, dann Gesamt-LSQ.",
                                          "Axis from normals, radius from circle fit, then full LSQ."),
                           "sphere": tr("Algebraischer Fit + geometrische Verfeinerung.", "Algebraic fit + geometric refinement."),
                           "cone": tr("Spitze aus Tangentialebenen, Achse aus Normalenkegel.", "Apex from tangent planes, axis from normal cone."),
                           "torus": tr("Für Kantenrundungen entlang Kreisbögen.", "For edge fillets along circular arcs."),
                           "ellipsoid": tr("Allgemeine Quadrik → Ellipsoid.", "General quadric → ellipsoid.")}.get(key, "")))
        return w

    # ------------------------------------------------------------------ Pinsel / Auswahl
    def _slider_to_r(self, v):
        return round(0.05 * (1.025 ** v), 3)

    def _r_to_slider(self, r):
        return int(np.clip(np.log(max(r, 0.05) / 0.05) / np.log(1.025), 1, 400))

    def _brush_changed(self, r):
        self.brush_slider.blockSignals(True)
        self.brush_slider.setValue(self._r_to_slider(r))
        self.brush_slider.blockSignals(False)
        self.viewer.set_brush_radius(r)

    def _on_stroke(self, ev):
        if self.model is None:
            return
        if ev["event"] == "begin":
            self.checkpoint(tr("Pinselstrich", "Brush stroke"))
            self._stroke_last = ev["pos"]
            self._stroke_normal = self.model.fit_normals[ev["cell"]]
            self._brush(ev["cell"], ev["pos"], ev["erase"])
        elif ev["event"] == "move":
            p0 = getattr(self, "_stroke_last", None)
            p1 = ev["pos"]
            r = self.viewer.brush_radius
            d = np.linalg.norm(p1 - p0) if p0 is not None else 0
            if p0 is not None and 0.6 * r < d < 6 * r:
                n = int(np.ceil(d / (0.5 * r)))
                tree = self.model.centroid_tree
                for t in np.linspace(0, 1, n + 1)[1:-1]:
                    q = p0 + t * (p1 - p0)
                    dist, cid = tree.query(q)
                    if dist < 0.5 * r:
                        self._brush(int(cid), q, ev["erase"], render=False)
            self._brush(ev["cell"], p1, ev["erase"])
            self._stroke_last = p1
        else:
            self._stroke_last = None
            self.viewer.render()

    def _brush(self, cell, pos, erase, render=True):
        m = self.model
        r = self.viewer.brush_radius
        cand = np.asarray(m.centroid_tree.query_ball_point(pos, r), dtype=np.int64)
        cand = np.union1d(cand, [cell])
        comp = m.component_containing(cand, np.array([cell]))
        if self.brush_normal.isChecked() and getattr(self, "_stroke_normal", None) is not None:
            cosl = np.cos(np.radians(self.brush_normal_ang.value()))
            comp = comp[m.fit_normals[comp] @ self._stroke_normal >= cosl]
        if not erase and self.brush_protect.isChecked():
            comp = comp[self.assigned[comp] == 0]
        if len(comp) == 0:
            return
        old = self.sel[comp].copy()
        self.sel[comp] = not erase
        changed = comp[old != self.sel[comp]]
        if len(changed):
            self.recolor(changed, render=render)

    def _pick_fill(self, ev):
        labels = self.model.angle_graph_labels(self.fill_ang.value())
        comp = np.nonzero(labels == labels[ev["cell"]])[0]
        if not ev["erase"] and self.brush_protect.isChecked():
            comp = comp[self.assigned[comp] == 0]
        self.checkpoint(tr("Füllen", "Fill"))
        self.sel[comp] = not ev["erase"]
        self.recolor()
        self._msg(tr(f"Gefüllt: {len(comp):,} Dreiecke", f"Filled: {len(comp):,} triangles").replace(",", "."))

    def sel_clear(self):
        if self.model is None or not self.sel.any():
            return
        self.checkpoint(tr("Auswahl leeren", "Clear selection"))
        self.sel[:] = False
        self.recolor()

    def sel_invert(self):
        if self.model is None:
            return
        self.checkpoint(tr("Auswahl invertieren", "Invert selection"))
        self.sel = ~self.sel
        if self.brush_protect.isChecked():
            self.sel &= self.assigned == 0
        self.recolor()

    def sel_grow(self, rings):
        if self.model is None:
            return
        self.checkpoint(tr("Auswahl wachsen/schrumpfen", "Grow/shrink selection"))
        if rings > 0:
            self.sel = self.model.grow(self.sel, rings)
            if self.brush_protect.isChecked():
                self.sel &= self.assigned == 0
        else:
            self.sel = self.model.shrink(self.sel, -rings)
        self.recolor()

    def sel_unassigned(self):
        if self.model is None:
            return
        self.checkpoint(tr("Freie Flächen wählen", "Select free areas"))
        self.sel = self.assigned == 0
        self.recolor()

    def sel_synthetic(self):
        if self.model is None:
            return
        self.checkpoint(tr("Künstliche Dreiecke wählen", "Select synthetic triangles"))
        self.sel = self.model.synthetic.copy()
        self.recolor()

    # ------------------------------------------------------------------ Fitten
    def _fit_kind_changed(self, i):
        self.fit_stack.setCurrentIndex(self._fit_pages[self.fit_kind.itemData(i)])

    def _fit_opts(self):
        o = {"tol": self.fit_tol.value(), "robust": self.fit_robust.isChecked(), "snap": self.fit_snap.value(),
             "outline": self.pl_outline.currentData(), "outline_tol": self.pl_outline_tol.value(),
             "ff_ctrl": self.ff_ctrl.value(), "ff_smooth": self.ff_smooth.value() / 100.0,
             "ff_mode": self.ff_mode.currentData(),
             "ex_profile": self.ex_profile.currentData(), "ex_fillets": self.ex_fillets.currentData(),
             "ex_smooth": self.ex_smooth.value() / 100.0, "ex_step": self.ex_step.value(),
             "ex_extend0": self.ex_ext0.value(), "ex_extend1": self.ex_ext1.value(),
             "rv_profile": self.rv_profile.currentData(), "rv_smooth": self.rv_smooth.value() / 100.0}
        axes = {"x": np.array([1.0, 0, 0]), "y": np.array([0, 1.0, 0]), "z": np.array([0, 0, 1.0])}
        o["ex_dir"] = axes.get(self.ex_dir.currentData())
        o["rv_axis"] = axes.get(self.rv_axis.currentData())
        if self.fit_to_planes.isChecked():
            o["planes"] = [(f.params["origin"], f.params["normal"]) for f in self.features if f.kind == "plane"]
        return o

    def _fit_worker(self, model, ids, kind, opts, defl):
        feat, info = fit_feature(model, ids, kind, opts)
        shp = feat.shape()
        V, F = ob.tessellate(shp, defl)
        dev = None
        try:
            import pyvista as pv
            Vd, Fd = ob.tessellate(shp, min(defl, opts["tol"] / 3))
            tgt = pp.vf_to_pd(Vd, Fd)
            dev = np.asarray(pv.PolyData(model.centroids[ids]).compute_implicit_distance(tgt).point_data["implicit_distance"])
            if feat.kind in ("extrusion", "revolve", "freeform") or not feat.stats:
                feat.stats.update({"dev_rms": float(np.sqrt(np.mean(dev ** 2))),
                                   "dev_p95": float(np.percentile(np.abs(dev), 95))})
        except Exception:
            pass
        return {"feature": feat, "info": info, "mesh": (V, F), "ids": ids, "dev": dev}

    def do_fit(self):
        if not self._need_mesh():
            return
        ids = np.nonzero(self.sel)[0]
        if len(ids) < 3:
            self._msg(tr("Erst Flächen auswählen (Pinsel B / Füllen F).", "Select areas first (brush B / fill F)."))
            return
        kind = self.fit_kind.currentData()
        self._run(tr("Fitte…", "Fitting…"), self._fit_worker, self._fit_done, self.model, ids, kind,
                  self._fit_opts(), self._disp_defl())

    def _fit_done(self, res):
        self.preview = res
        f = res["feature"]
        V, F = res["mesh"]
        self.viewer.set_overlay("preview", V, F, color=(0.2, 0.85, 0.45), opacity=0.55)
        txt = f.summary()
        if res["info"].get("alternatives"):
            alt = ", ".join(f"{KIND_LABELS[k]} {v:.4f}" for k, v in sorted(res["info"]["alternatives"].items(), key=lambda t: t[1]))
            txt += "\n" + tr("RMS aller Kandidaten: ", "RMS of all candidates: ") + alt
        if res["info"].get("note"):
            txt += "\n" + res["info"]["note"]
        if res["dev"] is not None:
            d = res["dev"]
            txt += "\n" + tr("Abstand Scan→Modell", "Distance scan→model") + \
                f": RMS {np.sqrt(np.mean(d ** 2)):.4f}, P95 {np.percentile(np.abs(d), 95):.4f}, max {np.abs(d).max():.4f} mm"
        self.fit_result.setPlainText(txt)
        self._show_preview_dev()
        self._update_enabled()

    def _show_preview_dev(self, *_):
        if self.preview is None or self.preview.get("dev") is None or not self.chk_fit_dev.isChecked():
            self.viewer.hide_scalars(render=False)
            self.viewer.overlay_visible("preview", True)
            self.viewer.render()
            return
        self.viewer.overlay_visible("preview", False)
        ids = self.preview["ids"]
        V, F, _ = self.model.submesh(ids)
        tol = self.fit_tol.value()
        self.viewer.show_cell_scalars(V, F, self.preview["dev"], (-3 * tol, 3 * tol))

    def grow_fit(self):
        if self.preview is None:
            self._msg(tr("Zuerst fitten (Leertaste).", "Fit first (space)."))
            return
        f = self.preview["feature"]
        if f.kind not in ("plane", "sphere", "cylinder", "cone", "torus", "ellipsoid"):
            self._msg(tr("Erweitern geht nur für Primitive.", "Growing works for primitives only."))
            return
        excl = (self.assigned > 0) if self.brush_protect.isChecked() else None
        ids = grow_selection_to_fit(self.model, f, self.preview["ids"], tol_factor=3.0,
                                    tol=self.fit_tol.value(), angle_deg=25, exclude=excl)
        self.checkpoint(tr("Auswahl erweitern", "Grow selection"))
        self.sel[:] = False
        self.sel[ids] = True
        self.recolor()
        self.fit_kind.setCurrentIndex([k for k, _ in fit_kinds()].index(f.kind))
        QTimer.singleShot(0, self.do_fit)

    def accept_fit(self):
        if self.preview is None:
            return
        res = self.preview
        f = res["feature"]
        self.checkpoint(tr(f"Feature „{f.name}“ übernehmen", f"Accept feature “{f.name}”"))
        self.features.append(f)
        self.feature_meshes[self._geom_key(f) + (round(float(self._disp_defl()), 5),)] = res["mesh"]
        self.preview = None
        self.viewer.remove_overlay("preview", render=False)
        self.viewer.hide_scalars(render=False)
        self.sel[:] = False
        self._rebuild_assigned()
        self._sync_feature_overlays()
        self.recolor()
        self.refresh_tree(select_id=f.id)
        self.fit_result.setPlainText(tr(f"Übernommen: {f.name}", f"Accepted: {f.name}"))
        self._update_enabled()

    def discard_fit(self, render=True):
        self.preview = None
        self.viewer.remove_overlay("preview", render=False)
        self.viewer.hide_scalars(render=render)
        if hasattr(self, "btn_accept"):
            self._update_enabled()
