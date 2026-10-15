"""Seite 1 · Scan: prüfen, reparieren, Löcher/Boden, Füllungen bearbeiten, korrigieren, vereinfachen."""
from __future__ import annotations

import numpy as np
from qtpy.QtWidgets import (QCheckBox, QFormLayout, QHBoxLayout, QLabel, QListWidget, QListWidgetItem,
                            QMessageBox, QVBoxLayout, QWidget)

from core import mesh_ops as mo
from core import preprocess as pp
from core.features import fixed_plane_feature
from core.geom_utils import pca_frame, unit
from core.holes import fill_hole, find_holes
from core.i18n import tr

from .common import Section, button, combo, dspin, hint, ispin, row


def fill_modes():
    return [("smooth", tr("glatt (setzt Krümmung fort)", "smooth (continues curvature)")),
            ("tight", tr("straff (Membran)", "tight (membrane)")),
            ("flat", tr("eben (Ausgleichsebene)", "flat (best-fit plane)"))]


def _plane_from_points(P, mode, value, ref, n_out):
    """wie mesh_ops.plane_from_rims, aber aus gespeicherten Randpunkten."""
    n_out = unit(n_out)
    if mode == "z":
        n = np.array([0, 0, 1.0]) * (1 if n_out[2] >= 0 else -1)
        return np.array([0, 0, float(value)]), n
    if mode in ("offset", "plane"):
        o, n = np.asarray(ref[0], float), unit(ref[1])
        if n @ n_out < 0:
            n = -n
        if mode == "plane":
            return o, n
        side = 1.0 if ((P.mean(0) - o) @ n) >= 0 else -1.0
        return o + side * float(value) * n, n
    c, ev, evec = pca_frame(P)
    n = evec[:, 0] if ev[0] < ev[-1] else evec[:, 2]
    n = unit(n)
    if n @ n_out < 0:
        n = -n
    if mode == "lowest":
        c = c + n * float(((P - c) @ n).max())
    return c, n


class ScanPage:
    # ------------------------------------------------------------------ Aufbau
    def _page_scan(self):
        w = QWidget()
        v = QVBoxLayout(w)
        v.addLayout(row(button(tr("Netz öffnen…", "Open mesh…"), lambda: self.open_mesh()),
                        button(tr("Projekt öffnen…", "Open project…"), lambda: self.load_project())))
        self.lbl_info = QLabel(tr("Kein Netz geladen.", "No mesh loaded."))
        self.lbl_info.setWordWrap(True)
        v.addWidget(self.lbl_info)

        s = Section(tr("Prüfen", "Check"))
        s.add(button(tr("Prüfen: Rauschen, Löcher, Topologie", "Check: noise, holes, topology"), self.do_check,
                     tr("Schätzt das Scanrauschen (lokale Quadrik-Fits) und leitet Toleranzen ab.",
                        "Estimates scan noise (local quadric fits) and derives tolerances.")))
        self.lbl_check = QLabel("")
        self.lbl_check.setWordWrap(True)
        s.add(self.lbl_check)
        v.addWidget(s)

        s = Section(tr("Reparieren", "Repair"), expanded=False)
        f = QFormLayout()
        self.rep_merge = dspin(0.0, 0, 1, 0.001, 4, "mm")
        self.rep_minfrac = dspin(1.0, 0, 100, 0.5, 1, "%")
        self.rep_largest = QCheckBox(tr("nur größtes Teilstück behalten", "keep largest piece only"))
        f.addRow(tr("Punkte verschmelzen", "Merge points"), self.rep_merge)
        f.addRow(tr("Teilstücke < (rel.)", "Pieces < (rel.)"), self.rep_minfrac)
        f.addRow(self.rep_largest)
        s.add(f)
        s.add(row(button(tr("Reparieren", "Repair"), self.do_repair),
                  button(tr("Mannigfaltig machen", "Make manifold"), self.do_manifold,
                         tr("Doppelte/gefaltete Dreiecke, Kanten mit >2 Dreiecken, gekippte Normalen",
                            "Duplicate/folded triangles, edges with >2 triangles, flipped normals"))))
        self.spike_k = dspin(4.0, 1.5, 20, 0.5, 1, "σ")
        s.add(row(button(tr("Ausreißer-Spitzen glätten", "Smooth outlier spikes"), self.do_spikes), self.spike_k))
        v.addWidget(s)

        s = Section(tr("Löcher && Boden", "Holes && base"))
        self.chk_rims = QCheckBox(tr("Ränder rot anzeigen", "Show rims in red"))
        self.chk_rims.toggled.connect(self._show_rims)
        s.add(row(self.chk_rims, button(tr("Loch anklicken (H)", "Pick hole (H)"), lambda: self._set_tool("hole"))))
        self.hole_list = QListWidget()
        self.hole_list.setMaximumHeight(120)
        self.hole_list.currentRowChanged.connect(self._hole_list_changed)
        s.add(self.hole_list)
        self.hole_mode = combo(fill_modes())
        f = QFormLayout()
        f.addRow(tr("Füllart", "Fill mode"), self.hole_mode)
        s.add(f)
        self.hole_max = dspin(5.0, 0.1, 1000, 0.5, 1, "mm")
        s.add(row(button(tr("Gewähltes füllen", "Fill selected"), self.fill_selected_hole),
                  button(tr("Alle bis Ø", "All up to Ø"), self.fill_holes_upto), self.hole_max))
        s.add(hint(tr("Ø = flächengleicher Durchmesser. Neue Dreiecke sind künstlich (violett), zählen beim "
                      "Fitten nur schwach und bleiben unter „Füllungen“ bearbeitbar.",
                      "Ø = area-equivalent diameter. New triangles are synthetic (violet), count only weakly "
                      "in fits and stay editable under “Fills”.")))
        g = QWidget()
        gf = QFormLayout(g)
        gf.setContentsMargins(0, 6, 0, 0)
        gf.addRow(QLabel("<b>" + tr("Offener Boden (gewählter Rand = außen)", "Open base (selected rim = outer)") + "</b>"))
        self.base_mode = combo([("fit", tr("Ausgleichsebene des Rands", "best-fit plane of the rim")),
                                ("lowest", tr("am äußersten Randpunkt", "at the outermost rim point")),
                                ("z", tr("Ebene z = Wert", "plane z = value")),
                                ("offset", tr("Abstand zu Bezugsebene", "distance to reference plane")),
                                ("plane", tr("genau die Bezugsebene", "exactly the reference plane"))])
        self.base_value = dspin(0.0, -1e4, 1e4, 0.1, 3, "mm")
        self.base_ref = combo([])
        self.base_extend = QCheckBox(tr("Wände bis zur Ebene verlängern (tangential)", "extend walls to the plane (tangential)"))
        self.base_extend.setChecked(True)
        self.base_inner = QCheckBox(tr("innere Ränder nahe der Ebene mitnehmen (Bohrungen)", "include inner rims near the plane (holes)"))
        self.base_inner.setChecked(True)
        gf.addRow(tr("Ebene", "Plane"), self.base_mode)
        gf.addRow(tr("Wert", "Value"), self.base_value)
        gf.addRow(tr("Bezugsebene", "Reference plane"), self.base_ref)
        gf.addRow(self.base_extend)
        gf.addRow(self.base_inner)
        gf.addRow(row(button(tr("Boden schließen", "Close base"), self.close_base_ui, bold=True),
                      button(tr("Boden-Ebene ändern", "Change base plane"), self.change_base_plane,
                             tr("Für einen schon geschlossenen Boden (in „Füllungen“ gewählt oder der erste)",
                                "For an already closed base (selected under “Fills” or the first one)"))))
        gf.addRow(hint(tr("Die Wände werden entlang ihrer Tangente bis zur Ebene extrapoliert. Der Boden wird eine "
                          "festgelegte Ebene ohne Messdaten: sie wird nie neu gefittet, kann aber per Bedingung "
                          "(z. B. Abstand 10 mm zur Deckfläche) gesetzt werden.",
                          "Walls are extrapolated along their tangent to the plane. The floor becomes a fixed plane "
                          "without scan data: it is never refitted, but can be set by a constraint (e.g. distance "
                          "10 mm to the top face).")))
        s.add(g)
        v.addWidget(s)

        s = Section(tr("Füllungen nachbearbeiten", "Edit fills"), expanded=False)
        self.fill_list = QListWidget()
        self.fill_list.setMaximumHeight(130)
        self.fill_list.currentRowChanged.connect(self._fill_list_changed)
        s.add(self.fill_list)
        self.refill_mode = combo(fill_modes())
        s.add(row(button(tr("Neu füllen", "Refill"), self.refill_ui), self.refill_mode))
        self.force_src = combo([])
        s.add(row(button(tr("Als Ebene erzwingen", "Force plane"), self.force_plane_ui, bold=True), self.force_src))
        s.add(row(button(tr("Fairen (Falten glätten)", "Fair (smooth folds)"), self.fair_fill_ui),
                  button(tr("Entfernen (Loch wieder offen)", "Remove (reopen hole)"), self.remove_fill_ui),
                  button(tr("Auswählen", "Select"), self.select_fill_ui)))
        s.add(hint(tr("Membranfüllungen erzeugen Riffel, die als viele kleine Ebenen erkannt würden: hier fairen "
                      "oder als Ebene erzwingen (Ausgleichsebene des Rands bzw. koplanar zu einem Feature). Eine "
                      "erzwungene Ebene wird eine festgelegte Ebene (wie ein Boden).",
                      "Membrane fills create ripples that would be detected as many small planes: fair them here or "
                      "force a plane (best-fit plane of the rim or coplanar with a feature). A forced plane becomes a "
                      "fixed plane (like a floor).")))
        v.addWidget(s)

        s = Section(tr("Korrigieren (Auswahl)", "Correct (selection)"), expanded=False)
        self.cor_iter = ispin(15, 1, 500)
        self.cor_feat = combo([])
        self.cor_mode = combo(fill_modes())
        f = QFormLayout()
        f.addRow(tr("Iterationen", "Iterations"), self.cor_iter)
        s.add(f)
        s.add(row(button(tr("Glätten", "Smooth"), lambda: self.region_op("taubin"),
                         tr("Volumenerhaltende Glättung (Rauschen, Riffel)", "Volume-preserving smoothing (noise, ripples)")),
                  button(tr("Fairen", "Fair"), lambda: self.region_op("fair"),
                         tr("Glatteste Fläche mit festem Rand (Falten, Beulen, Murks)",
                            "Smoothest surface with fixed boundary (folds, dents, junk)"))))
        s.add(row(button(tr("Auf Feature projizieren", "Project onto feature"), lambda: self.region_op("project")),
                  self.cor_feat))
        s.add(row(button(tr("Löschen + neu füllen", "Delete + refill"), lambda: self.region_op("refill")), self.cor_mode))
        s.add(hint(tr("Typische Scan-Fehler: Murks in Bohrungen → löschen + füllen oder gleich als Bohrung "
                      "definieren; Reflexionsspitzen → glätten; flache Bereiche mit Rauschen → auf Ebene "
                      "projizieren.",
                      "Typical scan errors: junk in holes → delete + fill or define a hole right away; "
                      "reflection spikes → smooth; noisy flat areas → project onto plane.")))
        v.addWidget(s)

        s = Section(tr("Vereinfachen", "Simplify"), expanded=False)
        f = QFormLayout()
        self.simp_tol = dspin(0.05, 0.001, 5.0, 0.005, 3, "mm")
        self.simp_max = ispin(0, 0, 50_000_000, 10000)
        self.simp_max.setSpecialValueText(tr("unbegrenzt", "unlimited"))
        f.addRow(tr("max. Abweichung", "max. deviation"), self.simp_tol)
        f.addRow(tr("max. Dreiecke", "max. triangles"), self.simp_max)
        s.add(f)
        self.chk_dev = QCheckBox(tr("Abweichung zum Original anzeigen", "Show deviation to original"))
        self.chk_dev.toggled.connect(self._toggle_simplify_dev)
        s.add(row(button(tr("Vereinfachen", "Simplify"), self.do_simplify), self.chk_dev))
        v.addWidget(s)
        v.addStretch(1)
        self._holes = None
        self._hole_sel = None
        self._simp_dev = None
        return w

    # ------------------------------------------------------------------ Prüfen / Reparieren
    def _suggest_params(self):
        m = self.model
        if m is None:
            return
        e = m.mean_edge_length()
        self.brush_spin.setValue(round(max(0.1, min(m.diag * 0.03, 10 * e)), 2))
        self.hole_max.setValue(round(max(1.0, 0.03 * m.diag), 1))
        if self.doc.tol:
            self._apply_tol(self.doc.tol)

    def _apply_tol(self, ns):
        self.fit_tol.setValue(round(ns["fit_tol"], 3))
        self.det_eps.setValue(round(ns["detect_eps"], 3))
        self.simp_tol.setValue(round(ns["simplify_tol"], 3))

    def do_check(self):
        if not self._need_mesh():
            return
        m = self.model

        def work():
            ns = mo.estimate_noise(m)
            rep = mo.manifold_report(m)
            holes = find_holes(m)
            return ns, rep, holes

        def done(r):
            ns, rep, holes = r
            self.doc.tol = ns
            self._apply_tol(ns)
            big = max((h["diameter"] for h in holes), default=0)
            txt = tr(f"Rauschen σ ≈ <b>{ns['sigma']:.4f} mm</b> → Fit-Toleranz {ns['fit_tol']:.3f}, "
                     f"Erkennung {ns['detect_eps']:.3f}, Vereinfachen {ns['simplify_tol']:.3f} mm (übernommen)<br>",
                     f"Noise σ ≈ <b>{ns['sigma']:.4f} mm</b> → fit tolerance {ns['fit_tol']:.3f}, "
                     f"detection {ns['detect_eps']:.3f}, simplify {ns['simplify_tol']:.3f} mm (applied)<br>")
            txt += tr(f"{len(holes)} Löcher (größtes Ø ≈ {big:.1f} mm), {rep['components']} Teil(e), "
                      f"{rep['nonmanifold_edges']} nicht-mannigfaltige, {rep['flipped_edges']} gekippte Kanten",
                      f"{len(holes)} holes (largest Ø ≈ {big:.1f} mm), {rep['components']} part(s), "
                      f"{rep['nonmanifold_edges']} non-manifold, {rep['flipped_edges']} flipped edges")
            if rep["nonmanifold_edges"] or rep["flipped_edges"]:
                txt += "<br>→ " + tr("„Mannigfaltig machen“ empfohlen", "“Make manifold” recommended")
            self.lbl_check.setText(txt)
        self._run(tr("Prüfe Scan…", "Checking scan…"), work, done)

    def do_repair(self):
        args = dict(merge_tol=self.rep_merge.value(), min_component_frac=self.rep_minfrac.value() / 100,
                    keep_largest=self.rep_largest.isChecked())

        def fn(d, rep):
            m, r = pp.repair(d.model, **args)
            d.replace_model(m)
            return r
        self.run_doc(tr("Reparieren", "Repair"), fn)

    def do_manifold(self):
        def fn(d, rep):
            m, r, fmap = mo.repair_manifold(d.model)
            d.replace_model(m, face_map=fmap)
            return r
        self.run_doc(tr("Mannigfaltig machen", "Make manifold"), fn)

    def do_spikes(self):
        k = self.spike_k.value()

        def fn(d, rep):
            sigma = float(d.tol.get("sigma") or mo.estimate_noise(d.model)["sigma"])
            m, n = mo.remove_spikes(d.model, sigma, k)
            d.replace_model(m)
            return [tr(f"{n} Spitzen geglättet", f"{n} spikes smoothed")]
        self.run_doc(tr("Spitzen glätten", "Smooth spikes"), fn)

    # ------------------------------------------------------------------ Löcher
    def _refresh_holes(self):
        self.hole_list.blockSignals(True)
        self.hole_list.clear()
        if self.model is None:
            self.hole_list.blockSignals(False)
            return
        if self._holes is None:
            try:
                self._holes = find_holes(self.model)[::-1]
            except Exception:
                self._holes = []
        for h in self._holes:
            self.hole_list.addItem(QListWidgetItem(f"Ø ≈ {h['diameter']:.2f} mm   ({h['n']} " + tr("Randkanten", "rim edges") + ")"))
        self.hole_list.blockSignals(False)
        self._hole_sel = None
        self.viewer.remove_overlay("__hole_sel", render=False)
        self._show_rims(self.chk_rims.isChecked())
        self._refresh_base_refs()

    def _refresh_base_refs(self):
        planes = [f for f in self.features if f.kind == "plane"]
        for cb in (self.base_ref, self.force_src):
            cur = cb.currentData()
            cb.blockSignals(True)
            cb.clear()
            if cb is self.force_src:
                cb.addItem(tr("Ausgleichsebene des Rands", "best-fit plane of the rim"), None)
            for f in planes:
                cb.addItem(f.name, f.id)
            i = cb.findData(cur)
            cb.setCurrentIndex(i if i >= 0 else 0)
            cb.blockSignals(False)
        cur = self.cor_feat.currentData()
        self.cor_feat.clear()
        for f in self.features:
            if f.kind in ("plane", "cylinder", "cone", "sphere", "torus", "freeform", "ellipsoid", "hole",
                          "extrusion", "revolve"):
                self.cor_feat.addItem(f.name, f.id)
        i = self.cor_feat.findData(cur)
        if i >= 0:
            self.cor_feat.setCurrentIndex(i)

    def _show_rims(self, on):
        if self.model is None:
            return
        if on:
            if self._holes is None:
                self._refresh_holes()
                return
            self.viewer.set_lines("__rims", [self.model.V[np.r_[h["loop"], h["loop"][:1]]] for h in self._holes],
                                  color="#ff3b30", width=4)
        else:
            self.viewer.remove_overlay("__rims")

    def _hole_list_changed(self, r):
        if self._holes is None or r < 0 or r >= len(self._holes):
            return
        self._select_hole(r)

    def _select_hole(self, i):
        h = self._holes[i]
        self._hole_sel = i
        self.viewer.set_lines("__hole_sel", [self.model.V[np.r_[h["loop"], h["loop"][:1]]]], color="#ffd60a", width=5)
        if self.hole_list.currentRow() != i:
            self.hole_list.blockSignals(True)
            self.hole_list.setCurrentRow(i)
            self.hole_list.blockSignals(False)
        self._msg(tr(f"Loch gewählt: Ø ≈ {h['diameter']:.2f} mm", f"Hole selected: Ø ≈ {h['diameter']:.2f} mm"))

    def _pick_hole(self, pos):
        if self._holes is None:
            self._refresh_holes()
        if not self._holes:
            self._msg(tr("Keine Löcher (offene Ränder) im Netz.", "No holes (open rims) in the mesh."))
            return
        best = min(range(len(self._holes)),
                   key=lambda i: np.min(np.linalg.norm(self.model.V[self._holes[i]["loop"]] - pos, axis=1)))
        self._select_hole(best)

    def fill_selected_hole(self):
        if self._hole_sel is None or self._holes is None:
            self._msg(tr("Erst ein Loch wählen (Liste oder H + Klick an den Rand).",
                         "Select a hole first (list or H + click on the rim)."))
            return
        loop = self._holes[self._hole_sel]["loop"]
        mode = self.hole_mode.currentData()

        def fn(d, rep):
            m, new = fill_hole(d.model, loop, mode)
            d.replace_model(m)
            return [tr(f"Loch gefüllt: {len(new)} Dreiecke", f"hole filled: {len(new)} triangles")]
        self.run_doc(tr("Loch füllen", "Fill hole"), fn)

    def fill_holes_upto(self):
        if not self._need_mesh():
            return
        if self._holes is None:
            self._refresh_holes()
        dmax = self.hole_max.value()
        todo = [h["loop"] for h in (self._holes or []) if h["diameter"] <= dmax]
        if not todo:
            self._msg(tr(f"Keine Löcher bis Ø {dmax} mm.", f"No holes up to Ø {dmax} mm."))
            return
        mode = self.hole_mode.currentData()

        def fn(d, rep):
            m = d.model
            n = 0
            for loop in todo:
                try:
                    m, _ = fill_hole(m, loop, mode)
                    n += 1
                except Exception:
                    continue
            d.replace_model(m)
            return [tr(f"{n} Löcher bis Ø {dmax} mm gefüllt", f"{n} holes up to Ø {dmax} mm filled")]
        self.run_doc(tr("Löcher füllen", "Fill holes"), fn)

    # ------------------------------------------------------------------ Boden
    def _base_ref_plane(self):
        fid = self.base_ref.currentData()
        f = self.doc.feature(fid) if fid is not None else None
        if f is None:
            return None
        return np.asarray(f.params["origin"], float), unit(f.params["normal"])

    def close_base_ui(self):
        if self._hole_sel is None or self._holes is None:
            # Standard: größter Rand
            if self._holes:
                self._select_hole(0)
            else:
                self._msg(tr("Kein offener Rand vorhanden.", "No open rim."))
                return
        mode = self.base_mode.currentData()
        ref = self._base_ref_plane()
        if mode in ("offset", "plane") and ref is None:
            self._msg(tr("Für diese Art eine Bezugsebene (Ebenen-Feature) wählen.", "Choose a reference plane feature for this mode."))
            return
        outer = self._holes[self._hole_sel]
        others = [h for i, h in enumerate(self._holes) if i != self._hole_sel]
        value = self.base_value.value()
        extend = self.base_extend.isChecked()
        inner = self.base_inner.isChecked()

        def fn(d, rep):
            m = d.model
            plane = mo.plane_from_rims(m, [outer["loop"]], mode, value, ref)
            loops = [outer["loop"]]
            if inner:
                loops += mo.rims_near_plane(m, others, plane, max_dist=max(1.0, 0.02 * m.diag))
            m2, s_ids, c_ids, r = mo.close_base(m, loops, plane, extend=extend)
            f = fixed_plane_feature(m2, c_ids, plane, name=tr("Boden", "Floor"), tol=d.fit_tol)
            m2.fills[-1]["feature"] = f.id
            d.replace_model(m2)
            d.features.append(f)
            return r
        self.run_doc(tr("Boden schließen", "Close base"), fn)

    def _selected_fill(self):
        r = self.fill_list.currentRow()
        fills = self.model.fills if self.model is not None else []
        if 0 <= r < len(fills):
            return fills[r]
        return None

    def change_base_plane(self):
        if not self._need_mesh():
            return
        fl = self._selected_fill()
        if fl is None or fl.get("kind") != "base":
            fl = next((x for x in self.model.fills if x.get("kind") == "base"), None)
        if fl is None:
            self._msg(tr("Noch kein Boden geschlossen.", "No base closed yet."))
            return
        mode = self.base_mode.currentData()
        ref = self._base_ref_plane()
        if mode in ("offset", "plane") and ref is None:
            self._msg(tr("Für diese Art eine Bezugsebene wählen.", "Choose a reference plane for this mode."))
            return
        P = np.vstack([np.asarray(r) for r in fl["rim"]])
        plane = _plane_from_points(P, mode, self.base_value.value(), ref, fl["plane"][1])
        self._refill_with_plane(fl, plane, tr("Boden-Ebene ändern", "Change base plane"))

    def _refill_with_plane(self, fl, plane, label):
        fid = fl["id"]
        feat_id = fl.get("feature")

        def fn(d, rep):
            m3, fmap, new_ids, cap_ids = mo.refill(d.model, fid, "plane", plane=plane)
            d.replace_model(m3, face_map=np.r_[fmap, -np.ones(0, np.int64)])
            f = d.feature(feat_id) if feat_id is not None else None
            if f is not None:
                nf = fixed_plane_feature(m3, cap_ids, plane, name=f.name, tol=d.fit_tol)
                f.params = nf.params
                f.face_ids = np.asarray(cap_ids, np.int64)
                f.invalidate()
            else:
                f = fixed_plane_feature(m3, cap_ids, plane, name=tr("Füllebene", "Fill plane"), tol=d.fit_tol)
                f.name = f"{f.name} {f.id}"
                d.features.append(f)
            m3.fills[-1]["feature"] = f.id
            return [tr(f"{len(new_ids)} Dreiecke neu, Ebene festgelegt ({f.name})",
                       f"{len(new_ids)} new triangles, plane fixed ({f.name})")]
        self.run_doc(label, fn)

    # ------------------------------------------------------------------ Füllungen
    def refresh_fills(self):
        if not hasattr(self, "fill_list"):
            return
        cur = self.fill_list.currentRow()
        self.fill_list.blockSignals(True)
        self.fill_list.clear()
        names = {"hole": tr("Loch", "hole"), "base": tr("Boden", "base"), "replace": tr("Ersetzt", "replaced")}
        modes = dict(fill_modes())
        modes["plane"] = tr("Ebene (festgelegt)", "plane (fixed)")
        for fl in (self.model.fills if self.model is not None else []):
            lab = fl.get("label") or names.get(fl.get("kind"), fl.get("kind"))
            self.fill_list.addItem(f"#{fl['id']} {lab} – {modes.get(fl.get('mode'), fl.get('mode'))}, "
                                   f"{len(fl['faces'])} " + tr("Dreiecke", "triangles"))
        if 0 <= cur < self.fill_list.count():
            self.fill_list.setCurrentRow(cur)
        self.fill_list.blockSignals(False)
        self._refresh_base_refs()

    def _fill_list_changed(self, r):
        fl = self._selected_fill()
        if fl is None:
            self.viewer.remove_overlay("__fill_rim")
            return
        self.viewer.set_lines("__fill_rim", [np.vstack([R, R[:1]]) for R in (np.asarray(x) for x in fl["rim"])],
                              color="#c084fc", width=5)

    def select_fill_ui(self):
        fl = self._selected_fill()
        if fl is None:
            return
        self.checkpoint(tr("Auswahl aus Füllung", "Selection from fill"))
        self.sel[:] = False
        self.sel[np.asarray(fl["faces"], np.int64)] = True
        self.recolor()

    def refill_ui(self):
        fl = self._selected_fill()
        if fl is None:
            self._msg(tr("Erst eine Füllung wählen.", "Select a fill first."))
            return
        mode = self.refill_mode.currentData()
        fid = fl["id"]

        def fn(d, rep):
            m3, fmap, new_ids, _ = mo.refill(d.model, fid, mode)
            d.replace_model(m3, face_map=fmap)
            return [tr(f"neu gefüllt: {len(new_ids)} Dreiecke", f"refilled: {len(new_ids)} triangles")]
        self.run_doc(tr("Neu füllen", "Refill"), fn)

    def force_plane_ui(self):
        fl = self._selected_fill()
        if fl is None:
            self._msg(tr("Erst eine Füllung wählen.", "Select a fill first."))
            return
        src = self.force_src.currentData()
        P = np.vstack([np.asarray(r) for r in fl["rim"]])
        if src is None:
            c, ev, evec = pca_frame(P)
            n = unit(evec[:, 0] if ev[0] < ev[-1] else evec[:, 2])
            # Normale nach außen: wie die Füllung bisher
            fn_ = self.model.normals[np.asarray(fl["faces"], np.int64)].mean(0) if len(fl["faces"]) else n
            plane = (c, n if n @ fn_ >= 0 else -n)
        else:
            f = self.doc.feature(src)
            n = unit(f.params["normal"])
            fn_ = self.model.normals[np.asarray(fl["faces"], np.int64)].mean(0) if len(fl["faces"]) else n
            plane = (np.asarray(f.params["origin"], float), n if n @ fn_ >= 0 else -n)
        self._refill_with_plane(fl, plane, tr("Als Ebene erzwingen", "Force plane"))

    def fair_fill_ui(self):
        fl = self._selected_fill()
        if fl is None:
            return
        fid = fl["id"]

        def fn(d, rep):
            d.replace_model(mo.fair_fill(d.model, fid))
            return [tr("Füllung gefairt", "fill faired")]
        self.run_doc(tr("Füllung fairen", "Fair fill"), fn)

    def remove_fill_ui(self):
        fl = self._selected_fill()
        if fl is None:
            return
        fid = fl["id"]
        feat_id = fl.get("feature")

        def fn(d, rep):
            m2, fmap, _ = mo.remove_fill(d.model, fid)
            d.replace_model(m2, face_map=fmap)
            if feat_id is not None:
                d.features = [f for f in d.features if f.id != feat_id]
            return [tr("Füllung entfernt – der Rand ist wieder offen", "fill removed – the rim is open again")]
        self.run_doc(tr("Füllung entfernen", "Remove fill"), fn)

    # ------------------------------------------------------------------ Korrigieren
    def region_op(self, op):
        if not self._need_mesh():
            return
        ids = np.nonzero(self.sel)[0]
        if len(ids) < 3:
            self._msg(tr("Erst einen Bereich auswählen (Pinsel B).", "Select an area first (brush B)."))
            return
        it = self.cor_iter.value()
        fid = self.cor_feat.currentData()
        mode = self.cor_mode.currentData()

        def fn(d, rep):
            if op in ("taubin", "fair"):
                d.replace_model(mo.smooth_region(d.model, ids, it, op))
                return [tr("Bereich geglättet", "area smoothed")]
            if op == "project":
                f = d.feature(fid)
                if f is None:
                    raise RuntimeError(tr("Kein Feature gewählt.", "No feature chosen."))
                d.replace_model(mo.project_region(d.model, ids, f))
                return [tr(f"Bereich auf {f.name} projiziert", f"area projected onto {f.name}")]
            m3, fmap, nl = mo.delete_and_fill(d.model, ids, mode)
            d.replace_model(m3, face_map=fmap)
            return [tr(f"{len(ids)} Dreiecke gelöscht, {nl} Löcher neu gefüllt", f"{len(ids)} triangles deleted, {nl} holes refilled")]
        self.run_doc(tr("Korrigieren", "Correct"), fn)

    # ------------------------------------------------------------------ Vereinfachen
    def do_simplify(self):
        if not self._need_mesh():
            return
        tol = self.simp_tol.value()
        mx = self.simp_max.value() or None
        holder = {}

        def fn(d, rep):
            orig = d.model
            m2, info = pp.simplify_to_tolerance(orig, tol, max_faces=mx)
            holder["dev"] = pp.deviation(m2.V, orig, max_points=10 ** 9)[0]
            d.replace_model(m2)
            holder["info"] = info
            return [tr(f"{info.get('faces_before', 0):,} → {info['faces']:,} Dreiecke; P99.9 Abw. {info['dev_max']:.4f} mm",
                       f"{info.get('faces_before', 0):,} → {info['faces']:,} triangles; P99.9 dev. {info['dev_max']:.4f} mm").replace(",", ".")]

        def after():
            self._simp_dev = holder.get("dev")
            if holder.get("info", {}).get("reduction", 1) < 0.05:
                QMessageBox.information(self, tr("Vereinfachen", "Simplify"),
                                        tr("Kaum Reduktion möglich: Toleranz liegt im Bereich des Rauschens.",
                                           "Hardly any reduction possible: tolerance is within the noise level."))
        self.run_doc(tr("Vereinfachen", "Simplify"), fn, after=after)

    def _toggle_simplify_dev(self, on):
        if on and self._simp_dev is not None and self.model is not None and len(self._simp_dev) == self.model.n_verts:
            tol = self.simp_tol.value()
            self.viewer.show_point_scalars(self.model.V, self.model.F, self._simp_dev, (-tol, tol))
            self.viewer.set_mesh_visible(False)
        else:
            self.viewer.hide_scalars()
            self.viewer.set_mesh_visible(self.act_mesh_vis.isChecked())
