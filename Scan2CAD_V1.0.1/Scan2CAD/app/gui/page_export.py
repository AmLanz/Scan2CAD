"""Seite 7 · Export: Scan-Topologie (geschlossener Volumenkörper), getrimmte Flächen oder Körper/Booleans;
STEP-Vorschau, weitere Exporte."""
from __future__ import annotations

import os

import numpy as np
from qtpy.QtWidgets import (QButtonGroup, QCheckBox, QFileDialog, QFormLayout, QLabel, QMessageBox,
                            QPlainTextEdit, QRadioButton, QVBoxLayout, QWidget)

from core import export as exp
from core import occ_build as ob
from core import preprocess as pp
from core.i18n import tr

from .common import Section, button, combo, dspin, hint, row


class ExportPage:
    def _page_export(self):
        w = QWidget()
        v = QVBoxLayout(w)
        s = Section(tr("Art", "Mode"))
        self.ex_mode_group = QButtonGroup(self)
        self.ex_modes = {}
        for key, label, tip in (
                ("scan", tr("Scan-Topologie: geschlossener Volumenkörper (empfohlen)", "Scan topology: closed solid (recommended)"),
                 tr("Nachbarschaften der Flächen aus dem Scan; Kanten = exakte Schnitt-/Berührkurven; Bohrungen & "
                    "Körper per Boolean. Braucht ein geschlossenes Netz.",
                    "Face adjacency from the scan; edges = exact intersection/tangency curves; holes & solids by "
                    "boolean. Needs a closed mesh.")),
                ("trim", tr("Getrimmte Flächen (verschnitten)", "Trimmed surfaces (intersected)"),
                 tr("Flächen werden verlängert und gegeneinander verschnitten (General Fuse).",
                    "Surfaces are extended and split against each other (General Fuse).")),
                ("bool", tr("Körper / Booleans + einzelne Flächen", "Solids / booleans + individual surfaces"),
                 tr("Körper-Features vereinigen/abziehen, Flächen-Features einzeln.",
                    "Fuse/subtract solid features, surface features individually."))):
            rb = QRadioButton(label)
            rb.setToolTip(tip)
            self.ex_mode_group.addButton(rb)
            self.ex_modes[key] = rb
            s.add(rb)
        self.ex_modes["scan"].setChecked(True)
        v.addWidget(s)

        s = Section(tr("Optionen Scan-Topologie", "Scan topology options"), expanded=False)
        f = QFormLayout()
        self.ex_fillet = QCheckBox(tr("erkannte Kantenrundungen verrunden", "fillet detected edge blends"))
        self.ex_fillet.setToolTip(tr("Kantenrundungen aus dem Scan (Schritt „Kantenrundungen und Splitter“) werden "
                                     "zunächst als scharfe Kanten gebaut. Hier als echte Verrundung zwischen den "
                                     "beiden Ebenen übernehmen – ab dem angegebenen Radius (kleinere sind meist nur "
                                     "Scan-Glättung).",
                                     "Edge blends from the scan (step “edge blends and fragments”) are first built as "
                                     "sharp edges. Apply them here as real fillets between the two planes – from the "
                                     "given radius (smaller ones are usually just scan smoothing)."))
        self.ex_fillet_r = dspin(0.5, 0.0, 50, 0.1, 2, "mm")
        f.addRow(self.ex_fillet)
        f.addRow(tr("ab Radius", "from radius"), self.ex_fillet_r)
        s.add(f)
        v.addWidget(s)

        s = Section(tr("Optionen für „getrimmt“ / „Körper“", "Options for “trimmed” / “solids”"), expanded=False)
        f = QFormLayout()
        self.ex_bool = QCheckBox(tr("Körper vereinigen / abziehen / schneiden", "fuse / subtract / cut solids"))
        self.ex_bool.setChecked(True)
        self.ex_surf = QCheckBox(tr("Flächen-Features exportieren", "export surface features"))
        self.ex_surf.setChecked(True)
        self.ex_redundant = QCheckBox(tr("Flächen weglassen, die auf einem Körper liegen", "drop surfaces lying on a solid"))
        self.ex_redundant.setChecked(True)
        self.ex_trim_ext = dspin(1.0, 0.05, 20, 0.25, 2, "mm")
        self.ex_sew = dspin(0.0, 0, 5, 0.01, 3, "mm")
        self.ex_sew.setSpecialValueText(tr("nicht vernähen", "do not sew"))
        self.ex_rest = QCheckBox(tr("nicht übersetzte Flächen als facettiertes Netz anhängen", "append untranslated areas as faceted mesh"))
        for wdg in (self.ex_bool, self.ex_surf, self.ex_redundant):
            f.addRow(wdg)
        f.addRow(tr("Überlappung zum Verschneiden", "Overlap for trimming"), self.ex_trim_ext)
        f.addRow(tr("Flächen vernähen (ohne Verschneiden)", "Sew surfaces (without trimming)"), self.ex_sew)
        f.addRow(self.ex_rest)
        s.add(f)
        v.addWidget(s)

        s = Section(tr("Ausgabe", "Output"))
        f = QFormLayout()
        self.ex_orig = QCheckBox(tr("in ursprünglicher Scan-Lage exportieren", "export in the original scan pose"))
        self.ex_schema = combo([("AP214IS", "AP214"), ("AP242DIS", "AP242"), ("AP203", "AP203")])
        f.addRow(self.ex_orig)
        f.addRow(tr("Schema", "Schema"), self.ex_schema)
        s.add(f)
        self.ex_color = combo([("src", tr("Herkunft der Flächen", "origin of the faces")),
                               ("dev", tr("Abweichung zum Scan", "deviation from the scan")),
                               ("plain", tr("einfarbig", "plain"))])
        self.ex_color.setToolTip(tr("Herkunft: grau = exakt auf dem Feature (Ebene, Zylinder …), orange = Ersatz-Freiform "
                                    "durch die Scan-Grenze, rot = Dreiecke aus dem Scan.\n"
                                    "Abweichung: Abstand Körper–Scan (rot = Körper außerhalb, blau = innerhalb).\n"
                                    "In jedem Fall prüft die Vorschau jedes Feature gegen den Körper (Protokoll) und "
                                    "markiert Scanbereiche ohne Entsprechung im Körper gelb.",
                                    "Origin: grey = exactly on the feature (plane, cylinder …), orange = substitute "
                                    "freeform through the scan boundary, red = triangles from the scan.\n"
                                    "Deviation: distance solid–scan (red = solid outside, blue = inside).\n"
                                    "Either way the preview checks every feature against the solid (log) and marks "
                                    "scan areas without counterpart in the solid in yellow."))
        f2 = QFormLayout()
        f2.addRow(tr("Vorschau färben", "Preview colours"), self.ex_color)
        s.add(f2)
        self.btn_step_prev = button(tr("STEP-Vorschau (P)", "STEP preview (P)"))
        self.btn_step_prev.setCheckable(True)
        self.btn_step_prev.toggled.connect(self.toggle_step_preview)
        s.add(row(self.btn_step_prev, button(tr("STEP exportieren…", "Export STEP…"), self.export_step, bold=True)))
        self.lbl_closed = QLabel("")
        self.lbl_closed.setWordWrap(True)
        s.add(self.lbl_closed)
        v.addWidget(s)

        s = Section(tr("Weitere Exporte", "Further exports"), expanded=False)
        for label, fn in ((tr("Netz als STL/PLY/OBJ speichern…", "Save mesh as STL/PLY/OBJ…"), self.export_mesh),
                          (tr("Ganzes Netz als facettiertes STEP…", "Whole mesh as faceted STEP…"), self.export_faceted),
                          (tr("Feature-Bericht (JSON)…", "Feature report (JSON)…"), self.export_report),
                          (tr("Projekt speichern…", "Save project…"), self.save_project)):
            s.add(button(label, fn))
        v.addWidget(s)
        self.export_log = QPlainTextEdit()
        self.export_log.setReadOnly(True)
        self.export_log.setMinimumHeight(140)
        v.addWidget(self.export_log)
        v.addWidget(hint(tr("Scan-Topologie: Jedes Dreieck gehört genau einer Fläche; wo Flächen im Scan aneinander "
                            "grenzen, entsteht eine gemeinsame Kante – darum ist der Körper dicht. Freie Streifen an "
                            "Kanten werden den Nachbarn zugeschlagen, unerklärte Bereiche (z. B. Füllungen) bekommen "
                            "Hilfsebenen.", "Scan topology: every triangle belongs to exactly one surface; where "
                            "surfaces meet in the scan, a shared edge is created – hence the solid is watertight. Free "
                            "strips at edges are given to neighbours, unexplained areas (e.g. fills) get auxiliary planes.")))
        v.addStretch(1)
        return w

    def _ex_mode(self):
        return next(k for k, rb in self.ex_modes.items() if rb.isChecked())

    def _rest_mesh(self):
        if not self.ex_rest.isChecked() or self.model is None:
            return None
        free = np.nonzero(self.assigned == 0)[0]
        if len(free) == 0:
            return None
        V, F, _ = self.model.submesh(free)
        return (V, F)

    def _export_kwargs(self):
        return dict(apply_booleans=self.ex_bool.isChecked(), include_surfaces=self.ex_surf.isChecked(),
                    sew_tol=self.ex_sew.value(), rest_mesh=self._rest_mesh(),
                    drop_redundant=self.ex_redundant.isChecked(), trim=(self._ex_mode() == "trim"),
                    scan_model=self.model, trim_ext=self.ex_trim_ext.value(), tol=self.fit_tol.value(),
                    blends=list(self.doc.tol.get("blends") or []),
                    fillet_min_r=self.ex_fillet_r.value() if self.ex_fillet.isChecked() else None)

    def _build_items(self, mode, model, feats, kw, tol, report=None):
        """Im Hintergrund: Liste (Name, Shape, Farbe), Log, Zusatzinfo."""
        log = []
        if mode == "scan":
            res = exp.build_scan_topology(model, feats, tol=tol, log=log.append, progress=report,
                                          blends=kw.get("blends"), fillet_min_r=kw.get("fillet_min_r"))
            return res["items"], log, res
        items = exp.assemble(feats, log=log.append, **kw)
        return items, log, None

    # ------------------------------------------------------------------ Vorschau
    def toggle_step_preview(self, on):
        for wdg in (self.act_step_prev, self.btn_step_prev):
            if wdg.isChecked() != on:
                wdg.blockSignals(True)
                wdg.setChecked(on)
                wdg.blockSignals(False)
        if not on:
            self._exit_step_preview()
            return
        if not self.features or self.model is None:
            self._msg(tr("Noch keine Features – nichts zu zeigen.", "No features yet – nothing to show."))
            self.toggle_step_preview(False)
            return
        mode = self._ex_mode()
        feats = list(self.features)
        kw = self._export_kwargs()
        model = self.model
        tol = self.fit_tol.value()
        defl = self._disp_defl()

        color = self.ex_color.currentData()
        clim = max(5 * tol, 0.2)

        def work():
            items, log, res = self._build_items(mode, model, feats, kw, tol, self.runner.report)
            fid = self._fidelity(items, model, feats, tol, defl, log)
            out = self._tessellate_items(items, kw, defl)
            dev = self._deviation(out, model, log) if color == "dev" else None
            return out, log, res, dev, fid

        def done(r):
            out, log, res, dev, fid = r
            if not self.act_step_prev.isChecked():
                return
            if res is not None:
                self.doc.preview = res
            self._show_preview_items(out, log, res, fid)
            if dev is not None:
                self.viewer.show_point_scalars(dev[0], dev[1], dev[2], (-clim, clim),
                                               tr("Körper–Scan [mm]", "Solid–scan [mm]"))
            elif color == "src" and fid is not None and len(fid["src"]):
                V, F = fid["tess"][0], fid["tess"][1]
                self.viewer.show_cell_scalars(V, F, fid["src"].astype(float), (0, 2), None,
                                              cmap=["#b8c4d0", "#f0a030", "#e0403a"])
                self.lbl_closed.setText(self.lbl_closed.text() + "<br><span style='color:#9aa4ae'>" + tr(
                    "Farben: grau = exakt auf Feature · orange = Ersatz-Freiform · rot = Scan-Dreiecke",
                    "colours: grey = exactly on feature · orange = substitute freeform · red = scan triangles")
                    + "</span>")

        if not self._run(tr("Baue STEP-Vorschau…", "Building STEP preview…"), work, done):
            self.toggle_step_preview(False)

    def _tessellate_items(self, items, kw, defl):
        out, edges = [], []
        for name, shp, rgb in items:
            if name == "Restnetz" and kw.get("rest_mesh") is not None:
                V, F = kw["rest_mesh"]
            else:
                V, F = ob.tessellate(shp, defl)
                edges += ob.edge_polylines(shp, max(defl, 0.02))
            out.append((name, V, F, rgb))
        return out, edges

    @staticmethod
    def _fidelity(items, model, feats, tol, defl, log):
        """Abgleich Features ↔ Körper (core.fidelity); Zeilen ins Protokoll."""
        from core import fidelity as fd
        shapes = [shp for name, shp, _ in items if name != "Restnetz" and shp is not None]
        if not shapes:
            return None
        try:
            body = shapes[0] if len(shapes) == 1 else ob.compound(shapes)
            r = fd.check(model, feats, body, tol, defl=defl)
        except Exception as e:  # noqa: BLE001
            log.append(tr(f"Abgleich Features ↔ STEP fehlgeschlagen: {e}", f"check features ↔ STEP failed: {e}"))
            return None
        log += r["lines"]
        return r

    @staticmethod
    def _deviation(out, model, log):
        """Vorzeichenbehafteter Abstand der Körper-Tessellierung zum Scan (+ = Körper außerhalb)."""
        items, _ = out
        Vs, Fs, off = [], [], 0
        for name, V, F, _ in items:
            if name == "Restnetz" or len(F) == 0:
                continue
            Vs.append(V)
            Fs.append(F + off)
            off += len(V)
        if not Vs:
            return None
        V = np.vstack(Vs)
        F = np.vstack(Fs)
        tgt = pp.vf_to_pd(model.V, model.F)
        import pyvista as pv
        # große ebene Dreiecke verfeinern, sonst wird die Farbe nur zwischen den Ecken interpoliert
        try:
            pd = pp.vf_to_pd(V, F).subdivide_adaptive(max_edge_len=max(0.006 * float(model.diag), 0.3),
                                                      max_n_tris=1_500_000, max_n_passes=12)
            V = np.asarray(pd.points)
            F = np.asarray(pd.regular_faces)
        except Exception:
            pass
        d = np.asarray(pv.PolyData(V).compute_implicit_distance(tgt).point_data["implicit_distance"])
        a = np.abs(d)
        log.append(tr(f"Abweichung Körper–Scan: Median {np.median(a):.3f} mm, P95 {np.percentile(a, 95):.3f} mm, "
                      f"max {a.max():.3f} mm",
                      f"deviation solid–scan: median {np.median(a):.3f} mm, P95 {np.percentile(a, 95):.3f} mm, "
                      f"max {a.max():.3f} mm"))
        return V, F, d

    def show_scan_preview(self, res):
        """Ergebnis der Scan-Topologie (z. B. aus 'Auf gut Glück') anzeigen."""
        defl = self._disp_defl()

        def work():
            return self._tessellate_items(res["items"], {}, defl)

        def done(out):
            for wdg in (self.act_step_prev, self.btn_step_prev):
                wdg.blockSignals(True)
                wdg.setChecked(True)
                wdg.blockSignals(False)
            self._show_preview_items(out, res["report"], res)
        self._run(tr("Tesselliere Vorschau…", "Tessellating preview…"), work, done)

    def _show_preview_items(self, out, log, res, fid=None):
        items, edges = out
        self.step_preview_on = True
        self.viewer.clear_overlays("step_")
        self.viewer.set_mesh_visible(False)
        self.viewer.hide_scalars(render=False)
        self._update_feature_visibility()
        for i, (name, V, F, rgb) in enumerate(items):
            col = (0.62, 0.62, 0.62) if name == "Restnetz" else (rgb or (0.8, 0.75, 0.55))
            self.viewer.set_overlay(f"step_{i}", V, F, color=col, opacity=1.0)
        self.viewer.set_lines("step_edges", edges, color="#1c1c1c", width=1.5)
        if res is not None and res.get("open_edges"):
            self.viewer.set_lines("step_open", res["open_edges"], color="#ff3b30", width=5)
        spots = (fid or {}).get("spots") or []
        if spots:
            C = np.array([sp["center"] for sp in spots[:20]])
            self.viewer.set_points("step_spots", C, color="#ffd60a", size=16)
            self.viewer.set_text("step_spots_lbl", C, [f"{sp['area']:.0f} mm², {sp['max']:.1f} mm" for sp in spots[:20]],
                                 color="#ffd60a")
        self.viewer.set_origin_axes(False)
        self.export_log.setPlainText("\n".join(log))
        if res is not None or fid is not None:
            ok = res.get("solid") if res is not None else None
            txt = ""
            if res is not None:
                txt = ("<span style='color:#3dd68c'><b>✔ " + tr("Geschlossener Volumenkörper", "Closed solid") + "</b></span>" if ok else
                       "<span style='color:#ff6369'><b>✘ " + tr("Nicht geschlossen", "Not closed") + "</b></span> – " +
                       tr("offene Kanten rot", "open edges in red"))
            if fid is not None and fid["rows"]:
                rows = fid["rows"]
                n_ok = sum(r["status"] == "ok" for r in rows)
                n_miss = sum(r["status"] in ("missing", "partial") for r in rows)
                n_sub = len(rows) - n_ok - n_miss
                col = "#3dd68c" if n_ok == len(rows) else ("#ff6369" if n_miss else "#f0a030")
                txt += (("<br>" if txt else "") + f"<span style='color:{col}'>"
                        + tr(f"{n_ok}/{len(rows)} Features exakt im Körper", f"{n_ok}/{len(rows)} features exact in the solid")
                        + (tr(f", {n_sub} ersetzt", f", {n_sub} substituted") if n_sub else "")
                        + (tr(f", {n_miss} fehlen (teilweise)", f", {n_miss} missing (partly)") if n_miss else "")
                        + "</span>")
            if spots:
                txt += "<br><span style='color:#ffd60a'>" + tr(
                    f"{len(spots)} Scanbereiche ohne Entsprechung (gelb markiert)",
                    f"{len(spots)} scan areas without counterpart (marked yellow)") + "</span>"
            self.lbl_closed.setText(txt)
        self._msg(tr("STEP-Vorschau: zeigt genau den Export-Inhalt (P zum Beenden)",
                     "STEP preview: shows exactly what is exported (P to leave)"), 12000)

    def _exit_step_preview(self):
        if not self.step_preview_on and not self.viewer.overlays_with_prefix("step_"):
            return
        self.step_preview_on = False
        self.viewer.hide_scalars(render=False)
        self.viewer.clear_overlays("step_")
        self.viewer.set_mesh_visible(self.act_mesh_vis.isChecked())
        self._update_feature_visibility()
        self._update_origin_axes()

    # ------------------------------------------------------------------ Export
    def export_step(self):
        if not self.features or self.model is None:
            self._msg(tr("Noch keine Features.", "No features yet."))
            return
        base = os.path.splitext(self.doc.mesh_path or "modell")[0] + ".step"
        path, _ = QFileDialog.getSaveFileName(self, tr("STEP exportieren", "Export STEP"), base, "STEP (*.step *.stp)")
        if not path:
            return
        mode = self._ex_mode()
        kw = self._export_kwargs()
        rest = kw["rest_mesh"]
        if rest is not None and len(rest[1]) > 150000:
            if QMessageBox.question(self, tr("Restnetz", "Remaining mesh"),
                                    tr(f"Restnetz hat {len(rest[1]):,} Dreiecke – sehr große Datei. Trotzdem anhängen?",
                                       f"Remaining mesh has {len(rest[1]):,} triangles – very large file. Append anyway?")) != QMessageBox.Yes:
                kw["rest_mesh"] = None
        feats = list(self.features)
        model = self.model
        tol = self.fit_tol.value()
        schema = self.ex_schema.currentData()
        T_post = np.linalg.inv(self.T_total) if self.ex_orig.isChecked() else None

        def work():
            items, log, res = self._build_items(mode, model, feats, kw, tol, self.runner.report)
            if not items:
                raise RuntimeError(tr("Nichts zu exportieren.", "Nothing to export."))
            self._fidelity(items, model, feats, tol, None, log)
            if T_post is not None and not np.allclose(T_post, np.eye(4)):
                items = [(n, ob.transform_shape(s, T_post), c) for n, s, c in items]
                log.append(tr("in ursprüngliche Scan-Lage zurücktransformiert", "transformed back to the original scan pose"))
            named = ob.write_step(items, path, schema)
            log.append(tr(f"STEP geschrieben: {path}", f"STEP written: {path}") + (tr(" (mit Namen/Farben)", " (with names/colours)") if named else ""))
            return log

        def done(log):
            self.export_log.setPlainText("\n".join(log))
            self._msg(tr(f"STEP gespeichert: {path}", f"STEP saved: {path}"), 15000)
        self._run(tr("Exportiere STEP…", "Exporting STEP…"), work, done)

    def export_mesh(self):
        if not self._need_mesh():
            return
        path, _ = QFileDialog.getSaveFileName(self, tr("Netz speichern", "Save mesh"), "netz.stl",
                                              "STL (*.stl);;PLY (*.ply);;OBJ (*.obj)")
        if path:
            pp.save_mesh(self.model, path)
            self._msg(tr(f"Netz gespeichert: {path}", f"Mesh saved: {path}"))

    def export_faceted(self):
        if not self._need_mesh():
            return
        if self.model.n_faces > 200000:
            if QMessageBox.question(self, tr("Facettiertes STEP", "Faceted STEP"),
                                    tr("Sehr viele Dreiecke → große Datei, lange Laufzeit. Trotzdem?",
                                       "Very many triangles → large file, long runtime. Continue?")) != QMessageBox.Yes:
                return
        path, _ = QFileDialog.getSaveFileName(self, tr("Facettiertes STEP", "Faceted STEP"), "netz_facettiert.step",
                                              "STEP (*.step *.stp)")
        if not path:
            return
        m = self.model

        def work():
            shp = ob.mesh_to_faceted(m.V, m.F, make_solid=m.is_closed)
            ob.write_step([("Netz", shp, None)], path)
            return path
        self._run(tr("Erzeuge facettiertes STEP…", "Creating faceted STEP…"), work,
                  lambda p: self._msg(tr(f"Gespeichert: {p}", f"Saved: {p}"), 15000))

    def export_report(self):
        if not self.features:
            return
        path, _ = QFileDialog.getSaveFileName(self, tr("Feature-Bericht", "Feature report"), "features.json", "JSON (*.json)")
        if path:
            exp.export_report(path, self.features)
            self._msg(tr(f"Bericht gespeichert: {path}", f"Report saved: {path}"))
