"""Seite 6 · Symmetrie & Bedingungen."""
from __future__ import annotations

import math

import numpy as np
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (QCheckBox, QFormLayout, QLabel, QListWidget, QListWidgetItem, QPlainTextEdit,
                            QVBoxLayout, QWidget)

from core import constraints as cs
from core import symmetry as sy
from core.geom_utils import unit
from core.i18n import tr

from .common import Section, button, combo, dspin, hint, ispin, row


def ctypes():
    return [(k, cs.TYPES[k]) for k in cs.TYPES.keys()]


class SymConPage:
    def _page_symcon(self):
        w = QWidget()
        v = QVBoxLayout(w)
        s = Section(tr("Symmetrien", "Symmetries"))
        self.sym_list = QListWidget()
        self.sym_list.setMinimumHeight(120)
        self.sym_list.itemChanged.connect(lambda _: self._draw_symmetries())
        s.add(self.sym_list)
        self.chk_sym_show = QCheckBox(tr("im Modell anzeigen", "show in model"))
        self.chk_sym_show.setChecked(True)
        self.chk_sym_show.toggled.connect(lambda _: self._draw_symmetries())
        self.chk_sym_sel = QCheckBox(tr("nur Auswahl", "selection only"))
        self.sym_min = dspin(85, 50, 100, 1, 0, "%")
        s.add(row(button(tr("Suchen", "Find"), self.sym_detect, bold=True), self.chk_sym_sel,
                  QLabel(tr("Schwelle", "threshold")), self.sym_min))
        s.add(row(button(tr("Löschen", "Delete"), self.sym_delete),
                  button(tr("Erzwingen", "Enforce"), self.sym_enforce,
                         tr("Bedingungen 'invariant unter Symmetrie' für alle passenden Features anlegen",
                            "Create 'invariant under symmetry' constraints for all matching features")),
                  self.chk_sym_show))
        f = QFormLayout()
        self.sym_src = combo([])
        self.sym_n = ispin(0, 0, 36)
        self.sym_n.setSpecialValueText("∞")
        f.addRow(tr("Neu aus", "New from"), self.sym_src)
        f.addRow(tr("Zähligkeit (Achse)", "Order (axis)"), self.sym_n)
        s.add(f)
        s.add(row(button(tr("Spiegelebene anlegen", "Add mirror plane"), lambda: self.sym_add("mirror")),
                  button(tr("Drehachse anlegen", "Add rotation axis"), lambda: self.sym_add("rot"))))
        s.add(hint(tr("Global = ganzes Bauteil (Anteil passender Punkte), sonst Muster einzelner Features "
                      "(Lochkreis, Spiegelpaar, koaxial). Symmetrie-Elemente ohne Messdaten (z. B. eine Drehachse) "
                      "lassen sich über Bedingungen festlegen – etwa: Achse liegt in den Mittelebenen zweier "
                      "Flächenpaare.", "Global = whole part (share of matching points), otherwise patterns of "
                      "individual features (bolt circle, mirror pair, coaxial). Symmetry elements without scan data "
                      "(e.g. a rotation axis) can be fixed by constraints – e.g. the axis lies in the mid-planes of "
                      "two pairs of faces.")))
        v.addWidget(s)

        s = Section(tr("Bedingungen", "Constraints"))
        f = QFormLayout()
        self.con_type = combo(ctypes())
        self.con_type.currentIndexChanged.connect(self._con_type_changed)
        self.con_a, self.con_b, self.con_c = combo([]), combo([]), combo([])
        self.con_val = dspin(0, -1e5, 1e5, 0.01, 4)
        f.addRow(tr("Art", "Type"), self.con_type)
        f.addRow("A", self.con_a)
        f.addRow("B", self.con_b)
        self.lbl_con_c = QLabel(tr("Ebene", "Plane"))
        f.addRow(self.lbl_con_c, self.con_c)
        f.addRow(tr("Wert", "Value"), self.con_val)
        s.add(f)
        s.add(row(button(tr("Hinzufügen", "Add"), self.con_add), button(tr("Wert messen", "Measure value"), self.con_measure)))
        self.con_list = QListWidget()
        self.con_list.setMinimumHeight(110)
        s.add(self.con_list)
        s.add(row(button(tr("Entfernen", "Remove"), self.con_delete),
                  button(tr("Lösen (Neu-Fit über Scan)", "Solve (refit against scan)"), self.con_solve, bold=True)))
        s.add(QLabel("<b>" + tr("Vorschläge", "Suggestions") + "</b>"))
        self.sug_list = QListWidget()
        self.sug_list.setMinimumHeight(100)
        s.add(self.sug_list)
        s.add(row(button(tr("Vorschlagen", "Suggest"), self.con_suggest),
                  button(tr("Angehakte übernehmen", "Add checked"), self.con_take_suggestions)))
        self.con_report = QPlainTextEdit()
        self.con_report.setReadOnly(True)
        self.con_report.setMaximumHeight(140)
        s.add(self.con_report)
        s.add(hint(tr("Bedingungen werden nicht an den Parametern 'verbogen', sondern alle beteiligten Flächen "
                      "werden gemeinsam neu gegen den Scan gefittet (Bedingungen stark gewichtet). Flächen ohne "
                      "Messdaten (festgelegter Boden, Füllebenen, Symmetrieachsen) folgen nur den Bedingungen – so "
                      "lässt sich z. B. der Abstand Deckfläche–Boden eines offenen Scans vorgeben.",
                      "Constraints do not 'bend' parameters; all involved surfaces are refitted together against "
                      "the scan (constraints heavily weighted). Surfaces without scan data (fixed floor, fill planes, "
                      "symmetry axes) only follow the constraints – e.g. to set the top-to-floor distance of an open "
                      "scan.")))
        v.addWidget(s)
        v.addStretch(1)
        self._suggestions = []
        self._con_type_changed()
        return w

    # ------------------------------------------------------------------ Referenzen
    def _ref_items(self):
        items = []
        planes = [f for f in self.features if f.kind == "plane"]
        for f in self.features:
            if f.kind in cs.SUPPORTED_KINDS:
                items.append((f"f:{f.id}", f.name))
        for i, a in enumerate(planes):
            na = unit(a.params["normal"])
            for b in planes[i + 1:]:
                nb = unit(b.params["normal"])
                if abs(float(na @ nb)) > math.cos(math.radians(3)):
                    off = abs(float((np.asarray(b.params["origin"]) - a.params["origin"]) @ na))
                    if off > 1e-3:
                        items.append((f"mid:{a.id}:{b.id}", tr(f"Mitte {a.name} | {b.name}", f"mid {a.name} | {b.name}")))
        for s in self.doc.symmetries:
            items.append((f"s:{s['id']}", f"{s['name']} ({sy.sym_label(s)})"))
        for k in ("X", "Y", "Z"):
            items.append((f"w:{k}", tr(f"Weltachse {k}", f"world axis {k}")))
        for k in ("XY", "YZ", "XZ"):
            items.append((f"w:{k}", tr(f"Weltebene {k}", f"world plane {k}")))
        return items

    def _ref_names(self):
        return dict(self._ref_items())

    def refresh_symcon(self):
        if not hasattr(self, "sym_list"):
            return
        items = self._ref_items()
        for cb in (self.con_a, self.con_b, self.con_c):
            cur = cb.currentData()
            cb.blockSignals(True)
            cb.clear()
            for k, lab in items:
                cb.addItem(lab, k)
            i = cb.findData(cur)
            if i >= 0:
                cb.setCurrentIndex(i)
            cb.blockSignals(False)
        cur = self.sym_src.currentData()
        self.sym_src.clear()
        for k, lab in items:
            if k.startswith(("f:", "mid:", "w:")):
                self.sym_src.addItem(lab, k)
        i = self.sym_src.findData(cur)
        if i >= 0:
            self.sym_src.setCurrentIndex(i)
        # Symmetrie-Liste
        self.sym_list.blockSignals(True)
        shown = {self.sym_list.item(i).data(Qt.UserRole): self.sym_list.item(i).checkState() == Qt.Checked
                 for i in range(self.sym_list.count())}
        self.sym_list.clear()
        for s in self.doc.symmetries:
            txt = f"{s['name']}: {sy.sym_label(s)} – {100 * s.get('score', 1):.0f}%"
            if s.get("info"):
                txt += f" – {s['info']}"
            if s.get("scope") == "global":
                txt += tr(" (ganzes Bauteil)", " (whole part)")
            it = QListWidgetItem(txt)
            it.setFlags(it.flags() | Qt.ItemIsUserCheckable)
            it.setCheckState(Qt.Checked if shown.get(s["id"], True) else Qt.Unchecked)
            it.setData(Qt.UserRole, s["id"])
            it.setToolTip(sy.sym_summary(s))
            self.sym_list.addItem(it)
        self.sym_list.blockSignals(False)
        # Bedingungen
        names = self._ref_names()
        self.con_list.clear()
        for c in self.doc.constraints:
            it = QListWidgetItem(cs.describe(c, names))
            it.setData(Qt.UserRole, c["id"])
            self.con_list.addItem(it)
        self._draw_symmetries()

    # ------------------------------------------------------------------ Anzeige
    def _draw_symmetries(self):
        self.viewer.clear_overlays("__sym_")
        on_page = self.toolbox.currentIndex() == 5 if hasattr(self, "toolbox") else True
        if self.model is None or not self.chk_sym_show.isChecked() or not on_page:
            return
        L = 0.7 * self.model.diag
        labels_p, labels_t = [], []
        for i in range(self.sym_list.count()):
            it = self.sym_list.item(i)
            if it.checkState() != Qt.Checked:
                continue
            s = next((x for x in self.doc.symmetries if x["id"] == it.data(Qt.UserRole)), None)
            if s is None:
                continue
            if s["type"] == "mirror":
                self.viewer.set_plane_glyph(f"__sym_{s['id']}", s["o"], s["nrm"], L, color="#4cc2ff")
                labels_p.append(np.asarray(s["o"]))
            else:
                d = unit(s["d"])
                p = np.asarray(s["p"], float)
                c = self.model.centroids.mean(0)
                p = p + ((c - p) @ d) * d
                col = "#ffb224" if s.get("n", 0) else "#3dd68c"
                self.viewer.set_lines(f"__sym_{s['id']}", [np.vstack([p - 0.6 * L * d, p + 0.6 * L * d])], color=col, width=3)
                labels_p.append(p + 0.6 * L * d)
            labels_t.append(s["name"])
        if labels_t:
            self.viewer.set_text("__sym_labels", np.array(labels_p), labels_t)

    # ------------------------------------------------------------------ Symmetrien
    def sym_detect(self):
        tol = self.fit_tol.value()
        min_score = self.sym_min.value() / 100.0
        ids = np.nonzero(self.sel)[0] if self.chk_sym_sel.isChecked() else None
        if ids is not None and len(ids) < 50:
            self._msg(tr("Für „nur Auswahl“ erst einen Bereich auswählen.", "Select an area first for “selection only”."))
            return

        def fn(d, rep):
            from core.mesh_model import MeshModel
            if ids is not None:
                V, F, _ = d.model.submesh(ids)
                sub = MeshModel(V, F, d.model.synthetic[ids])
                found = sy.detect_global(sub, tol, [], min_score=min_score)
                for k, s in enumerate(found):
                    s["id"] = sy.new_sym_id(d.symmetries + found[:k])
                    s["name"] = f"S{s['id']}"
                    s["scope"] = "selection"
                    s["info"] = tr("Auswahl", "selection")
                d.symmetries = d.symmetries + found
                return [f"{s['name']}: {sy.sym_label(s)} ({100 * s['score']:.0f}%)" for s in found] or \
                    [tr("Keine Symmetrien gefunden.", "No symmetries found.")]
            keep = [s for s in d.symmetries if not s.get("auto")]
            found = sy.detect_all(d.model, d.features, tol, existing=keep, min_score=min_score)
            for s in found:
                s["auto"] = True
            d.symmetries = keep + found
            return [f"{s['name']}: {sy.sym_label(s)} ({100 * s['score']:.0f}%)" for s in found] or \
                [tr("Keine Symmetrien gefunden.", "No symmetries found.")]
        self.run_doc(tr("Symmetrien suchen", "Find symmetries"), fn)

    def sym_delete(self):
        it = self.sym_list.currentItem()
        if it is None:
            return
        sid = it.data(Qt.UserRole)
        self.checkpoint(tr("Symmetrie löschen", "Delete symmetry"))
        self.doc.symmetries = [s for s in self.doc.symmetries if s["id"] != sid]
        self.doc.constraints = [c for c in self.doc.constraints if f"s:{sid}" not in c["refs"]]
        self.refresh_symcon()
        self._refresh_align_combos()

    def sym_enforce(self):
        it = self.sym_list.currentItem()
        if it is None:
            self._msg(tr("Erst eine Symmetrie wählen.", "Select a symmetry first."))
            return
        s = next((x for x in self.doc.symmetries if x["id"] == it.data(Qt.UserRole)), None)
        if s is None or self.model is None:
            return
        from core.analysis import feature_measurable
        tol = self.fit_tol.value()
        L = float(self.model.diag)
        gs = ({"type": "plane", "o": np.asarray(s["o"], float), "n": unit(s["nrm"])} if s["type"] == "mirror"
              else {"type": "axis", "p": np.asarray(s["p"], float), "d": unit(s["d"])})
        have = {(c["type"], frozenset(c["refs"])) for c in self.doc.constraints}
        new = []
        for f in self.features:
            if f.kind not in cs.SUPPORTED_KINDS:
                continue
            ref = f"f:{f.id}"
            if ("invariant", frozenset([ref, f"s:{s['id']}"])) in have:
                continue
            if s.get("features") and f.id not in s["features"]:
                continue
            try:
                m = feature_measurable(f)
            except Exception:
                continue
            g = {"type": m["type"] if m["type"] != "sphere" else "point", "o": m.get("o"), "n": m.get("n"),
                 "p": m.get("p"), "d": m.get("d"), "r": m.get("r")}
            r = cs.constraint_residual({"type": "invariant", "refs": ["a", "b"]}, {"a": g, "b": gs}, L)
            if s.get("features") or float(np.max(np.abs(r))) < max(4 * tol, 0.1):
                new.append(ref)
        if not new:
            self._msg(tr("Keine passenden Features.", "No matching features."))
            return
        self.checkpoint(tr("Symmetrie erzwingen", "Enforce symmetry"))
        for ref in new:
            self.doc.constraints.append(cs.new_constraint("invariant", [ref, f"s:{s['id']}"], None, self.doc.constraints))
        self.refresh_symcon()
        self._msg(tr(f"{len(new)} Bedingungen angelegt – mit „Lösen“ anwenden", f"{len(new)} constraints created – apply with “Solve”"))

    def _ref_geom(self, ref):
        """Ebene (o, n) oder Achse (p, d) einer Referenz für das Anlegen von Symmetrien."""
        if ref.startswith("w:"):
            g = cs._world(ref)
        elif ref.startswith("mid:"):
            _, a, b = ref.split(":")
            fa, fb = self.doc.feature(int(a)), self.doc.feature(int(b))
            na, nb = unit(fa.params["normal"]), unit(fb.params["normal"])
            if na @ nb < 0:
                nb = -nb
            g = {"type": "plane", "o": 0.5 * (np.asarray(fa.params["origin"]) + fb.params["origin"]), "n": unit(na + nb)}
        else:
            from core.analysis import feature_measurable
            m = feature_measurable(self.doc.feature(int(ref[2:])))
            g = {"type": m["type"], "o": m.get("o"), "n": m.get("n"), "p": m.get("p"), "d": m.get("d")}
        return g

    def sym_add(self, kind):
        ref = self.sym_src.currentData()
        if ref is None:
            return
        g = self._ref_geom(ref)
        s = {"id": sy.new_sym_id(self.doc.symmetries), "score": 1.0, "scope": "manual", "features": []}
        s["name"] = f"S{s['id']}"
        if kind == "mirror":
            if g["type"] != "plane":
                self._msg(tr("Für eine Spiegelebene eine Ebene/Mittelebene wählen.", "Choose a plane/mid-plane for a mirror plane."))
                return
            s.update(type="mirror", o=np.asarray(g["o"], float), nrm=unit(g["n"]))
        else:
            if g["type"] == "axis":
                s.update(type="rot", n=self.sym_n.value(), p=np.asarray(g["p"], float), d=unit(g["d"]))
            elif g["type"] == "plane":
                s.update(type="rot", n=self.sym_n.value(), p=np.asarray(g["o"], float), d=unit(g["n"]))
            else:
                self._msg(tr("Für eine Drehachse eine Achse oder Ebenennormale wählen.", "Choose an axis or plane normal."))
                return
        self.checkpoint(tr("Symmetrie anlegen", "Add symmetry"))
        self.doc.symmetries.append(s)
        self.refresh_symcon()
        self._refresh_align_combos()

    # ------------------------------------------------------------------ Bedingungen
    def _con_type_changed(self, *_):
        t = self.con_type.currentData()
        ar = cs.ARITY.get(t, 2)
        self.con_b.setEnabled(ar != 1)
        self.con_c.setEnabled(t == "symmetric")
        self.lbl_con_c.setEnabled(t == "symmetric")
        self.con_val.setEnabled(t in cs.NEEDS_VALUE)
        self.con_val.setSuffix(" °" if t == "angle" else " mm")

    def con_add(self):
        t = self.con_type.currentData()
        refs = [self.con_a.currentData()]
        if cs.ARITY.get(t, 2) != 1:
            refs.append(self.con_b.currentData())
        if t == "symmetric":
            refs.append(self.con_c.currentData())
        if any(r is None for r in refs) or (len(refs) > 1 and refs[0] == refs[1]):
            self._msg(tr("Referenzen wählen (A und B verschieden).", "Choose references (A and B different)."))
            return
        val = self.con_val.value() if t in cs.NEEDS_VALUE else None
        self.checkpoint(tr("Bedingung hinzufügen", "Add constraint"))
        self.doc.constraints.append(cs.new_constraint(t, refs, val, self.doc.constraints))
        self.refresh_symcon()

    def con_measure(self):
        """Aktuellen Wert (Abstand/Winkel/Ø) als Vorgabe eintragen."""
        from core.analysis import measure
        t = self.con_type.currentData()
        try:
            ga = self._ref_geom(self.con_a.currentData())
            if t in ("diameter", "radius"):
                f = self.doc.feature(int(self.con_a.currentData()[2:]))
                r = f.params.get("radius", 0.5 * f.params.get("d", 0))
                self.con_val.setValue(2 * r if t == "diameter" else r)
                return
            gb = self._ref_geom(self.con_b.currentData())
            ma = {"type": ga["type"], "o": ga.get("o"), "n": ga.get("n"), "p": ga.get("p"), "d": ga.get("d")}
            mb = {"type": gb["type"], "o": gb.get("o"), "n": gb.get("n"), "p": gb.get("p"), "d": gb.get("d")}
            rows = measure(ma, mb)
            want = "°" if t == "angle" else "mm"
            for lab, val, u in rows:
                if u == want and not isinstance(val, np.ndarray) and not lab.startswith("Ø"):
                    self.con_val.setValue(float(val))
                    return
        except Exception as e:  # noqa: BLE001
            self._msg(str(e))

    def con_delete(self):
        it = self.con_list.currentItem()
        if it is None:
            return
        cid = it.data(Qt.UserRole)
        self.checkpoint(tr("Bedingung entfernen", "Remove constraint"))
        self.doc.constraints = [c for c in self.doc.constraints if c["id"] != cid]
        self.refresh_symcon()

    def con_suggest(self):
        if self.model is None:
            return
        have = {(c["type"], frozenset(c["refs"])) for c in self.doc.constraints}
        sug = [s for s in cs.suggest(self.features, self.doc.symmetries, self.fit_tol.value())
               if (s[0]["type"], frozenset(s[0]["refs"])) not in have]
        self._suggestions = sug
        names = self._ref_names()
        self.sug_list.clear()
        for c, why, dev in sug:
            it = QListWidgetItem(f"{cs.describe(c, names)}   – {why}")
            it.setFlags(it.flags() | Qt.ItemIsUserCheckable)
            it.setCheckState(Qt.Unchecked)
            self.sug_list.addItem(it)
        if not sug:
            self._msg(tr("Keine Vorschläge.", "No suggestions."))

    def con_take_suggestions(self):
        take = [self._suggestions[i][0] for i in range(self.sug_list.count())
                if self.sug_list.item(i).checkState() == Qt.Checked and i < len(self._suggestions)]
        if not take:
            return
        self.checkpoint(tr("Vorschläge übernehmen", "Add suggestions"))
        for c in take:
            self.doc.constraints.append(cs.new_constraint(c["type"], c["refs"], c.get("value"), self.doc.constraints))
        self.sug_list.clear()
        self._suggestions = []
        self.refresh_symcon()

    def con_solve(self):
        if not self.doc.constraints:
            self._msg(tr("Keine Bedingungen.", "No constraints."))
            return
        tol = self.fit_tol.value()
        holder = {}

        def fn(d, rep):
            res = cs.solve(d.model, d.features, d.symmetries, d.constraints, tol=tol)
            cs.apply_solution(d.model, d.features, d.symmetries, res, tol)
            # festgelegte Ebenen mit Füllung (Boden, erzwungene Ebenen): Netz-Deckel mitziehen
            from core import mesh_ops as mo
            for fl in list(d.model.fills):
                f = d.feature(fl.get("feature")) if fl.get("feature") is not None else None
                if f is None or not f.params.get("fixed") or f"f:{f.id}" not in res["params"]:
                    continue
                try:
                    plane = (np.asarray(f.params["origin"], float), unit(f.params["normal"]))
                    m3, fmap, new_ids, cap_ids = mo.refill(d.model, fl["id"], "plane", plane=plane)
                    d.replace_model(m3, face_map=fmap)
                    f.face_ids = np.asarray(cap_ids, np.int64)
                    m3.fills[-1]["feature"] = f.id
                except Exception:
                    pass
            # Bohrungen: Flächenzuordnung zur neuen Lage
            from core.bores import hole_face_ids
            for f in d.features:
                if f.kind == "hole" and f"f:{f.id}" in res["params"]:
                    f.face_ids = hole_face_ids(d.model, f.params, tol=tol)
            holder["rep"] = res["report"]
            return [tr("Bedingungen gelöst", "constraints solved") if res["ok"] else
                    tr("⚠ nicht alle Bedingungen erfüllbar", "⚠ not all constraints satisfiable")]
        self.run_doc(tr("Bedingungen lösen", "Solve constraints"), fn,
                     after=lambda: self.con_report.setPlainText("\n".join(holder.get("rep", []))))
