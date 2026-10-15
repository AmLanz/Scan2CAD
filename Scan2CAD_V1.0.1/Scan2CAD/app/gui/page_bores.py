"""Seite 5 · Bohrungen: alle Bohrungen durchgehen und korrigieren (Achse, Ø, Tiefe, Boden, Senkung,
Gewinde), Normwerte, Bohrung per Klick (K) oder aus Auswahl."""
from __future__ import annotations

import numpy as np
from qtpy.QtWidgets import (QCheckBox, QFormLayout, QGridLayout, QLabel, QListWidget, QVBoxLayout, QWidget)

from core.bores import (COARSE, hole_face_ids, hole_faces_near_click, hole_from_faces, hole_summary, snap_hole,
                        suggest_thread, tap_drill, thread_pitches)
from core.features import Feature
from core.geom_utils import unit
from core.i18n import tr

from .common import Section, button, combo, dspin, hint, row, set_combo


def bottoms():
    return [("flat", tr("flach", "flat")), ("drill", tr("Bohrerspitze", "drill point")),
            ("ball", tr("Kugel", "ball")), ("through", tr("durchgehend", "through"))]


def heads():
    return [("none", tr("keine", "none")), ("csk", tr("Kegelsenkung", "countersink")),
            ("cbore", tr("Flachsenkung", "counterbore"))]


def thread_displays():
    return [("cosmetic", tr("nur Angabe (Kernloch)", "annotation only (tap drill)")),
            ("nominal", tr("Nenn-Ø modelliert", "nominal Ø modelled")),
            ("helix", tr("Gewinde modelliert (experimentell)", "thread modelled (experimental)"))]


class BoresPage:
    def _page_bores(self):
        w = QWidget()
        v = QVBoxLayout(w)
        v.addLayout(row(button(tr("Alle erkennen", "Detect all"), self.do_detect_holes),
                        button(tr("Aus Auswahl", "From selection"), self.bore_from_selection,
                               tr("Sauberen Teil der Bohrungswand anmalen, dann hier klicken",
                                  "Paint the clean part of the hole wall, then click here")),
                        button(tr("Anklicken (K)", "Click (K)"), lambda: self._set_tool("bore"))))
        self.bore_list = QListWidget()
        self.bore_list.setMinimumHeight(140)
        self.bore_list.currentRowChanged.connect(self._bore_row_changed)
        self.bore_list.itemDoubleClicked.connect(lambda _: self.bore_center())
        v.addWidget(self.bore_list)
        v.addLayout(row(button("◀", lambda: self.bore_step(-1)), button("▶", lambda: self.bore_step(1)),
                        button(tr("Zentrieren", "Centre view"), self.bore_center),
                        button(tr("Löschen", "Delete"), self.bore_delete)))
        self.bore_status = QLabel("")
        self.bore_status.setWordWrap(True)
        v.addWidget(self.bore_status)

        s = Section(tr("Achse && Durchmesser", "Axis && diameter"))
        grid = QGridLayout()
        self.b_pt = [dspin(0, -1e5, 1e5, 0.01, 4, "mm") for _ in range(3)]
        self.b_dir = [dspin(0, -1, 1, 0.001, 5) for _ in range(3)]
        grid.addWidget(QLabel(tr("Eintritt", "Entry")), 0, 1)
        grid.addWidget(QLabel(tr("Richtung", "Direction")), 0, 2)
        for i, nm in enumerate("XYZ"):
            grid.addWidget(QLabel(nm), 1 + i, 0)
            grid.addWidget(self.b_pt[i], 1 + i, 1)
            grid.addWidget(self.b_dir[i], 1 + i, 2)
        s.add(grid)
        s.add(row(QLabel(tr("Richtung:", "Direction:")), button("X", lambda: self._bore_dir(0)),
                  button("Y", lambda: self._bore_dir(1)), button("Z", lambda: self._bore_dir(2)),
                  button(tr("Umkehren", "Flip"), lambda: self._bore_dir(-1))))
        f = QFormLayout()
        self.b_d = dspin(5, 0.1, 500, 0.01, 4, "mm")
        f.addRow("Ø", self.b_d)
        s.add(f)
        v.addWidget(s)

        s = Section(tr("Tiefe && Boden", "Depth && bottom"))
        f = QFormLayout()
        self.b_depth = dspin(10, 0, 5000, 0.1, 3, "mm")
        self.b_bottom = combo(bottoms())
        self.b_point = dspin(118, 30, 180, 1, 1, "°")
        self.lbl_seen = QLabel("")
        f.addRow(tr("Tiefe (zylindrisch)", "Depth (cylindrical)"), self.b_depth)
        f.addRow("", self.lbl_seen)
        f.addRow(tr("Boden", "Bottom"), self.b_bottom)
        f.addRow(tr("Spitzenwinkel", "Point angle"), self.b_point)
        s.add(f)
        v.addWidget(s)

        s = Section(tr("Senkung", "Countersink / counterbore"), expanded=False)
        f = QFormLayout()
        self.b_head = combo(heads())
        self.b_hd = dspin(10, 0.1, 500, 0.1, 3, "mm")
        self.b_ha = dspin(90, 30, 150, 1, 1, "°")
        self.b_hdepth = dspin(5, 0, 500, 0.1, 3, "mm")
        f.addRow(tr("Art", "Type"), self.b_head)
        f.addRow(tr("Senk-Ø", "Head Ø"), self.b_hd)
        f.addRow(tr("Senkwinkel", "Countersink angle"), self.b_ha)
        f.addRow(tr("Senktiefe (Flachsenkung)", "Counterbore depth"), self.b_hdepth)
        s.add(f)
        v.addWidget(s)

        s = Section(tr("Gewinde", "Thread"), expanded=False)
        f = QFormLayout()
        self.b_thread = combo([("", tr("kein Gewinde", "no thread"))] + [(k, k) for k in COARSE])
        self.b_thread.currentIndexChanged.connect(self._bore_thread_changed)
        self.b_pitch = combo([])
        self.b_tlen = dspin(8, 0, 5000, 0.5, 2, "mm")
        self.b_tdisp = combo(thread_displays())
        self.b_tcore = QCheckBox(tr("Ø auf Kernloch setzen", "set Ø to tap drill"))
        self.b_tcore.setChecked(True)
        f.addRow(tr("Größe", "Size"), self.b_thread)
        f.addRow(tr("Steigung", "Pitch"), self.b_pitch)
        f.addRow(tr("Gewindelänge", "Thread length"), self.b_tlen)
        f.addRow(tr("Darstellung", "Representation"), self.b_tdisp)
        f.addRow(self.b_tcore)
        s.add(f)
        s.add(button(tr("Gewinde vorschlagen (aus Ø)", "Suggest thread (from Ø)"), self.bore_suggest_thread))
        s.add(hint(tr("STEP kennt keine 'kosmetischen' Gewinde: „nur Angabe“ exportiert das Kernloch und schreibt "
                      "das Gewinde in den Namen/Bericht; „Nenn-Ø“ modelliert die Bohrung im Gewindebereich mit "
                      "Nenndurchmesser; die modellierte Schraubenlinie ist experimentell (große Dateien).",
                      "STEP has no cosmetic threads: “annotation only” exports the tap drill and writes the thread "
                      "into the name/report; “nominal Ø” models the hole with the nominal diameter over the thread "
                      "length; the modelled helix is experimental (large files).")))
        v.addWidget(s)

        self.b_depth_step = combo([(0.0, tr("Tiefe nicht runden", "do not round depth")), (0.1, "0,1 mm"),
                                   (0.5, "0,5 mm"), (1.0, "1 mm")], 2)
        v.addLayout(row(button(tr("Übernehmen", "Apply"), self.bore_apply, bold=True),
                        button(tr("Werte glätten", "Smooth values"), self.bore_snap,
                               tr("Ø auf Norm (Kernloch, Durchgangsloch, Bohrer), Winkel auf 90/118/120…, Achse auf X/Y/Z",
                                  "Ø to standard (tap drill, clearance, drill), angles to 90/118/120…, axis to X/Y/Z")),
                        self.b_depth_step))
        v.addLayout(row(button(tr("Alle glätten", "Smooth all"), self.bore_snap_all)))
        v.addLayout(row(button(tr("Umgebung als Drehprofil", "Surroundings as turned profile"), self.bore_port,
                               tr("Achse senkrecht zur Eintrittsfläche, Ringnuten/Einstiche/Fasen um diese Bohrung aus "
                                  "dem gemeinsamen Drehprofil; Schein-Bohrungen (Nutwände) werden entfernt",
                                  "Axis normal to the entry face, ring grooves/undercuts/chamfers around this hole from "
                                  "the common turned profile; spurious holes (groove walls) are removed")),
                        button(tr("… für alle", "… for all"), self.bore_port_all)))
        v.addWidget(hint(tr("Achse und Ø stammen aus dem sauber erfassten Teil der Wand; alles im Bohrungsvolumen "
                            "(Murks, Füllmembranen) wird beim Export durch die exakte Bohrung ersetzt. Ist der Boden "
                            "nicht erfasst, gilt die sichtbare Tiefe – hier anpassen.",
                            "Axis and Ø come from the cleanly scanned part of the wall; everything inside the hole "
                            "volume (junk, fill membranes) is replaced by the exact hole on export. If the bottom was "
                            "not scanned, the visible depth applies – adjust it here.")))
        v.addStretch(1)
        self._bore_id = None
        return w

    # ------------------------------------------------------------------ Liste
    def _holes_feats(self):
        return [f for f in self.features if f.kind == "hole"]

    def refresh_bores(self):
        if not hasattr(self, "bore_list"):
            return
        hs = self._holes_feats()
        self.bore_list.blockSignals(True)
        self.bore_list.clear()
        sel_row = -1
        for i, f in enumerate(hs):
            warn = "  ⚠" if (not f.params.get("bottom_seen", True) and f.params.get("bottom") != "through") else ""
            self.bore_list.addItem(f"{f.name}: {hole_summary(f.params).split(', ⚠')[0]}{warn}")
            if f.id == self._bore_id:
                sel_row = i
        if sel_row < 0 and hs:
            sel_row = 0
        self.bore_list.setCurrentRow(sel_row)
        self.bore_list.blockSignals(False)
        self._bore_id = hs[sel_row].id if sel_row >= 0 else None
        self._bore_load()

    def _bore_row_changed(self, r):
        hs = self._holes_feats()
        if 0 <= r < len(hs):
            self._bore_id = hs[r].id
            self._bore_load()
            self._bore_highlight()

    def bore_select_id(self, fid, from_tree=False):
        hs = self._holes_feats()
        for i, f in enumerate(hs):
            if f.id == fid:
                self.bore_list.setCurrentRow(i)
                return

    def bore_step(self, d):
        n = self.bore_list.count()
        if n:
            self.bore_list.setCurrentRow((self.bore_list.currentRow() + d) % n)
            self.bore_center()

    def _bore(self):
        return self.doc.feature(self._bore_id) if self._bore_id is not None else None

    def _bore_load(self):
        f = self._bore()
        widgets = [*self.b_pt, *self.b_dir, self.b_d, self.b_depth, self.b_bottom, self.b_point, self.b_head,
                   self.b_hd, self.b_ha, self.b_hdepth, self.b_thread, self.b_pitch, self.b_tlen, self.b_tdisp]
        for wdg in widgets:
            wdg.blockSignals(True)
        if f is None:
            self.bore_status.setText(tr("Keine Bohrung gewählt.", "No hole selected."))
        else:
            p = f.params
            for i in range(3):
                self.b_pt[i].setValue(float(p["point"][i]))
                self.b_dir[i].setValue(float(unit(p["axis"])[i]))
            self.b_d.setValue(float(p["d"]))
            self.b_depth.setValue(float(p["depth"]))
            set_combo(self.b_bottom, p.get("bottom", "flat"))
            self.b_point.setValue(float(p.get("point_angle", 118)))
            set_combo(self.b_head, p.get("head", "none"))
            self.b_hd.setValue(float(p.get("head_d", 2 * p["d"])))
            self.b_ha.setValue(float(p.get("head_angle", 90)))
            self.b_hdepth.setValue(float(p.get("head_depth", 0)))
            th = p.get("thread") or {}
            set_combo(self.b_thread, th.get("size", ""))
            self._fill_pitches(th.get("size", ""), th.get("pitch"))
            self.b_tlen.setValue(float(th.get("length", p["depth"])) if th else float(p["depth"]))
            set_combo(self.b_tdisp, th.get("display", "cosmetic"))
            seen = p.get("seen_depth")
            self.lbl_seen.setText(tr(f"sichtbar im Scan: {seen:.2f} mm", f"visible in scan: {seen:.2f} mm")
                                  if seen is not None else "")
            st = hole_summary(p)
            if not p.get("bottom_seen", True) and p.get("bottom") != "through":
                st = "<span style='color:#ffd24a'>" + st + "</span>"
            self.bore_status.setText(f"<b>{f.name}</b>: {st}")
        for wdg in widgets:
            wdg.blockSignals(False)

    def _fill_pitches(self, size, cur=None):
        self.b_pitch.blockSignals(True)
        self.b_pitch.clear()
        for i, pch in enumerate(thread_pitches(size) if size else []):
            self.b_pitch.addItem(f"{pch:g}" + (tr(" (Regel)", " (coarse)") if i == 0 else tr(" (fein)", " (fine)")), pch)
        if cur is not None:
            set_combo(self.b_pitch, cur)
        self.b_pitch.blockSignals(False)

    def _bore_thread_changed(self, _i):
        size = self.b_thread.currentData()
        self._fill_pitches(size)
        if size and self.b_tcore.isChecked():
            self.b_d.setValue(tap_drill(size, self.b_pitch.currentData()))

    def _bore_dir(self, k):
        d = np.array([s.value() for s in self.b_dir])
        if k < 0:
            d = -d
        else:
            e = np.eye(3)[k]
            d = e if d @ e >= 0 else -e
        for i in range(3):
            self.b_dir[i].setValue(float(d[i]))

    def _bore_highlight(self):
        f = self._bore()
        if f is None:
            self.viewer.remove_overlay("__bore_axis")
            return
        p = f.params
        a = unit(p["axis"])
        L = float(p["depth"]) + 0.6 * float(p["d"]) + 3
        self.viewer.set_lines("__bore_axis", [np.vstack([p["point"] - 3 * a, p["point"] + L * a])], color="#ffd60a", width=4)

    def bore_center(self):
        f = self._bore()
        if f is None:
            return
        p = f.params
        a = unit(p["axis"])                     # Bohrrichtung (ins Material)
        depth = min(float(p["depth"]), 2.5 * float(p["d"]))
        # von der Eintrittsseite schräg (35°) in die Bohrung schauen, seitlich zur bisherigen Kamera
        cam = self.viewer.plotter.camera
        side = np.asarray(cam.position, float) - np.asarray(cam.focal_point, float)
        side = side - (side @ a) * a
        if np.linalg.norm(side) < 1e-6:
            side = np.cross(a, [1.0, 0.0, 0.0] if abs(a[0]) < 0.9 else [0.0, 1.0, 0.0])
        side = unit(side)
        ang = np.radians(35.0)
        view = np.cos(ang) * a - np.sin(ang) * side
        self.viewer.focus(np.asarray(p["point"]) + 0.3 * depth * a, distance=max(25.0, 9.0 * float(p["d"])),
                          view_dir=view)
        self._bore_highlight()

    # ------------------------------------------------------------------ Bearbeiten
    def _bore_params_from_ui(self, f):
        p = dict(f.params)
        p["point"] = np.array([s.value() for s in self.b_pt])
        d = np.array([s.value() for s in self.b_dir])
        p["axis"] = unit(d if np.linalg.norm(d) > 1e-9 else f.params["axis"])
        p["d"] = self.b_d.value()
        p["depth"] = self.b_depth.value()
        p["bottom"] = self.b_bottom.currentData()
        p["point_angle"] = self.b_point.value()
        p["head"] = self.b_head.currentData()
        p["head_d"] = self.b_hd.value()
        p["head_angle"] = self.b_ha.value()
        p["head_depth"] = self.b_hdepth.value()
        size = self.b_thread.currentData()
        p["thread"] = ({"size": size, "pitch": self.b_pitch.currentData(), "length": self.b_tlen.value(),
                        "display": self.b_tdisp.currentData()} if size else None)
        if p["bottom"] != f.params.get("bottom") or abs(p["depth"] - f.params["depth"]) > 1e-9:
            p["bottom_seen"] = True          # Anwender hat entschieden
        return p

    def _bore_set(self, f, p, label):
        self.checkpoint(label)
        f.params = p
        if size := (p.get("thread") or {}).get("size"):
            base = f.name.split(" [")[0]
            f.name = f"{base} [{size}]"
        f.face_ids = hole_face_ids(self.model, p, tol=self.fit_tol.value())
        f.invalidate()
        self.refresh_all()
        self._bore_highlight()

    def bore_apply(self):
        f = self._bore()
        if f is None:
            return
        self._bore_set(f, self._bore_params_from_ui(f), tr(f"Bohrung „{f.name}“ ändern", f"Edit hole “{f.name}”"))

    def _axis_dirs(self):
        dirs = [np.eye(3)[i] for i in range(3)]
        for f in self.features:
            if f.kind == "plane":
                n = unit(f.params["normal"])
                if all(abs(float(n @ d)) < 0.9999 for d in dirs):
                    dirs.append(n)
        return np.array(dirs)

    def bore_snap(self):
        f = self._bore()
        if f is None:
            return
        p = self._bore_params_from_ui(f)
        q, notes = snap_hole(p, depth_step=float(self.b_depth_step.currentData() or 0), axis_dirs=self._axis_dirs(),
                             meas=self.fit_tol.value())
        self._bore_set(f, q, tr("Bohrung glätten", "Smooth hole"))
        self._msg("; ".join(notes) or tr("Schon glatt.", "Already smooth."), 12000)

    def bore_snap_all(self):
        hs = self._holes_feats()
        if not hs:
            return
        self.checkpoint(tr("Alle Bohrungen glätten", "Smooth all holes"))
        step = float(self.b_depth_step.currentData() or 0)
        for f in hs:
            q, _ = snap_hole(f.params, depth_step=step, axis_dirs=self._axis_dirs(), meas=self.fit_tol.value())
            f.params = q
            f.face_ids = hole_face_ids(self.model, q, tol=self.fit_tol.value())
            f.invalidate()
        self.refresh_all()

    def bore_suggest_thread(self):
        f = self._bore()
        if f is None:
            return
        s = suggest_thread(self.b_d.value())
        if s is None:
            self._msg(tr("Kein Kernloch-Ø passt.", "No tap drill Ø matches."))
            return
        size, pitch = s
        set_combo(self.b_thread, size)
        self._fill_pitches(size, pitch)
        self.b_tlen.setValue(max(self.b_depth.value() - 2 * pitch, 0.5 * self.b_depth.value()))
        self._msg(tr(f"Vorschlag: {size}×{pitch:g} – mit „Übernehmen“ bestätigen",
                     f"Suggestion: {size}×{pitch:g} – confirm with “Apply”"))

    def bore_delete(self):
        f = self._bore()
        if f is None:
            return
        self.checkpoint(tr("Bohrung löschen", "Delete hole"))
        self.doc.features = [x for x in self.features if x.id != f.id]
        self._bore_id = None
        self.refresh_all()

    def _run_ports(self, near, label):
        from core.ports import analyze_ports, apply_ports
        tol = self.fit_tol.value()

        def fn(d, rep):
            res = analyze_ports(d.model, d.features, tol, labels=d.assigned(), near=near)
            apply_ports(d, res)
            return res["report"] or [tr("Kein Drehprofil gefunden (Achse/Eintrittsfläche prüfen).",
                                        "No turned profile found (check axis/entry face).")]
        self.run_doc(label, fn, after=self.refresh_bores)

    def bore_port(self):
        f = self._bore()
        if f is None:
            return
        p = f.params
        self._run_ports((np.asarray(p["point"], float), 0.75 * float(p["d"]) + 1.0),
                        tr(f"Drehprofil um {f.name}", f"Turned profile around {f.name}"))

    def bore_port_all(self):
        self._run_ports(None, tr("Drehprofile an allen Bohrungen", "Turned profiles at all holes"))

    # ------------------------------------------------------------------ Neu
    def _add_hole_result(self, ids_fn):
        tol = self.fit_tol.value()
        holder = {}

        def fn(d, rep):
            ids = ids_fn(d)
            res = hole_from_faces(d.model, ids, d.features, tol)
            f = Feature("hole", res["params"], "sub", res["face_ids"])
            f.name = tr("Bohrung", "Hole") + f" {f.id}"
            d.features = [x for x in d.features if x.id not in set(res["consumed"])]
            d.features.append(f)
            holder["id"] = f.id
            return [f"{f.name}: {hole_summary(f.params)}"] + res["notes"]

        def after():
            self._bore_id = holder.get("id")
            self.goto_page("bores")
            self.refresh_bores()
            self.bore_center()
        self.run_doc(tr("Bohrung anlegen", "Create hole"), fn, after=after)

    def bore_from_selection(self):
        ids = np.nonzero(self.sel)[0]
        if len(ids) < 12:
            self._msg(tr("Erst den sauberen Teil der Bohrungswand anmalen.", "Paint the clean part of the hole wall first."))
            return
        self.sel[:] = False
        self.recolor(render=False)
        self._add_hole_result(lambda d: ids)

    def bore_from_click(self, cell):
        tol = self.fit_tol.value()
        self._add_hole_result(lambda d: hole_faces_near_click(d.model, cell, tol))
