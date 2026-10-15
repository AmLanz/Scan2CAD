"""Seite 2 · Ausrichten: schnell, 3-2-1 (Features/Auswahl/Symmetrien), numerisch, optimieren."""
from __future__ import annotations

import numpy as np
from qtpy.QtWidgets import QCheckBox, QComboBox, QFormLayout, QGridLayout, QLabel, QVBoxLayout, QWidget

from core.analysis import floor_translation, optimize_alignment
from core.features import snap_feature
from core.fitting import KIND_LABELS, auto_fit
from core.geom_utils import orthonormal_frame, unit
from core.i18n import tr

from .common import Section, button, combo, dspin, hint, rot_matrix, row


class AlignPage:
    def _page_align(self):
        w = QWidget()
        v = QVBoxLayout(w)
        s = Section(tr("Schnell", "Quick"))
        grid = QGridLayout()
        btns = [(tr("Hauptachsen (PCA)", "Principal axes (PCA)"), self.align_pca),
                (tr("Schwerpunkt → 0", "Centroid → 0"), self.center_centroid),
                (tr("Auf Boden (min Z = 0)", "Onto floor (min Z = 0)"), self.drop_to_floor),
                (tr("Mitte XY → 0", "Centre XY → 0"), self.center_xy)]
        for i, (label, fn) in enumerate(btns):
            grid.addWidget(button(label, fn), i // 2, i % 2)
        grid.addWidget(QLabel(tr("Drehen um 90°:", "Rotate by 90°:")), 2, 0, 1, 2)
        rg = QGridLayout()
        for j, nm in enumerate("XYZ"):
            for k, sgn in enumerate((+1, -1)):
                rg.addWidget(button(f"{nm} {'+' if sgn > 0 else '−'}90°",
                                    lambda a=j, s_=sgn: self._apply_transform(rot_matrix(a, 90 * s_), f"90° {'XYZ'[a]}")),
                             k, j)
        grid.addLayout(rg, 3, 0, 1, 2)
        s.add(grid)
        v.addWidget(s)

        s = Section(tr("Optimieren (aus erkannten Features)", "Optimise (from detected features)"))
        f = QFormLayout()
        self.opt_capture = dspin(6.0, 0.5, 30, 0.5, 1, "°")
        self.opt_straight = dspin(1.0, 0, 10, 0.1, 2, "°")
        self.opt_floor = QCheckBox(tr("Bodenebene auf z = 0", "floor plane to z = 0"))
        self.opt_floor.setChecked(True)
        f.addRow(tr("Fangwinkel zu X/Y/Z", "Capture angle to X/Y/Z"), self.opt_capture)
        f.addRow(tr("danach Achsen gerade stellen bis", "then straighten axes up to"), self.opt_straight)
        f.addRow(self.opt_floor)
        s.add(f)
        s.add(button(tr("Ausrichtung optimieren", "Optimise alignment"), self.optimize_align_ui, bold=True))
        s.add(hint(tr("Alle Ebenen-Normalen und Achsen, die schon fast auf X/Y/Z liegen, werden gemeinsam "
                      "(gewichtete Kabsch-Rotation) exakt darauf gedreht; anschließend werden Features innerhalb "
                      "des zweiten Winkels mit festgehaltener Richtung neu gefittet.",
                      "All plane normals and axes that are already nearly on X/Y/Z are rotated onto them together "
                      "(weighted Kabsch rotation); features within the second angle are then refitted with the "
                      "direction held fixed.")))
        v.addWidget(s)

        s = Section(tr("3-2-1: an Geometrie ausrichten", "3-2-1: align to geometry"), expanded=False)
        f = QFormLayout()
        self.al_z = QComboBox()
        self.al_x = QComboBox()
        self.al_o = combo([("zsrc", tr("Punkt der Z-Quelle", "point of the Z source")),
                           ("xsrc", tr("Punkt der X-Quelle (in Z-Ebene)", "point of the X source (in Z plane)")),
                           ("sel", tr("Schwerpunkt Auswahl (in Z-Ebene)", "selection centroid (in Z plane)")),
                           ("bbc", tr("Bounding-Box Mitte", "bounding box centre")),
                           ("bbb", tr("Bounding-Box Mitte unten", "bounding box bottom centre")),
                           ("bbmin", tr("Bounding-Box Ecke min", "bounding box min corner")),
                           ("cen", tr("Schwerpunkt Netz", "mesh centroid")),
                           ("keep", tr("unverändert", "unchanged"))])
        self.al_flip = QCheckBox(tr("Z umkehren", "flip Z"))
        self.al_flipx = QCheckBox(tr("X umkehren", "flip X"))
        f.addRow(tr("Z-Richtung aus", "Z direction from"), self.al_z)
        f.addRow(tr("X-Richtung aus", "X direction from"), self.al_x)
        f.addRow(tr("Ursprung", "Origin"), self.al_o)
        f.addRow(row(self.al_flip, self.al_flipx))
        s.add(f)
        s.add(button(tr("Ausrichten", "Align"), self.align_custom, bold=True))
        s.add(hint(tr("Quellen: Auswahl (Ebene/Zylinder wird automatisch gefittet), Features, Symmetrien "
                      "(Spiegelebene → Normale, Drehachse → Achse) oder aktuelle Achsen.",
                      "Sources: selection (plane/cylinder fitted automatically), features, symmetries "
                      "(mirror plane → normal, rotation axis → axis) or current axes.")))
        v.addWidget(s)

        s = Section(tr("Numerisch", "Numeric"), expanded=False)
        grid = QGridLayout()
        self.num_t = [dspin(0, -1e5, 1e5, 0.1, 3, "mm") for _ in range(3)]
        self.num_r = [dspin(0, -360, 360, 1, 2, "°") for _ in range(3)]
        grid.addWidget(QLabel(tr("verschieben", "translate")), 0, 1)
        grid.addWidget(QLabel(tr("drehen", "rotate")), 0, 2)
        for i, nm in enumerate("XYZ"):
            grid.addWidget(QLabel(nm), 1 + i, 0)
            grid.addWidget(self.num_t[i], 1 + i, 1)
            grid.addWidget(self.num_r[i], 1 + i, 2)
        grid.addWidget(button(tr("Verschieben", "Translate"), self.apply_numeric_translate), 4, 1)
        grid.addWidget(button(tr("Drehen (X→Y→Z)", "Rotate (X→Y→Z)"), self.apply_numeric_rotate), 4, 2)
        s.add(grid)
        v.addWidget(s)

        s = Section(tr("Anzeige && Lage", "Display && pose"), expanded=False)
        self.chk_origin = QCheckBox(tr("Achsenkreuz im Ursprung (X rot, Y grün, Z blau)", "Axes at origin (X red, Y green, Z blue)"))
        self.chk_origin.setChecked(True)
        self.chk_origin.toggled.connect(self._update_origin_axes)
        s.add(self.chk_origin)
        s.add(button(tr("Ursprüngliche Scan-Lage wiederherstellen", "Restore original scan pose"), self.restore_original_pose))
        self.lbl_pose = QLabel("")
        self.lbl_pose.setWordWrap(True)
        self.lbl_pose.setStyleSheet("color: #8a8f98;")
        s.add(self.lbl_pose)
        v.addWidget(s)
        v.addStretch(1)
        return w

    # ------------------------------------------------------------------ Grundfunktion
    def _apply_transform(self, T, label=None):
        if not self._need_mesh():
            return
        self.checkpoint(label or tr("Ausrichten", "Align"))
        self.doc.apply_transform(T)
        self.discard_fit(render=False)
        self.refresh_all(reset_camera=True)
        self.viewer.view("iso")

    def _update_origin_axes(self, *_):
        L = 0.25 * (self.model.diag if self.model is not None else 50)
        self.viewer.set_origin_axes(self.chk_origin.isChecked() and self.model is not None, L)

    def align_pca(self):
        if not self._need_mesh():
            return
        from core.mesh_ops import principal_axes_rotation
        R, c = principal_axes_rotation(self.model)
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = -R @ c
        self._apply_transform(T, tr("Hauptachsen", "Principal axes"))

    def center_centroid(self):
        if not self._need_mesh():
            return
        c = (self.model.centroids * self.model.areas[:, None]).sum(0) / self.model.areas.sum()
        T = np.eye(4)
        T[:3, 3] = -c
        self._apply_transform(T, tr("Schwerpunkt → Ursprung", "Centroid → origin"))

    def drop_to_floor(self):
        if not self._need_mesh():
            return
        T = np.eye(4)
        T[2, 3] = -self.model.V[:, 2].min()
        self._apply_transform(T, tr("Auf Boden legen", "Onto floor"))

    def center_xy(self):
        if not self._need_mesh():
            return
        mn, mx = self.model.bbox
        T = np.eye(4)
        T[:2, 3] = -0.5 * (mn[:2] + mx[:2])
        self._apply_transform(T, tr("Mitte XY", "Centre XY"))

    def apply_numeric_translate(self):
        T = np.eye(4)
        T[:3, 3] = [s.value() for s in self.num_t]
        self._apply_transform(T, tr("Verschieben", "Translate"))

    def apply_numeric_rotate(self):
        T = np.eye(4)
        for ax in range(3):
            if abs(self.num_r[ax].value()) > 0:
                T = rot_matrix(ax, self.num_r[ax].value()) @ T
        self._apply_transform(T, tr("Drehen", "Rotate"))

    def restore_original_pose(self):
        if not self._need_mesh():
            return
        self._apply_transform(np.linalg.inv(self.T_total), tr("Originallage", "Original pose"))

    # ------------------------------------------------------------------ Optimieren
    def optimize_align_ui(self):
        if not self._need_mesh():
            return
        cap, straight, floor = self.opt_capture.value(), self.opt_straight.value(), self.opt_floor.isChecked()

        def fn(d, rep):
            mn, mx = d.model.bbox
            T, r = optimize_alignment(d.features, cap, center=0.5 * (mn + mx))
            d.apply_transform(T)
            out = [tr(f"{r['n_used']}/{r['n_total']} Richtungen, Drehung {r['rotation_deg']:.3f}°, "
                      f"max. Abweichung {r['max_before']:.3f}° → {r['max_after']:.4f}°",
                      f"{r['n_used']}/{r['n_total']} directions, rotation {r['rotation_deg']:.3f}°, "
                      f"max deviation {r['max_before']:.3f}° → {r['max_after']:.4f}°")]
            if floor:
                Tf = floor_translation(d.features)
                if Tf is not None:
                    d.apply_transform(Tf)
            if straight > 0:
                n = 0
                for f in d.features:
                    if f.params.get("fixed") or f.kind not in ("plane", "cylinder", "cone", "torus"):
                        continue
                    try:
                        if snap_feature(d.model, f, straight, 0.0, robust_tol=2 * d.fit_tol):
                            n += 1
                    except Exception:
                        pass
                out.append(tr(f"{n} Features gerade gestellt", f"{n} features straightened"))
            return out
        self.run_doc(tr("Ausrichtung optimieren", "Optimise alignment"), fn,
                     after=lambda: self.viewer.view("iso"))

    # ------------------------------------------------------------------ 3-2-1
    def _refresh_align_combos(self):
        dir_kinds = ("plane", "cylinder", "cone", "torus", "extrusion", "revolve", "hole")
        for cb, is_x in ((self.al_z, False), (self.al_x, True)):
            cur = cb.currentData()
            cb.blockSignals(True)
            cb.clear()
            if is_x:
                cb.addItem(tr("– beliebig (aus aktuellem X) –", "– any (from current X) –"), "keep")
            cb.addItem(tr("Auswahl (Ebene/Zylinder automatisch)", "Selection (plane/cylinder automatic)"), "sel")
            for f in self.features:
                if f.kind in dir_kinds:
                    cb.addItem(tr("Feature: ", "Feature: ") + f.name, f"feat:{f.id}")
            for s in self.doc.symmetries:
                cb.addItem(tr("Symmetrie: ", "Symmetry: ") + s["name"], f"sym:{s['id']}")
            for i, nm in enumerate("XYZ"):
                cb.addItem(tr(f"aktuelle {nm}-Achse", f"current {nm} axis"), f"world:{i}")
            i = cb.findData(cur) if cur is not None else -1
            cb.setCurrentIndex(i if i >= 0 else 0)
            cb.blockSignals(False)

    def _feature_dir_origin(self, f):
        p, k = f.params, f.kind
        if k == "plane":
            return p["normal"], p["origin"]
        if k in ("cylinder", "revolve"):
            return p["axis"], p["point"]
        if k == "hole":
            return p["axis"], p["point"]
        if k == "cone":
            return p["axis"], p["apex"]
        if k == "torus":
            return p["axis"], p["center"]
        if k == "extrusion":
            return p["dir"], p["origin"] + p["z0"] * p["dir"]
        if k in ("sphere", "ellipsoid"):
            return None, p["center"]
        return None, None

    def _selection_dir_origin(self):
        ids = np.nonzero(self.sel)[0]
        if len(ids) < 10:
            raise RuntimeError(tr("Für „Auswahl“ zuerst eine Fläche wählen (z. B. Füllen F + Klick).",
                                  "For “selection” select a surface first (e.g. fill F + click)."))
        m = self.model
        C, N, A = m.centroids[ids], m.fit_normals[ids], m.areas[ids]
        fr, _ = auto_fit(C, N, A, tol=self.fit_tol.value(), kinds=["plane", "cylinder", "cone"])
        p = fr.params
        cen = (C * A[:, None]).sum(0) / A.sum()
        if fr.kind == "plane":
            o = cen - ((cen - p["origin"]) @ p["normal"]) * p["normal"]
            return unit(p["normal"]), o, KIND_LABELS["plane"]
        a = unit(p["axis"])
        base = p["point"] if fr.kind == "cylinder" else p["apex"]
        return a, base + ((cen - base) @ a) * a, KIND_LABELS[fr.kind]

    def _resolve_source(self, data):
        kind, _, val = str(data).partition(":")
        if kind == "sel":
            d, o, what = self._selection_dir_origin()
            return d, o, tr(f"Auswahl ({what})", f"selection ({what})")
        if kind == "feat":
            f = self.doc.feature(int(val))
            if f is None:
                raise RuntimeError(tr("Feature nicht gefunden.", "Feature not found."))
            d, o = self._feature_dir_origin(f)
            return (None if d is None else unit(d)), o, f.name
        if kind == "sym":
            s = next((x for x in self.doc.symmetries if x["id"] == int(val)), None)
            if s is None:
                raise RuntimeError(tr("Symmetrie nicht gefunden.", "Symmetry not found."))
            if s["type"] == "mirror":
                return unit(s["nrm"]), np.asarray(s["o"], float), s["name"]
            return unit(s["d"]), np.asarray(s["p"], float), s["name"]
        if kind == "world":
            return np.eye(3)[int(val)], None, tr(f"{'XYZ'[int(val)]}-Achse", f"{'XYZ'[int(val)]} axis")
        return None, None, "–"

    def align_custom(self):
        if not self._need_mesh():
            return
        try:
            z, oz, zname = self._resolve_source(self.al_z.currentData())
            if z is None:
                raise RuntimeError(tr("Z-Quelle liefert keine Richtung.", "Z source has no direction."))
            if self.al_flip.isChecked():
                z = -z
            xd = self.al_x.currentData() or "keep"
            x, ox, xname = self._resolve_source(xd) if xd != "keep" else (None, None, "–")
        except Exception as e:  # noqa: BLE001
            self._msg(str(e), 12000)
            return
        if x is None or abs(unit(x) @ z) > 0.995:
            x = np.eye(3)[0] if abs(z[0]) < 0.9 else np.eye(3)[1]
        x, y, z = orthonormal_frame(z, x)
        if self.al_flipx.isChecked():
            x, y = -x, -y
        R = np.vstack([x, y, z])
        mode = self.al_o.currentData()
        Vr = self.model.V @ R.T
        if mode == "keep":
            t = np.zeros(3)
        elif mode in ("bbc", "bbb", "bbmin"):
            mn, mx = Vr.min(0), Vr.max(0)
            o_new = {"bbc": 0.5 * (mn + mx), "bbb": np.r_[0.5 * (mn[:2] + mx[:2]), mn[2]], "bbmin": mn}[mode]
            t = -o_new
        elif mode == "cen":
            m = self.model
            c = (m.centroids * m.areas[:, None]).sum(0) / m.areas.sum()
            t = -R @ c
        else:
            pz = oz if oz is not None else np.zeros(3)
            if mode == "zsrc":
                p = pz
            elif mode == "xsrc":
                p = ox if ox is not None else pz
                p = p + ((pz - p) @ z) * z
            else:
                ids = np.nonzero(self.sel)[0]
                if len(ids) == 0:
                    self._msg(tr("Ursprung „Auswahl“: keine Auswahl.", "Origin “selection”: nothing selected."))
                    return
                c = self.model.centroids[ids].mean(0)
                p = c + ((pz - c) @ z) * z
            t = -R @ p
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = t
        self._apply_transform(T, tr("Ausrichten (3-2-1)", "Align (3-2-1)"))
        self._msg(tr(f"Ausgerichtet: Z aus {zname}, X aus {xname}", f"Aligned: Z from {zname}, X from {xname}"))
