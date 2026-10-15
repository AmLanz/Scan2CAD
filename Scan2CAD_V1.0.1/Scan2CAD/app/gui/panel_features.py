"""Rechter Reiter · Features: Liste, Rolle, Reihenfolge, Neu-Fit, Einrasten, Anzeige."""
from __future__ import annotations

import numpy as np
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (QAbstractItemView, QCheckBox, QComboBox, QFormLayout, QGroupBox, QHeaderView,
                            QLabel, QMessageBox, QPlainTextEdit, QSlider, QTreeWidget, QTreeWidgetItem,
                            QVBoxLayout, QWidget)

from core import occ_build as ob
from core.features import ROLE_LABELS, ROLES, snap_feature
from core.fitting import KIND_LABELS
from core.i18n import tr

from .common import button, color_icon, dspin, row


class FeaturePanel:
    def _panel_features(self):
        w = QWidget()
        v = QVBoxLayout(w)
        self.tree = QTreeWidget()
        self.tree.setColumnCount(4)
        self.tree.setHeaderLabels([tr("Name", "Name"), tr("Typ", "Type"), tr("Rolle", "Role"), "RMS [mm]"])
        self.tree.setRootIsDecorated(False)
        self.tree.setSelectionMode(QAbstractItemView.ExtendedSelection)
        hdr = self.tree.header()
        hdr.setStretchLastSection(False)
        hdr.setSectionResizeMode(0, QHeaderView.Stretch)
        for c in (1, 2, 3):
            hdr.setSectionResizeMode(c, QHeaderView.ResizeToContents)
        self.tree.itemChanged.connect(self._tree_item_changed)
        self.tree.itemSelectionChanged.connect(self._tree_sel_changed)
        self.tree.itemDoubleClicked.connect(self._tree_double)
        v.addWidget(self.tree)
        self.role_combo = QComboBox()
        self.role_combo.activated.connect(self._role_chosen)
        v.addLayout(row(QLabel(tr("Rolle:", "Role:")), self.role_combo))
        v.addLayout(row(button("▲", lambda: self.move_feature(-1)), button("▼", lambda: self.move_feature(1)),
                        button(tr("Auswahl laden", "Load selection"), self.feature_to_selection),
                        button(tr("Löschen", "Delete"), self.delete_features)))
        v.addWidget(button(tr("Mit aktueller Auswahl neu fitten", "Refit with current selection"), self.refit_feature))
        g = QGroupBox(tr("Konstruktionsabsicht", "Design intent"))
        f = QFormLayout(g)
        self.snap_ang = dspin(3.0, 0.1, 30, 0.5, 1, "°")
        f.addRow(tr("Richtung auf X/Y/Z rasten bis", "Snap direction to X/Y/Z up to"), self.snap_ang)
        f.addRow(button(tr("Einrasten + Maße runden", "Snap + round dimensions"), self.snap_features))
        v.addWidget(g)
        self.chk_feat_vis = QCheckBox(tr("Feature-Geometrie zeigen", "Show feature geometry"))
        self.chk_feat_vis.setChecked(True)
        self.chk_feat_vis.toggled.connect(self.act_feat_vis.setChecked)
        self.act_feat_vis.toggled.connect(self.chk_feat_vis.setChecked)
        self.feat_opacity = QSlider(Qt.Horizontal)
        self.feat_opacity.setRange(5, 100)
        self.feat_opacity.setValue(70)
        self.feat_opacity.valueChanged.connect(lambda v_: self.viewer.overlay_opacity(v_ / 100))
        v.addLayout(row(self.chk_feat_vis, self.feat_opacity))
        self.feat_info = QPlainTextEdit()
        self.feat_info.setReadOnly(True)
        self.feat_info.setMaximumHeight(190)
        v.addWidget(self.feat_info)
        return w

    # ------------------------------------------------------------------ Overlays
    def _geom_key(self, f):
        """Inhaltsschlüssel der Feature-Geometrie. Rechenschritte arbeiten auf Kopien des Dokuments
        (neue Objekte) – ein Schlüssel über die Parameter erkennt unveränderte Features wieder,
        statt nach jedem Schritt alle Overlays neu zu tessellieren."""
        import hashlib
        import json
        from core.features import to_jsonable
        try:
            js = json.dumps(to_jsonable(f.params), sort_keys=True, default=str)
        except Exception:  # noqa: BLE001
            return (self.uid(f), f.version, f.role)
        return (f.kind, f.role, hashlib.md5(js.encode()).hexdigest())

    def _feature_mesh(self, f):
        defl = self._disp_defl()
        key = self._geom_key(f) + (round(float(defl), 5),)
        ent = self.feature_meshes.get(key)
        if ent is not None:
            return ent
        mesh = ob.tessellate(f.shape(), defl)
        if len(self.feature_meshes) > 2000:
            self.feature_meshes.clear()
        self.feature_meshes[key] = mesh
        return mesh

    def _overlay_props(self, f):
        op = self.feat_opacity.value() / 100
        col = f.color
        if f.kind == "hole":
            col, op = (0.95, 0.35, 0.3), op * 0.55
        elif f.role == "sub":
            op *= 0.6
        if f.params.get("fixed"):
            col = (0.75, 0.62, 0.95)
        return col, op

    def _add_feature_overlay(self, f, render=True):
        name = f"feat_{f.id}"
        if f.role == "cut":
            self.viewer.remove_overlay(name, render=render)
            return
        try:
            V, F = self._feature_mesh(f)
        except Exception as e:  # noqa: BLE001
            self._msg(tr(f"{f.name}: Geometrie fehlgeschlagen – {e}", f"{f.name}: geometry failed – {e}"))
            self.viewer.remove_overlay(name, render=False)
            return
        vis = f.visible and self.act_feat_vis.isChecked() and not self.step_preview_on
        col, op = self._overlay_props(f)
        self.viewer.set_overlay(name, V, F, color=col, opacity=op, visible=vis)
        act = self.viewer._overlays.get(name)
        if act is not None:
            act._s2c_key = (self._geom_key(f), col, round(op, 3))
        if render:
            self.viewer.render()

    def _sync_feature_overlays(self):
        ids = {f.id for f in self.features}
        for name in self.viewer.overlays_with_prefix("feat_"):
            if int(name[5:]) not in ids:
                self.viewer.remove_overlay(name, render=False)
        for f in self.features:
            act = self.viewer._overlays.get(f"feat_{f.id}")
            col, op = self._overlay_props(f)
            if act is not None and getattr(act, "_s2c_key", None) == (self._geom_key(f), col, round(op, 3)):
                act.SetVisibility(f.visible and self.act_feat_vis.isChecked() and not self.step_preview_on)
                continue
            self._add_feature_overlay(f, render=False)
        self.viewer.render()

    def _update_feature_visibility(self, *_):
        on = self.act_feat_vis.isChecked() and not self.step_preview_on
        for f in self.features:
            self.viewer.overlay_visible(f"feat_{f.id}", on and f.visible)
        self.viewer.render()

    # ------------------------------------------------------------------ Liste
    def refresh_tree(self, select_id=None):
        self.tree.blockSignals(True)
        self.tree.clear()
        for f in self.features:
            kind = KIND_LABELS.get(f.kind, f.kind)
            if f.params.get("fixed"):
                kind += tr(" (festgelegt)", " (fixed)")
            elif f.params.get("aux"):
                kind += tr(" (Hilfe)", " (aux)")
            elif f.auto:
                kind += " (auto)"
            rms = f.stats.get("dev_rms", f.stats.get("rms", None))
            it = QTreeWidgetItem([f.name, kind, ROLE_LABELS.get(f.role, f.role), "–" if rms is None else f"{rms:.4f}"])
            it.setFlags(it.flags() | Qt.ItemIsUserCheckable | Qt.ItemIsEditable)
            it.setCheckState(0, Qt.Checked if f.visible else Qt.Unchecked)
            it.setIcon(0, color_icon(f.color))
            it.setData(0, Qt.UserRole, f.id)
            if f.kind == "hole" and not f.params.get("bottom_seen", True) and f.params.get("bottom") != "through":
                it.setForeground(1, Qt.yellow)
                it.setToolTip(1, tr("Boden nicht erfasst", "bottom not scanned"))
            self.tree.addTopLevelItem(it)
            if select_id == f.id:
                it.setSelected(True)
        self.tree.blockSignals(False)
        self._tree_sel_changed()
        if hasattr(self, "_refresh_base_refs"):
            self._refresh_base_refs()
        if hasattr(self, "refresh_measure_refs"):
            self.refresh_measure_refs()

    def _selected_features(self):
        ids = [it.data(0, Qt.UserRole) for it in self.tree.selectedItems()]
        return [f for f in self.features if f.id in ids]

    def _tree_sel_changed(self):
        fs = self._selected_features()
        self.role_combo.blockSignals(True)
        self.role_combo.clear()
        if len(fs) == 1:
            f = fs[0]
            for r in ROLES[f.kind]:
                self.role_combo.addItem(ROLE_LABELS[r], r)
            self.role_combo.setCurrentIndex(ROLES[f.kind].index(f.role) if f.role in ROLES[f.kind] else 0)
            self.feat_info.setPlainText(f.summary() + "\n" + f"{len(f.face_ids):,} ".replace(",", ".")
                                        + tr("zugeordnete Dreiecke", "assigned triangles"))
        else:
            common = None
            for f in fs:
                common = set(ROLES[f.kind]) if common is None else common & set(ROLES[f.kind])
            for r in ["surface", "add", "sub", "cut"]:
                if common and r in common:
                    self.role_combo.addItem(ROLE_LABELS[r], r)
            self.role_combo.setCurrentIndex(-1)
            self.feat_info.setPlainText(tr(f"{len(fs)} Features gewählt", f"{len(fs)} features selected") if fs else "")
        self.role_combo.blockSignals(False)
        # Bohrung im Bohrungen-Menü mitführen
        if len(fs) == 1 and fs[0].kind == "hole" and hasattr(self, "bore_select_id"):
            self.bore_select_id(fs[0].id, from_tree=True)

    def _tree_double(self, it, col):
        f = self.doc.feature(it.data(0, Qt.UserRole))
        if f is None:
            return
        try:
            V, _ = self._feature_mesh(f)
            if len(V):
                self.viewer.focus(V.mean(0))
        except Exception:
            pass

    def _tree_item_changed(self, it, col):
        f = self.doc.feature(it.data(0, Qt.UserRole))
        if f is None or col != 0:
            return
        f.visible = it.checkState(0) == Qt.Checked
        f.name = it.text(0) or f.name
        self.viewer.overlay_visible(f"feat_{f.id}", f.visible and self.act_feat_vis.isChecked())
        self.viewer.render()

    def _role_chosen(self, i):
        role = self.role_combo.itemData(i)
        fs = [f for f in self._selected_features() if role in ROLES[f.kind] and f.role != role]
        if not fs:
            return
        self.checkpoint(tr(f"Rolle → {ROLE_LABELS[role]}", f"Role → {ROLE_LABELS[role]}"))
        for f in fs:
            f.set_role(role)
        self._sync_feature_overlays()
        self.refresh_tree()

    def move_feature(self, d):
        fs = self._selected_features()
        if len(fs) != 1:
            return
        i = self.features.index(fs[0])
        j = i + d
        if 0 <= j < len(self.features):
            self.checkpoint(tr("Reihenfolge ändern", "Change order"))
            self.features[i], self.features[j] = self.features[j], self.features[i]
            self.refresh_tree(select_id=fs[0].id)

    def delete_features(self):
        fs = self._selected_features()
        if not fs:
            return
        self.checkpoint(tr(f"{len(fs)} Feature(s) löschen", f"Delete {len(fs)} feature(s)"))
        ids = {f.id for f in fs}
        self.doc.features = [f for f in self.features if f.id not in ids]
        refs = {f"f:{i}" for i in ids}
        self.doc.constraints = [c for c in self.doc.constraints
                                if not any(r in refs or any(r.startswith("mid:") and f":{i}" in r for i in ids) for r in c["refs"])]
        self.refresh_all()

    def feature_to_selection(self):
        fs = self._selected_features()
        if not fs or self.model is None:
            return
        self.checkpoint(tr("Auswahl aus Feature", "Selection from feature"))
        self.sel[:] = False
        for f in fs:
            ids = np.asarray(f.face_ids, np.int64)
            self.sel[ids[ids < len(self.sel)]] = True
        self.recolor()

    def refit_feature(self):
        fs = self._selected_features()
        if len(fs) != 1 or not self._need_mesh():
            return
        f = fs[0]
        if f.params.get("fixed") or f.kind == "hole":
            self._msg(tr("Festgelegte Ebenen und Bohrungen werden nicht aus dem Scan gefittet (Bohrungen-Menü / "
                         "Bedingungen benutzen).", "Fixed planes and holes are not fitted from the scan (use the holes "
                                                   "menu / constraints)."))
            return
        ids = np.nonzero(self.sel)[0]
        if len(ids) < 3:
            ids = np.asarray(f.face_ids)
        if len(ids) < 3:
            self._msg(tr("Keine Flächen zum Neu-Fitten.", "No areas to refit."))
            return
        old_role = f.role

        def done(res):
            nf = res["feature"]
            nf.name = f.name
            if old_role in ROLES[nf.kind]:
                nf.set_role(old_role)
            self.checkpoint(tr(f"„{f.name}“ neu fitten", f"Refit “{f.name}”"))
            i = self.features.index(f)
            self.features[i] = nf
            self.sel[:] = False
            self.refresh_all()
            self.refresh_tree(select_id=nf.id)
        self._run(tr("Fitte neu…", "Refitting…"), self._fit_worker, done, self.model, ids, f.kind, self._fit_opts(),
                  self._disp_defl())

    def snap_features(self):
        fs = self._selected_features() or self.features
        if not fs:
            return
        self.checkpoint(tr("Einrasten / Maße runden", "Snap / round dimensions"))
        msgs = []
        for f in fs:
            if f.params.get("fixed") or f.kind == "hole":
                continue
            try:
                ch = snap_feature(self.model, f, self.snap_ang.value(), self.fit_snap.value(), robust_tol=self.fit_tol.value())
            except Exception as e:  # noqa: BLE001
                ch = [tr(f"Fehler: {e}", f"error: {e}")]
            if ch:
                msgs.append(f"{f.name}: " + "; ".join(ch))
        self._sync_feature_overlays()
        self.refresh_tree()
        if msgs:
            QMessageBox.information(self, tr("Einrasten", "Snap"), "\n".join(msgs))
        else:
            self.undo_stack.pop()
            self._update_history_ui()
            self._msg(tr("Nichts zu rasten.", "Nothing to snap."))
