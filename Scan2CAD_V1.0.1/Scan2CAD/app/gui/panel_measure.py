"""Rechter Reiter · Messen: zwei Klicks (oder zwei Features) → Winkel, Abstände, Durchmesser."""
from __future__ import annotations

import numpy as np
from qtpy.QtWidgets import QCheckBox, QFormLayout, QLabel, QPlainTextEdit, QVBoxLayout, QWidget

from core.analysis import feature_measurable, format_measure, local_measurable, measure
from core.i18n import tr

from .common import button, combo, dspin, hint, row


class MeasurePanel:
    def _panel_measure(self):
        w = QWidget()
        v = QVBoxLayout(w)
        v.addWidget(hint(tr("Werkzeug M, dann zwei Stellen anklicken. Gehört die Stelle zu einem Feature, wird dessen "
                            "exakte Geometrie benutzt (Ebene, Achse, Kugelmitte); sonst wird lokal eine Ebene bzw. ein "
                            "Zylinder gefittet. Ergebnis: Winkel zwischen Flächen/Achsen, Abstände, Ø.",
                            "Tool M, then click two places. If a place belongs to a feature, its exact geometry is used "
                            "(plane, axis, sphere centre); otherwise a plane or cylinder is fitted locally. Result: angle "
                            "between faces/axes, distances, Ø.")))
        self.chk_meas_feat = QCheckBox(tr("Feature-Geometrie bevorzugen", "prefer feature geometry"))
        self.chk_meas_feat.setChecked(True)
        v.addLayout(row(button(tr("Messen (M)", "Measure (M)"), lambda: self._set_tool("measure"), bold=True),
                        button(tr("Zurücksetzen", "Reset"), self.measure_reset), self.chk_meas_feat))
        self.meas_radius = dspin(0.0, 0.0, 100, 0.5, 2, "mm")
        self.meas_radius.setSpecialValueText(tr("automatisch", "automatic"))
        f0 = QFormLayout()
        f0.addRow(tr("Umkreis lokaler Fit", "Radius of local fit"), self.meas_radius)
        v.addLayout(f0)
        self.lbl_meas_a = QLabel("A: –")
        self.lbl_meas_b = QLabel("B: –")
        v.addWidget(self.lbl_meas_a)
        v.addWidget(self.lbl_meas_b)
        f = QFormLayout()
        self.meas_fa = combo([])
        self.meas_fb = combo([])
        f.addRow(tr("oder Feature A", "or feature A"), self.meas_fa)
        f.addRow(tr("Feature B", "feature B"), self.meas_fb)
        v.addLayout(f)
        v.addWidget(button(tr("Features messen", "Measure features"), self.measure_features))
        self.meas_out = QPlainTextEdit()
        self.meas_out.setReadOnly(True)
        v.addWidget(self.meas_out, 1)
        self._meas = []
        return w

    def refresh_measure_refs(self):
        for cb in (self.meas_fa, self.meas_fb):
            cur = cb.currentData()
            cb.blockSignals(True)
            cb.clear()
            for f in self.features:
                try:
                    feature_measurable(f)
                except Exception:
                    continue
                cb.addItem(f.name, f.id)
            i = cb.findData(cur)
            if i >= 0:
                cb.setCurrentIndex(i)
            cb.blockSignals(False)

    def measure_reset(self):
        self._meas = []
        self.lbl_meas_a.setText("A: –")
        self.lbl_meas_b.setText("B: –")
        self.viewer.remove_overlay("__measure")

    def _meas_label(self, m):
        t = {"plane": tr("Ebene", "plane"), "axis": tr("Achse", "axis"), "point": tr("Punkt", "point"),
             "sphere": tr("Kugel", "sphere")}[m["type"]]
        extra = f", Ø {2 * m['r']:.4f}" if m.get("r") else ""
        rms = f", RMS {m['rms']:.4f}" if m.get("rms") is not None else ""
        return f"{m.get('name', '')} [{t}{extra}{rms}]"

    def measure_click(self, cell, pos):
        fid = int(self.assigned[cell]) if len(self.assigned) > cell else 0
        m = None
        if fid and self.chk_meas_feat.isChecked():
            f = self.doc.feature(fid)
            try:
                m = feature_measurable(f)
            except Exception:
                m = None
        if m is None:
            m = local_measurable(self.model, cell, radius=self.meas_radius.value() or None, tol=self.fit_tol.value())
        m["click"] = np.asarray(pos, float)
        if len(self._meas) >= 2:
            self._meas = []
        self._meas.append(m)
        pts = np.array([x["click"] for x in self._meas])
        self.viewer.set_points("__measure", pts, color="#ffd60a", size=14)
        (self.lbl_meas_a if len(self._meas) == 1 else self.lbl_meas_b).setText(
            ("A: " if len(self._meas) == 1 else "B: ") + self._meas_label(m))
        if len(self._meas) == 1:
            self.lbl_meas_b.setText("B: –")
        if len(self._meas) == 2:
            self._show_measure(self._meas[0], self._meas[1])

    def measure_features(self):
        fa, fb = self.doc.feature(self.meas_fa.currentData()), self.doc.feature(self.meas_fb.currentData())
        if fa is None or fb is None:
            return
        a, b = feature_measurable(fa), feature_measurable(fb)
        self.lbl_meas_a.setText("A: " + self._meas_label(a))
        self.lbl_meas_b.setText("B: " + self._meas_label(b))
        self._show_measure(a, b)

    def _show_measure(self, a, b):
        try:
            rows = measure(a, b)
            self.meas_out.setPlainText(format_measure(rows))
        except Exception as e:  # noqa: BLE001
            self.meas_out.setPlainText(str(e))
