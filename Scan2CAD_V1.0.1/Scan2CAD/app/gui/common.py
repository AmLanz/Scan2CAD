"""Kleine Qt-Hilfen, die alle Seiten benutzen."""
from __future__ import annotations

import numpy as np
from qtpy.QtCore import Qt
from qtpy.QtGui import QColor, QIcon, QPixmap
from qtpy.QtWidgets import (QComboBox, QDoubleSpinBox, QFormLayout, QGroupBox, QHBoxLayout, QLabel,
                            QLayout, QPushButton, QSpinBox, QToolButton, QVBoxLayout, QWidget)

SYN_RGB = np.array([205, 186, 238], np.uint8)       # künstliche Dreiecke (Füllungen): helles Violett


def dspin(val, mn, mx, step=0.01, dec=3, suffix=""):
    s = QDoubleSpinBox()
    s.setRange(mn, mx)
    s.setDecimals(dec)
    s.setSingleStep(step)
    s.setValue(val)
    if suffix:
        s.setSuffix(" " + suffix)
    s.setKeyboardTracking(False)
    return s


def ispin(val, mn, mx, step=1, suffix=""):
    s = QSpinBox()
    s.setRange(mn, mx)
    s.setSingleStep(step)
    s.setValue(val)
    if suffix:
        s.setSuffix(" " + suffix)
    return s


def combo(items, current=0):
    c = QComboBox()
    for key, label in items:
        c.addItem(label, key)
    if c.count():
        c.setCurrentIndex(current)
    return c


def set_combo(c, key):
    i = c.findData(key)
    if i >= 0:
        c.setCurrentIndex(i)


def hint(text):
    lab = QLabel(text)
    lab.setWordWrap(True)
    lab.setStyleSheet("color: #8a8f98;")
    return lab


def button(text, fn=None, tip=None, bold=False):
    b = QPushButton(text)
    if fn is not None:
        b.clicked.connect(lambda _=False: fn())
    if tip:
        b.setToolTip(tip)
    if bold:
        b.setStyleSheet("font-weight: bold;")
    return b


def row(*widgets, stretch_last=False):
    h = QHBoxLayout()
    h.setContentsMargins(0, 0, 0, 0)
    for w in widgets:
        if isinstance(w, int):
            h.addStretch(w)
        elif isinstance(w, QLayout):
            h.addLayout(w)
        else:
            h.addWidget(w)
    if stretch_last:
        h.addStretch(1)
    return h


def group(title, form=False):
    g = QGroupBox(title)
    lay = QFormLayout(g) if form else QVBoxLayout(g)
    return g, lay


class Section(QWidget):
    """Aufklappbarer Abschnitt (hält die Seiten übersichtlich)."""

    def __init__(self, title, expanded=True, parent=None):
        super().__init__(parent)
        self._btn = QToolButton()
        self._btn.setText(title)
        self._btn.setCheckable(True)
        self._btn.setChecked(expanded)
        self._btn.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self._btn.setArrowType(Qt.DownArrow if expanded else Qt.RightArrow)
        self._btn.setStyleSheet("QToolButton { border: none; font-weight: bold; padding: 4px 0; }")
        self._btn.toggled.connect(self._toggle)
        self.body = QWidget()
        self.lay = QVBoxLayout(self.body)
        self.lay.setContentsMargins(10, 0, 0, 6)
        v = QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(0)
        v.addWidget(self._btn)
        v.addWidget(self.body)
        self.body.setVisible(expanded)

    def _toggle(self, on):
        self._btn.setArrowType(Qt.DownArrow if on else Qt.RightArrow)
        self.body.setVisible(on)

    def expand(self, on=True):
        self._btn.setChecked(on)

    def add(self, w):
        if isinstance(w, QLayout):
            self.lay.addLayout(w)
        else:
            self.lay.addWidget(w)
        return w


def color_icon(rgb):
    pm = QPixmap(12, 12)
    pm.fill(QColor(int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255)))
    return QIcon(pm)


def rot_matrix(axis, deg):
    a = np.radians(deg)
    c, s = np.cos(a), np.sin(a)
    if axis == 0:
        R = np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    elif axis == 1:
        R = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    else:
        R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    T = np.eye(4)
    T[:3, :3] = R
    return T
