"""Hauptfenster: Rahmen, Verlauf (Rückgängig), Werkzeuge, Dateien, Sprache.

Die Arbeitsseiten liegen als Mixins in eigenen Modulen (page_*.py), die rechten Reiter in
panel_*.py. Alle Daten eines Projekts stecken in self.doc (core.document.Document).
Längere Rechnungen laufen im Hintergrund auf einer Kopie des Dokuments (run_doc) und werden
danach als ein Rückgängig-Schritt übernommen.
"""
from __future__ import annotations

import copy
import itertools
import os
import sys
import traceback

import numpy as np
from qtpy.QtCore import QProcess, QSettings, Qt
from qtpy.QtGui import QAction, QActionGroup, QKeySequence, QShortcut
from qtpy.QtWidgets import (QApplication, QComboBox, QDockWidget, QFileDialog, QLabel, QMainWindow,
                            QMessageBox, QProgressBar, QScrollArea, QTabWidget, QToolBar, QToolBox,
                            QToolButton)

from core import about
from core import preprocess as pp
from core.document import Document
from core.i18n import lang, tr
from core.project import load_project, save_project

from .about_dialog import AboutDialog, logo_icon
from .common import SYN_RGB
from .page_align import AlignPage
from .page_bores import BoresPage
from .page_detect import DetectPage
from .page_export import ExportPage
from .page_model import ModelPage
from .page_scan import ScanPage
from .page_symcon import SymConPage
from .panel_autopilot import AutopilotPanel
from .panel_features import FeaturePanel
from .panel_measure import MeasurePanel
from .viewer import BASE_RGB, SEL_RGB, Viewer
from .workers import TaskRunner

HISTORY_LIMIT = 60
HISTORY_MAX_MODELS = 6
_UID = itertools.count(1)

PAGES = ["scan", "align", "detect", "model", "bores", "symcon", "export"]


def tool_names():
    return {"navigate": tr("Navigieren", "Navigate"), "brush": tr("Pinsel", "Brush"),
            "erase": tr("Radierer", "Eraser"), "fill": tr("Füllen", "Fill"), "hole": tr("Lochrand", "Hole rim"),
            "bore": tr("Bohrung", "Hole tool"), "measure": tr("Messen", "Measure")}


def help_text():
    if lang() == "en":
        return """<h3>Workflow</h3><ol>
<li><b>Scan</b>: open STL/OBJ/PLY (mm). <i>Check</i> measures the scan noise and suggests tolerances.
Repair, close holes (each fill stays editable: refill, fair, force plane), close an open base
(walls are extrapolated to a plane, the floor becomes a fixed plane), correct regions, simplify.</li>
<li><b>Align</b>: quick (PCA, floor, 90°), 3-2-1 from features/selection/symmetries, numeric;
<i>Optimise</i> rotates so that detected planes/axes become exactly X/Y/Z.</li>
<li><b>Detect</b>: RANSAC shapes, merge similar, detect holes, remaining areas → freeform.</li>
<li><b>Model</b>: paint areas, fit a type, accept.</li>
<li><b>Holes</b>: step through all holes, correct axis/Ø/depth/bottom/countersink/thread, snap to
standard values. Tool K: click a hole wall.</li>
<li><b>Symmetry &amp; constraints</b>: detected symmetries; constraints (parallel, distance, coaxial,
axis in mid-plane, …) are enforced by a joint refit against the scan.</li>
<li><b>Export</b>: <i>scan topology</i> builds a closed solid (P = preview).</li></ol>
<p><b>✨ Feeling lucky</b> runs everything step by step; after each step you can look and intervene.</p>
<h3>Keys</h3><table>
<tr><td>N / Esc</td><td>navigate</td></tr><tr><td>B / E / F</td><td>brush / eraser / fill to edge</td></tr>
<tr><td>H</td><td>pick hole rim</td></tr><tr><td>K</td><td>hole tool (click a hole wall)</td></tr>
<tr><td>M</td><td>measure (two clicks)</td></tr><tr><td>[ / ]</td><td>brush smaller / larger</td></tr>
<tr><td>C</td><td>clear selection</td></tr><tr><td>Space / G / Enter</td><td>fit / grow / accept</td></tr>
<tr><td>P</td><td>STEP preview</td></tr><tr><td>R</td><td>reset camera</td></tr>
<tr><td>Ctrl+Z / Ctrl+Y</td><td>undo / redo</td></tr><tr><td>Ctrl+S / Ctrl+E</td><td>save project / export STEP</td></tr></table>
<p>Paint modes: left paints, Ctrl+left erases, right/Alt+left rotates. Pen tablets work like a mouse.</p>"""
    return """<h3>Ablauf</h3><ol>
<li><b>Scan</b>: STL/OBJ/PLY öffnen (mm). <i>Prüfen</i> misst das Scanrauschen und schlägt Toleranzen vor.
Reparieren, Löcher schließen (jede Füllung bleibt bearbeitbar: neu füllen, fairen, als Ebene
erzwingen), offenen Boden schließen (Wände werden bis zu einer Ebene verlängert, der Boden wird eine
festgelegte Ebene), Bereiche korrigieren, vereinfachen.</li>
<li><b>Ausrichten</b>: schnell (Hauptachsen, Boden, 90°), 3-2-1 aus Features/Auswahl/Symmetrien,
numerisch; <i>Optimieren</i> dreht so, dass erkannte Ebenen/Achsen exakt auf X/Y/Z liegen.</li>
<li><b>Erkennen</b>: Formen (RANSAC), Ähnliche zusammenführen, Bohrungen, Restflächen → Freiform.</li>
<li><b>Modellieren</b>: Bereiche anmalen, Typ fitten, übernehmen.</li>
<li><b>Bohrungen</b>: alle Bohrungen durchgehen, Achse/Ø/Tiefe/Boden/Senkung/Gewinde korrigieren,
auf Normwerte glätten. Werkzeug K: Bohrungswand anklicken.</li>
<li><b>Symmetrie &amp; Bedingungen</b>: erkannte Symmetrien; Bedingungen (parallel, Abstand, koaxial,
Achse in Mittelebene, …) werden per gemeinsamem Neu-Fit gegen den Scan erzwungen.</li>
<li><b>Export</b>: <i>Scan-Topologie</i> baut einen geschlossenen Volumenkörper (P = Vorschau).</li></ol>
<p><b>✨ Auf gut Glück</b> erledigt alles Schritt für Schritt; nach jedem Schritt kann man schauen und eingreifen.</p>
<h3>Tasten</h3><table>
<tr><td>N / Esc</td><td>Navigieren</td></tr><tr><td>B / E / F</td><td>Pinsel / Radierer / Füllen bis Kante</td></tr>
<tr><td>H</td><td>Lochrand wählen</td></tr><tr><td>K</td><td>Bohrungswerkzeug (Bohrungswand anklicken)</td></tr>
<tr><td>M</td><td>Messen (zwei Klicks)</td></tr><tr><td>[ / ]</td><td>Pinsel kleiner / größer</td></tr>
<tr><td>C</td><td>Auswahl leeren</td></tr><tr><td>Leertaste / G / Enter</td><td>Fitten / Erweitern / Übernehmen</td></tr>
<tr><td>P</td><td>STEP-Vorschau</td></tr><tr><td>R</td><td>Kamera zurücksetzen</td></tr>
<tr><td>Strg+Z / Strg+Y</td><td>Rückgängig / Wiederholen</td></tr><tr><td>Strg+S / Strg+E</td><td>Projekt speichern / STEP exportieren</td></tr></table>
<p>Mal-Modi: links malen, Strg+links radieren, rechts/Alt+links drehen. Grafiktablett wie Maus.</p>"""


class MainWindow(QMainWindow, ScanPage, AlignPage, DetectPage, ModelPage, BoresPage, SymConPage, ExportPage,
                 FeaturePanel, MeasurePanel, AutopilotPanel):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(tr(f"Scan2CAD {about.VERSION} – Scan → STEP mit Konstruktionsabsicht",
                               f"Scan2CAD {about.VERSION} – scan → STEP with design intent"))
        self.setWindowIcon(logo_icon())
        self.resize(1720, 1020)
        self.settings = QSettings("Scan2CAD", "Scan2CAD")
        self.doc = Document()
        self.sel = np.zeros(0, bool)
        self.assigned = np.zeros(0, np.int32)
        self.preview = None
        self.feature_meshes = {}
        self.undo_stack, self.redo_stack = [], []
        self.step_preview_on = False
        self._lut = np.zeros((1, 3), np.uint8)
        self._shown_model = None
        self.runner = TaskRunner(self)
        self.runner.busy_changed.connect(self._on_busy)
        self.runner.progress.connect(lambda t: self.lbl_busy.setText(t))

        self.viewer = Viewer(self)
        self.setCentralWidget(self.viewer)
        self.viewer.stroke.connect(self._on_stroke)
        self.viewer.picked.connect(self._on_pick)

        self._build_status()
        self._build_actions()
        self._build_left_dock()
        self._build_right_dock()
        self._build_menus()
        self._build_toolbar()
        self._build_shortcuts()
        self._set_tool("navigate")
        self._update_enabled()
        self._update_history_ui()

    # =================================================================== Kurzzugriffe
    @property
    def model(self):
        return self.doc.model

    @property
    def features(self):
        return self.doc.features

    @property
    def T_total(self):
        return self.doc.T_total

    def uid(self, f):
        if not hasattr(f, "_gui_uid"):
            f._gui_uid = next(_UID)
        return f._gui_uid

    # =================================================================== Aufbau
    def _build_left_dock(self):
        dock = QDockWidget(tr("Arbeitsschritte", "Workflow"), self)
        dock.setObjectName("workflow")
        dock.setFeatures(QDockWidget.DockWidgetMovable | QDockWidget.DockWidgetFloatable)
        self.toolbox = QToolBox()
        pages = [(self._page_scan, tr("1 · Scan", "1 · Scan")),
                 (self._page_align, tr("2 · Ausrichten", "2 · Align")),
                 (self._page_detect, tr("3 · Erkennen", "3 · Detect")),
                 (self._page_model, tr("4 · Modellieren", "4 · Model")),
                 (self._page_bores, tr("5 · Bohrungen", "5 · Holes")),
                 (self._page_symcon, tr("6 · Symmetrie && Bedingungen", "6 · Symmetry && constraints")),
                 (self._page_export, tr("7 · Export", "7 · Export"))]
        for fn, title in pages:
            sc = QScrollArea()
            sc.setWidgetResizable(True)
            sc.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
            sc.setWidget(fn())
            self.toolbox.addItem(sc, title)
        self.toolbox.currentChanged.connect(self._page_changed)
        dock.setWidget(self.toolbox)
        dock.setMinimumWidth(460)
        self.addDockWidget(Qt.LeftDockWidgetArea, dock)
        self.left_dock = dock

    def goto_page(self, key):
        self.toolbox.setCurrentIndex(PAGES.index(key))

    def _page_changed(self, i):
        key = PAGES[i] if 0 <= i < len(PAGES) else ""
        if key != "symcon" and hasattr(self, "viewer"):
            self.viewer.clear_overlays("__sym_")
        if key != "bores" and hasattr(self, "viewer"):
            self.viewer.remove_overlay("__bore_axis")
        if key == "bores":
            self.refresh_bores()
        elif key == "symcon":
            self.refresh_symcon()
        elif key == "scan":
            self.refresh_fills()

    def _build_right_dock(self):
        dock = QDockWidget(tr("Modell", "Model"), self)
        dock.setObjectName("panels")
        dock.setFeatures(QDockWidget.DockWidgetMovable | QDockWidget.DockWidgetFloatable)
        self.right_tabs = QTabWidget()
        self.right_tabs.addTab(self._panel_features(), tr("Features", "Features"))
        self.right_tabs.addTab(self._panel_measure(), tr("Messen", "Measure"))
        self.right_tabs.addTab(self._panel_autopilot(), tr("✨ Auf gut Glück", "✨ Feeling lucky"))
        dock.setWidget(self.right_tabs)
        dock.setMinimumWidth(400)
        self.addDockWidget(Qt.RightDockWidgetArea, dock)
        self.right_dock = dock

    def showEvent(self, ev):
        super().showEvent(ev)
        if not getattr(self, "_docks_sized", False):
            self._docks_sized = True
            self.resizeDocks([self.left_dock, self.right_dock], [480, 430], Qt.Horizontal)

    def _build_actions(self):
        self.act_undo = QAction(tr("↶ Rückgängig", "↶ Undo"), self)
        self.act_undo.setShortcuts([QKeySequence("Ctrl+Z")])
        self.act_undo.triggered.connect(self.undo)
        self.act_redo = QAction(tr("↷ Wiederholen", "↷ Redo"), self)
        self.act_redo.setShortcuts([QKeySequence("Ctrl+Y"), QKeySequence("Ctrl+Shift+Z")])
        self.act_redo.triggered.connect(self.redo)
        self.act_step_prev = QAction(tr("STEP-Vorschau (P)", "STEP preview (P)"), self, checkable=True)
        self.act_step_prev.toggled.connect(self.toggle_step_preview)
        self.act_mesh_vis = QAction(tr("Scan-Netz", "Scan mesh"), self, checkable=True)
        self.act_mesh_vis.setChecked(True)
        self.act_mesh_vis.toggled.connect(self._mesh_vis_toggled)
        self.act_feat_vis = QAction(tr("Features", "Features"), self, checkable=True)
        self.act_feat_vis.setChecked(True)
        self.act_feat_vis.toggled.connect(lambda _: self._update_feature_visibility())
        self.act_hide_assigned = QAction(tr("Zugeordnete Flächen ausgrauen", "Grey out assigned areas"), self,
                                         checkable=True)
        self.act_hide_assigned.toggled.connect(lambda _: self.recolor())
        self.act_lucky = QAction(tr("✨ Auf gut Glück", "✨ Feeling lucky"), self)
        self.act_lucky.setToolTip(tr("Alles automatisch – Schritt für Schritt mit Pausen zum Eingreifen",
                                     "Everything automatically – step by step with pauses to intervene"))
        self.act_lucky.triggered.connect(self.lucky_start)

    def _build_menus(self):
        mb = self.menuBar()
        m = mb.addMenu(tr("&Datei", "&File"))
        for label, fn, sc in ((tr("Netz öffnen…", "Open mesh…"), lambda: self.open_mesh(), "Ctrl+O"),
                              (tr("Projekt öffnen…", "Open project…"), lambda: self.load_project(), "Ctrl+Shift+O"),
                              (tr("Projekt speichern…", "Save project…"), self.save_project, "Ctrl+S"),
                              (tr("STEP exportieren…", "Export STEP…"), self.export_step, "Ctrl+E"),
                              (None, None, None),
                              (tr("Beenden", "Quit"), self.close, "Ctrl+Q")):
            if label is None:
                m.addSeparator()
                continue
            a = QAction(label, self)
            a.triggered.connect(fn)
            if sc:
                a.setShortcut(QKeySequence(sc))
            m.addAction(a)
        m = mb.addMenu(tr("&Bearbeiten", "&Edit"))
        m.addAction(self.act_undo)
        m.addAction(self.act_redo)
        m = mb.addMenu(tr("&Ansicht", "&View"))
        for label, key in ((tr("Isometrisch", "Isometric"), "iso"), (tr("Von +X", "From +X"), "+x"),
                           (tr("Von −X", "From −X"), "-x"), (tr("Von +Y", "From +Y"), "+y"),
                           (tr("Von −Y", "From −Y"), "-y"), (tr("Von oben (+Z)", "Top (+Z)"), "+z"),
                           (tr("Von unten (−Z)", "Bottom (−Z)"), "-z")):
            a = QAction(label, self)
            a.triggered.connect(lambda _=False, k=key: self.viewer.view(k))
            m.addAction(a)
        a = QAction(tr("Kamera zurücksetzen", "Reset camera"), self)
        a.setShortcut(QKeySequence("R"))
        a.triggered.connect(self.viewer.reset_camera)
        m.addAction(a)
        m.addSeparator()
        self.act_parallel = QAction(tr("Parallelprojektion", "Parallel projection"), self, checkable=True)
        self.act_parallel.toggled.connect(self.viewer.set_parallel)
        m.addAction(self.act_parallel)
        self.act_edges = QAction(tr("Dreieckskanten zeigen", "Show triangle edges"), self, checkable=True)
        self.act_edges.toggled.connect(self.viewer.set_edges)
        m.addAction(self.act_edges)
        m.addAction(self.act_hide_assigned)
        m.addAction(self.act_mesh_vis)
        m.addAction(self.act_feat_vis)
        m = mb.addMenu(tr("&Sprache", "&Language"))
        grp = QActionGroup(self)
        for code, label in (("de", "Deutsch"), ("en", "English")):
            a = QAction(label, self, checkable=True)
            a.setChecked(lang() == code)
            a.triggered.connect(lambda _=False, c=code: self.set_language(c))
            grp.addAction(a)
            m.addAction(a)
        m = mb.addMenu(tr("&Hilfe", "&Help"))
        a = QAction(tr("Ablauf && Tasten", "Workflow && keys"), self)
        a.setShortcut(QKeySequence("F1"))
        a.triggered.connect(self.show_help)
        m.addAction(a)
        m.addSeparator()
        a = QAction(tr("Über Scan2CAD / Zitieren…", "About Scan2CAD / Cite…"), self)
        a.triggered.connect(self.show_about)
        m.addAction(a)
        # Knopf in der rechten oberen Ecke
        btn = QToolButton(self)
        btn.setText(tr(f"Über / Zitieren · v{about.VERSION}", f"About / Cite · v{about.VERSION}"))
        btn.setIcon(logo_icon())
        btn.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        btn.setAutoRaise(True)
        btn.setToolTip(tr("Version, Lizenz, Autor – und wie man Scan2CAD zitiert",
                          "Version, license, author – and how to cite Scan2CAD"))
        btn.clicked.connect(self.show_about)
        mb.setCornerWidget(btn, Qt.TopRightCorner)
        self.btn_about = btn

    def _build_toolbar(self):
        tb = QToolBar(tr("Werkzeuge", "Tools"), self)
        tb.setObjectName("tools")
        tb.setMovable(False)
        self.addToolBar(Qt.TopToolBarArea, tb)
        a = QAction(tr("Öffnen", "Open"), self)
        a.triggered.connect(lambda: self.open_mesh())
        tb.addAction(a)
        a = QAction(tr("Speichern", "Save"), self)
        a.triggered.connect(self.save_project)
        tb.addAction(a)
        tb.addSeparator()
        tb.addAction(self.act_undo)
        tb.addAction(self.act_redo)
        tb.addSeparator()
        self.tool_actions = {}
        grp = QActionGroup(self)
        grp.setExclusive(True)
        keys = {"navigate": "N", "brush": "B", "erase": "E", "fill": "F", "hole": "H", "bore": "K", "measure": "M"}
        for key in ("navigate", "brush", "erase", "fill", "hole", "bore", "measure"):
            a = QAction(f"{tool_names()[key]} ({keys[key]})", self, checkable=True)
            a.triggered.connect(lambda _=False, k=key: self._set_tool(k))
            grp.addAction(a)
            tb.addAction(a)
            self.tool_actions[key] = a
        tb.addSeparator()
        tb.addAction(self.act_lucky)
        w = tb.widgetForAction(self.act_lucky)
        if w is not None:
            w.setStyleSheet("font-weight: bold; color: #ffd24a;")
        tb.addSeparator()
        tb.addAction(self.act_step_prev)
        tb.addAction(self.act_mesh_vis)
        tb.addAction(self.act_feat_vis)
        tb.addSeparator()
        for label, key in ((tr("Iso", "Iso"), "iso"), (tr("Oben", "Top"), "+z"), (tr("Vorne", "Front"), "-y"),
                           (tr("Rechts", "Right"), "+x")):
            a = QAction(label, self)
            a.triggered.connect(lambda _=False, k=key: self.viewer.view(k))
            tb.addAction(a)
        tb.addSeparator()
        self.lang_combo = QComboBox()
        self.lang_combo.addItem("DE", "de")
        self.lang_combo.addItem("EN", "en")
        self.lang_combo.setCurrentIndex(0 if lang() == "de" else 1)
        self.lang_combo.setToolTip(tr("Sprache der Oberfläche", "Interface language"))
        self.lang_combo.activated.connect(lambda i: self.set_language(self.lang_combo.itemData(i)))
        tb.addWidget(self.lang_combo)

    def _build_shortcuts(self):
        def sc(key, fn):
            s = QShortcut(QKeySequence(key), self)
            s.setContext(Qt.WindowShortcut)
            s.activated.connect(fn)
        sc("N", lambda: self._set_tool("navigate"))
        sc("Escape", lambda: self._set_tool("navigate"))
        sc("B", lambda: self._set_tool("brush"))
        sc("E", lambda: self._set_tool("erase"))
        sc("F", lambda: self._set_tool("fill"))
        sc("H", lambda: self._set_tool("hole"))
        sc("K", lambda: self._set_tool("bore"))
        sc("M", lambda: self._set_tool("measure"))
        sc("P", lambda: self.act_step_prev.toggle())
        sc("C", self.sel_clear)
        sc("[", lambda: self.brush_spin.setValue(self.brush_spin.value() / 1.25))
        sc("]", lambda: self.brush_spin.setValue(self.brush_spin.value() * 1.25))
        sc("Space", self.do_fit)
        sc("G", self.grow_fit)
        sc("Return", self.accept_fit)
        sc("Enter", self.accept_fit)

    def _build_status(self):
        sb = self.statusBar()
        self.prog = QProgressBar()
        self.prog.setRange(0, 0)
        self.prog.setMaximumWidth(160)
        self.prog.setVisible(False)
        self.lbl_busy = QLabel("")
        sb.addPermanentWidget(self.lbl_busy)
        sb.addPermanentWidget(self.prog)

    # =================================================================== Sprache / Hilfe
    def set_language(self, code):
        if code == lang():
            return
        self.settings.setValue("lang", code)
        ans = QMessageBox.question(
            self, tr("Sprache", "Language"),
            tr("Die Sprache wird nach einem Neustart gewechselt. Jetzt neu starten? Der aktuelle Stand wird "
               "mitgenommen (der Rückgängig-Verlauf nicht).",
               "The language changes after a restart. Restart now? The current state is carried over "
               "(the undo history is not)."))
        if ans != QMessageBox.Yes:
            return
        args = [a for a in sys.argv if not a.startswith("--lang") and not a.lower().endswith((".s2c", ".stl", ".obj", ".ply"))]
        args.append(f"--lang={code}")
        if self.model is not None:
            import tempfile
            tmp = os.path.join(tempfile.gettempdir(), "scan2cad_sprachwechsel.s2c")
            try:
                save_project(tmp, self.model, self.features, self.sel,
                             {"mesh_path": self.doc.mesh_path, "T_total": self.T_total.tolist(), "tol": self.doc.tol},
                             self.doc.symmetries, self.doc.constraints)
                args.append(tmp)
            except Exception as e:  # noqa: BLE001
                self._error(f"{e}\n\n{traceback.format_exc()}")
                return
        QProcess.startDetached(sys.executable, args)
        self.close()

    def show_help(self):
        QMessageBox.information(self, tr("Hilfe", "Help"), help_text())

    def show_about(self):
        dlg = AboutDialog(self)
        self._about_dialog = dlg
        dlg.exec()

    def _msg(self, text, timeout=8000):
        self.statusBar().showMessage(text, timeout)

    def _error(self, msg):
        short = msg.split("\n\n")[0]
        box = QMessageBox(QMessageBox.Warning, tr("Fehler", "Error"), short, parent=self)
        if "\n\n" in msg:
            box.setDetailedText(msg)
        box.exec()
        if hasattr(self, "lucky_on_error"):
            self.lucky_on_error()

    # =================================================================== Hintergrund
    def _on_busy(self, busy, label):
        self.prog.setVisible(busy)
        self.lbl_busy.setText(label if busy else "")
        self.left_dock.widget().setEnabled(not busy)
        self.viewer.set_busy(busy)
        if busy:
            QApplication.setOverrideCursor(Qt.WaitCursor)
        else:
            QApplication.restoreOverrideCursor()

    def _run(self, label, fn, on_done, *args, **kwargs):
        if self.runner.busy:
            self._msg(tr("Bitte warten – eine Berechnung läuft noch.", "Please wait – a computation is running."))
            return False
        return self.runner.run(label, fn, on_done, *args, on_error=self._error, **kwargs)

    def run_doc(self, label, fn, done_msg=None, after=None, *args):
        """fn(doc_kopie, report, *args) -> Liste von Meldungen; Ergebnis wird als ein Schritt übernommen."""
        if not self._need_mesh():
            return False
        work_doc = self.doc.clone()

        def work():
            out = fn(work_doc, self.runner.report, *args)
            return work_doc, out

        def done(res):
            d, out = res
            self.apply_doc(d, label)
            lines = [str(x) for x in (out or [])]
            if lines:
                self._msg(" · ".join(lines)[:400], 20000)
            if done_msg:
                done_msg(lines)
            if after:
                after()
        return self._run(label + "…", work, done)

    def _need_mesh(self):
        if self.model is None:
            self._msg(tr("Zuerst ein Netz laden.", "Load a mesh first."))
            return False
        return True

    def _update_enabled(self):
        has = self.model is not None
        for b in (getattr(self, "btn_fit", None), getattr(self, "btn_grow", None)):
            if b is not None:
                b.setEnabled(has)
        if hasattr(self, "btn_accept"):
            self.btn_accept.setEnabled(self.preview is not None)
            self.btn_discard.setEnabled(self.preview is not None)

    def _disp_defl(self):
        return max(0.005, 0.0015 * (self.model.diag if self.model is not None else 100))

    def _mesh_vis_toggled(self, on):
        if not self.step_preview_on:
            self.viewer.set_mesh_visible(on)

    # =================================================================== Verlauf
    def _state(self):
        d = self.doc
        return {"model": d.model, "features": [(f, f.get_state()) for f in d.features],
                "syms": copy.deepcopy(d.symmetries), "cons": copy.deepcopy(d.constraints),
                "sel": np.packbits(self.sel) if len(self.sel) else None, "nsel": len(self.sel),
                "T": d.T_total.copy(), "tol": dict(d.tol)}

    def checkpoint(self, label):
        """Zustand VOR einer Änderung sichern."""
        if self.model is None:
            return
        if self.step_preview_on:
            self.act_step_prev.setChecked(False)
        self.undo_stack.append((label, self._state()))
        if len(self.undo_stack) > HISTORY_LIMIT:
            self.undo_stack.pop(0)
        while True:
            models = []
            for _, st in self.undo_stack:
                if all(st["model"] is not m for m in models):
                    models.append(st["model"])
            if len(models) <= HISTORY_MAX_MODELS:
                break
            self.undo_stack.pop(0)
        self.redo_stack.clear()
        self._update_history_ui()

    def undo(self):
        if not self.undo_stack or self.runner.busy:
            return
        label, st = self.undo_stack.pop()
        self.redo_stack.append((label, self._state()))
        self._restore(st)
        self._msg(tr(f"Rückgängig: {label}", f"Undone: {label}"))
        self._update_history_ui()

    def redo(self):
        if not self.redo_stack or self.runner.busy:
            return
        label, st = self.redo_stack.pop()
        self.undo_stack.append((label, self._state()))
        self._restore(st)
        self._msg(tr(f"Wiederholt: {label}", f"Redone: {label}"))
        self._update_history_ui()

    def _restore(self, st):
        if self.step_preview_on:
            self.act_step_prev.setChecked(False)
        self.discard_fit(render=False)
        d = self.doc
        d.model = st["model"]
        feats = []
        for f, fs in st["features"]:
            f.set_state(fs)
            feats.append(f)
        d.features = feats
        d.symmetries = copy.deepcopy(st["syms"])
        d.constraints = copy.deepcopy(st["cons"])
        d.T_total = st["T"]
        d.tol = dict(st["tol"])
        d.preview = None
        n = st["nsel"]
        self.sel = np.unpackbits(st["sel"])[:n].astype(bool) if st["sel"] is not None else np.zeros(n, bool)
        self.refresh_all()

    def _update_history_ui(self):
        self.act_undo.setEnabled(bool(self.undo_stack))
        self.act_redo.setEnabled(bool(self.redo_stack))
        self.act_undo.setToolTip(tr("Rückgängig: ", "Undo: ") + (self.undo_stack[-1][0] if self.undo_stack else "–") + " (Ctrl+Z)")
        self.act_redo.setToolTip(tr("Wiederholen: ", "Redo: ") + (self.redo_stack[-1][0] if self.redo_stack else "–") + " (Ctrl+Y)")

    def _clear_history(self):
        self.undo_stack.clear()
        self.redo_stack.clear()
        self._update_history_ui()

    # =================================================================== Dokument übernehmen / anzeigen
    def apply_doc(self, new_doc, label):
        self.checkpoint(label)
        old_model = self.doc.model
        pose_changed = not np.allclose(self.doc.T_total, new_doc.T_total)
        self.doc = new_doc
        if new_doc.model is not old_model:
            n_old = len(self.sel)
            if new_doc.model is not None and old_model is not None and new_doc.model.n_faces >= n_old \
                    and np.array_equal(new_doc.model.F[:n_old], old_model.F):
                self.sel = np.r_[self.sel, np.zeros(new_doc.model.n_faces - n_old, bool)]
            else:
                self.sel = np.zeros(new_doc.model.n_faces if new_doc.model is not None else 0, bool)
        self.refresh_all(reset_camera=pose_changed)
        if pose_changed:
            self.viewer.view("iso")
        if new_doc.preview is not None:
            self.show_scan_preview(new_doc.preview)

    def refresh_all(self, reset_camera=False):
        m = self.model
        if m is not None and len(self.sel) != m.n_faces:
            self.sel = np.zeros(m.n_faces, bool)
        if m is not self._shown_model or reset_camera:
            self._shown_model = m
            self._holes = None
            self._hole_sel = None
            self.viewer.hide_scalars(render=False)
            self._rebuild_assigned()
            if m is not None:
                self.viewer.set_mesh(m.V, m.F, self._face_colors(), reset_camera=reset_camera)
        self._sync_feature_overlays()
        self._rebuild_assigned()
        self.recolor(render=False)
        self.refresh_tree()
        self._update_info()
        self._update_origin_axes()
        self._refresh_holes()
        self.refresh_fills()
        self.refresh_bores()
        self.refresh_symcon()
        self._refresh_align_combos()
        self._update_enabled()
        self.viewer.render()

    # =================================================================== Farben
    def _rebuild_assigned(self):
        n = self.model.n_faces if self.model is not None else 0
        self.assigned = np.zeros(n, np.int32)
        max_id = max([f.id for f in self.features], default=0)
        lut = np.tile(BASE_RGB, (max_id + 1, 1)).astype(np.float64)
        for f in self.features:
            ids = np.asarray(f.face_ids, np.int64)
            ids = ids[(ids >= 0) & (ids < n)]
            self.assigned[ids] = f.id
            lut[f.id] = 0.55 * np.array(f.color) * 255 + 0.45 * BASE_RGB
        self._lut = np.clip(lut, 0, 255).astype(np.uint8)

    def _face_colors(self, idx=None):
        n = len(self.sel)
        if idx is None:
            idx = np.arange(n)
        a = self.assigned[idx]
        if self.act_hide_assigned.isChecked():
            col = np.where((a > 0)[:, None], np.array([95, 98, 104], np.uint8), BASE_RGB).astype(np.uint8)
        else:
            col = self._lut[a]
        if self.model is not None and len(self.model.synthetic) == self.model.n_faces:
            syn = self.model.synthetic[idx] & (a == 0)
            col = np.where(syn[:, None], SYN_RGB, col).astype(np.uint8)
        col = np.where(self.sel[idx][:, None], SEL_RGB, col).astype(np.uint8)
        return col

    def recolor(self, idx=None, render=True):
        if self.model is None or len(self.sel) != self.model.n_faces:
            return
        if idx is None:
            self.viewer.set_colors(self._face_colors())
        elif len(idx):
            self.viewer.set_colors(self._face_colors(idx), idx)
        if hasattr(self, "lbl_sel"):
            self.lbl_sel.setText(tr("Auswahl: ", "Selection: ") + f"{int(self.sel.sum()):,} ".replace(",", ".")
                                 + tr("Dreiecke", "triangles"))
        if render:
            self.viewer.render(throttle=idx is not None)

    # =================================================================== Werkzeuge
    def _set_tool(self, tool, from_ui=False):
        self.viewer.set_tool(tool)
        b = getattr(self, "tool_buttons", {}).get(tool)
        if b is not None and not from_ui:
            b.blockSignals(True)
            b.setChecked(True)
            b.blockSignals(False)
        elif b is None and hasattr(self, "tool_group"):
            self.tool_group.setExclusive(False)
            for rb in self.tool_buttons.values():
                rb.blockSignals(True)
                rb.setChecked(False)
                rb.blockSignals(False)
            self.tool_group.setExclusive(True)
        a = getattr(self, "tool_actions", {}).get(tool)
        if a is not None:
            a.setChecked(True)
        self._msg(tr("Werkzeug: ", "Tool: ") + tool_names()[tool], 3000)
        if tool in ("brush", "erase", "fill") and self.toolbox.currentIndex() not in (1, 2, 3, 4, 0):
            self.goto_page("model")
        if tool == "hole":
            self.goto_page("scan")
            if not self.chk_rims.isChecked():
                self.chk_rims.setChecked(True)
        if tool == "bore":
            self.goto_page("bores")
            self._msg(tr("Bohrungswand (sauberen Teil) anklicken – Achse und Ø werden daraus bestimmt.",
                         "Click a hole wall (clean part) – axis and Ø are derived from it."), 10000)
        if tool == "measure":
            self.right_tabs.setCurrentIndex(1)
            self.measure_reset()

    def _on_pick(self, ev):
        if self.model is None:
            return
        t = ev["tool"]
        if t == "hole":
            self._pick_hole(ev["pos"])
        elif t == "fill":
            self._pick_fill(ev)
        elif t == "bore":
            self.bore_from_click(ev["cell"])
        elif t == "measure":
            self.measure_click(ev["cell"], ev["pos"])

    # =================================================================== Dateien
    def open_mesh(self, path=None):
        if path is None:
            last = self.settings.value("last_dir", "")
            path, _ = QFileDialog.getOpenFileName(self, tr("Netz öffnen", "Open mesh"), last,
                                                  tr("Netze", "Meshes") + " (*.stl *.STL *.obj *.ply *.vtk *.vtp);;"
                                                  + tr("Alle", "All") + " (*)")
            if not path:
                return
        self.settings.setValue("last_dir", os.path.dirname(path))

        def done(m):
            if self.step_preview_on:
                self.act_step_prev.setChecked(False)
            self.doc = Document(m, mesh_path=path)
            self.feature_meshes = {}
            self.viewer.clear_overlays()
            self.sel = np.zeros(m.n_faces, bool)
            self._shown_model = None
            self.refresh_all(reset_camera=True)
            self._clear_history()
            self._msg(tr(f"Geladen: {os.path.basename(path)}", f"Loaded: {os.path.basename(path)}"))
            self.goto_page("scan")
            self._suggest_params()
            self.lucky_reset()
        self._run(tr("Lade Netz…", "Loading mesh…"), pp.load_mesh, done, path)

    def _update_info(self):
        m = self.model
        if m is None:
            self.lbl_info.setText(tr("Kein Netz geladen.", "No mesh loaded."))
            if hasattr(self, "lbl_pose"):
                self.lbl_pose.setText("")
            return
        mn, mx = m.bbox
        ext = mx - mn
        closed = tr("geschlossen", "closed") if m.is_closed else \
            tr(f"offen ({m.boundary_edge_count} Randkanten)", f"open ({m.boundary_edge_count} boundary edges)")
        nsyn = int(m.synthetic.sum())
        self.lbl_info.setText(
            f"<b>{os.path.basename(self.doc.mesh_path or tr('Projekt', 'project'))}</b><br>"
            + f"{m.n_faces:,} ".replace(",", ".") + tr("Dreiecke", "triangles") + f", {m.n_verts:,} ".replace(",", ".")
            + tr("Punkte", "points") + f" – {closed}<br>"
            + tr("Ausdehnung", "Extent") + f": {ext[0]:.2f} × {ext[1]:.2f} × {ext[2]:.2f} mm<br>"
            + tr("mittl. Kantenlänge", "mean edge length") + f": {m.mean_edge_length():.3f} mm"
            + (("<br>" + tr("künstlich (Füllungen)", "synthetic (fills)") + f": {nsyn:,} ".replace(",", ".")
                + tr("Dreiecke", "triangles")) if nsyn else ""))
        R = self.T_total[:3, :3]
        t = self.T_total[:3, 3]
        ang = np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)))
        if hasattr(self, "lbl_pose"):
            self.lbl_pose.setText(tr(f"Gegenüber der Original-Scanlage: Drehung {ang:.2f}°, "
                                     f"Verschiebung ({t[0]:.2f}, {t[1]:.2f}, {t[2]:.2f}) mm",
                                     f"Relative to the original scan pose: rotation {ang:.2f}°, "
                                     f"translation ({t[0]:.2f}, {t[1]:.2f}, {t[2]:.2f}) mm"))

    def save_project(self):
        if not self._need_mesh():
            return
        base = os.path.splitext(self.doc.mesh_path or "projekt")[0] + ".s2c"
        path, _ = QFileDialog.getSaveFileName(self, tr("Projekt speichern", "Save project"), base,
                                              tr("Scan2CAD-Projekt", "Scan2CAD project") + " (*.s2c)")
        if not path:
            return
        try:
            save_project(path, self.model, self.features, self.sel,
                         {"mesh_path": self.doc.mesh_path, "T_total": self.T_total.tolist(), "tol": self.doc.tol},
                         self.doc.symmetries, self.doc.constraints)
            self._msg(tr(f"Projekt gespeichert: {path}", f"Project saved: {path}"))
        except Exception as e:  # noqa: BLE001
            self._error(f"{e}\n\n{traceback.format_exc()}")

    def load_project(self, path=None):
        if not path:
            path, _ = QFileDialog.getOpenFileName(self, tr("Projekt öffnen", "Open project"),
                                                  self.settings.value("last_dir", ""),
                                                  tr("Scan2CAD-Projekt", "Scan2CAD project") + " (*.s2c)")
        if not path:
            return

        def done(r):
            if self.step_preview_on:
                self.act_step_prev.setChecked(False)
            meta = r["meta"]
            self.doc = Document(r["model"], r["features"], r["symmetries"], r["constraints"],
                                np.asarray(meta.get("T_total", np.eye(4).tolist()), float),
                                meta.get("mesh_path") or path, meta.get("tol") or {})
            self.feature_meshes = {}
            self.viewer.clear_overlays()
            self.sel = r["selection"]
            self._shown_model = None
            self.refresh_all(reset_camera=True)
            self._clear_history()
            self._suggest_params()
            self.lucky_reset()
            self._msg(tr(f"Projekt geladen: {os.path.basename(path)}", f"Project loaded: {os.path.basename(path)}"))
        self._run(tr("Lade Projekt…", "Loading project…"), load_project, done, path)

    def closeEvent(self, ev):
        try:
            self.viewer.close()
        except Exception:
            pass
        super().closeEvent(ev)
