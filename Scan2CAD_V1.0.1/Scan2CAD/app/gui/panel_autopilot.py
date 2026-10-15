"""Rechter Reiter · ✨ Auf gut Glück: alle Schritte automatisch, mit Pausen zum Eingreifen."""
from __future__ import annotations

from qtpy.QtCore import Qt, QTimer
from qtpy.QtGui import QColor
from qtpy.QtWidgets import QCheckBox, QLabel, QListWidget, QListWidgetItem, QPlainTextEdit, QVBoxLayout, QWidget

from core import autopilot as ap
from core.i18n import tr

from .common import button, hint, row


class AutopilotPanel:
    def _panel_autopilot(self):
        w = QWidget()
        v = QVBoxLayout(w)
        v.addWidget(hint(tr("Führt den ganzen Weg vom Scan zum geschlossenen STEP aus. Nach jedem Schritt wird das "
                            "Zwischenergebnis angezeigt; mit Pause kann man in Ruhe schauen und eingreifen (Features "
                            "löschen, Füllungen ändern, Bohrungen korrigieren …) – der nächste Schritt arbeitet mit dem "
                            "aktuellen Stand. Jeder Schritt ist mit Strg+Z rückgängig zu machen.",
                            "Runs the whole way from scan to closed STEP. After each step the intermediate result is "
                            "shown; with pause you can look and intervene (delete features, change fills, correct holes …) "
                            "– the next step works on the current state. Every step can be undone with Ctrl+Z.")))
        self.lucky_list = QListWidget()
        for key, (de, en) in ap.STEPS:
            it = QListWidgetItem(tr(de, en))
            it.setFlags(it.flags() | Qt.ItemIsUserCheckable)
            it.setCheckState(Qt.Checked)
            it.setData(Qt.UserRole, key)
            self.lucky_list.addItem(it)
        v.addWidget(self.lucky_list)
        self.lucky_pause = QCheckBox(tr("nach jedem Schritt anhalten", "pause after each step"))
        self.lucky_pause.setChecked(True)
        v.addWidget(self.lucky_pause)
        self.btn_lucky_go = button(tr("▶ Start", "▶ Start"), self.lucky_continue, bold=True)
        v.addLayout(row(self.btn_lucky_go, button(tr("⏭ Ein Schritt", "⏭ One step"), self.lucky_one),
                        button(tr("■ Stopp", "■ Stop"), self.lucky_stop)))
        v.addLayout(row(button(tr("Von vorn", "Restart"), self.lucky_reset)))
        self.lucky_state = QLabel("")
        self.lucky_state.setWordWrap(True)
        v.addWidget(self.lucky_state)
        self.lucky_log = QPlainTextEdit()
        self.lucky_log.setReadOnly(True)
        v.addWidget(self.lucky_log, 1)
        self._lucky_i = 0
        self._lucky_running = False
        return w

    # ------------------------------------------------------------------ Steuerung
    def lucky_reset(self):
        if not hasattr(self, "lucky_list"):
            return
        self._lucky_i = 0
        self._lucky_running = False
        for i in range(self.lucky_list.count()):
            it = self.lucky_list.item(i)
            it.setBackground(QColor(0, 0, 0, 0))
            it.setText(it.text().lstrip("✔▶✘ ").strip())
        self.lucky_log.clear()
        self.btn_lucky_go.setText(tr("▶ Start", "▶ Start"))
        self.lucky_state.setText("")

    def lucky_start(self):
        self.right_tabs.setCurrentIndex(2)
        if not self._need_mesh():
            return
        if self._lucky_i >= self.lucky_list.count():
            self.lucky_reset()
        self.lucky_continue()

    def lucky_continue(self):
        if not self._need_mesh():
            return
        self._lucky_running = True
        self._lucky_next()

    def lucky_one(self):
        if not self._need_mesh():
            return
        self._lucky_running = False
        self._lucky_next(force_single=True)

    def lucky_stop(self):
        self._lucky_running = False
        self.lucky_state.setText(tr("angehalten", "stopped"))

    def lucky_on_error(self):
        self._lucky_running = False

    def _lucky_next(self, force_single=False):
        if self.runner.busy:
            return
        # nächsten angehakten Schritt suchen
        while self._lucky_i < self.lucky_list.count() and \
                self.lucky_list.item(self._lucky_i).checkState() != Qt.Checked:
            self._lucky_i += 1
        if self._lucky_i >= self.lucky_list.count():
            self._lucky_running = False
            self.lucky_state.setText("<b>" + tr("Fertig.", "Done.") + "</b> " +
                                     tr("Ergebnis: Export-Seite / STEP-Vorschau (P).", "Result: export page / STEP preview (P)."))
            self.btn_lucky_go.setText(tr("▶ Start", "▶ Start"))
            return
        i = self._lucky_i
        it = self.lucky_list.item(i)
        key = it.data(Qt.UserRole)
        it.setBackground(QColor(255, 210, 74, 60))
        self.lucky_list.scrollToItem(it)
        title = ap.step_title(key)
        self.lucky_state.setText(tr(f"läuft: {title}", f"running: {title}"))
        if self.step_preview_on and key != "preview":
            self.act_step_prev.setChecked(False)

        def fn(d, rep):
            return ap.run_step(key, d, progress=rep)

        def after():
            it.setBackground(QColor(61, 214, 140, 50))
            it.setText("✔ " + it.text().lstrip("✔▶✘ ").strip())
            self._lucky_i = i + 1
            if self._lucky_running and not force_single and not self.lucky_pause.isChecked():
                QTimer.singleShot(50, self._lucky_next)
            else:
                nxt = self._next_title()
                self.lucky_state.setText(tr(f"Pause – nächster Schritt: {nxt}. Ansehen, ggf. eingreifen, dann „Weiter“.",
                                            f"Paused – next step: {nxt}. Look, intervene if needed, then “Continue”.")
                                         if nxt else tr("Fertig.", "Done."))
                self.btn_lucky_go.setText(tr("▶ Weiter", "▶ Continue"))

        def done_msg(lines):
            self.lucky_log.appendPlainText(f"[{title}]")
            for l in lines:
                self.lucky_log.appendPlainText("   " + l)

        ok = self.run_doc(tr("Auf gut Glück: ", "Feeling lucky: ") + title, fn, done_msg=done_msg, after=after)
        if not ok:
            it.setBackground(QColor(255, 99, 105, 60))
            self._lucky_running = False

    def _next_title(self):
        j = self._lucky_i
        while j < self.lucky_list.count():
            if self.lucky_list.item(j).checkState() == Qt.Checked:
                return ap.step_title(self.lucky_list.item(j).data(Qt.UserRole))
            j += 1
        return None
