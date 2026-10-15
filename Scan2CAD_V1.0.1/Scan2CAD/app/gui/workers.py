"""Hintergrund-Ausführung langer Rechnungen (GUI bleibt bedienbar)."""
from __future__ import annotations

import threading
import traceback

from qtpy.QtCore import QObject, QRunnable, QThreadPool, Signal

# VTK (pyvista) ist nicht threadsicher: rechnet ein Hintergrund-Job mit VTK, während der Haupt-
# Thread rendert, kann es hängen bleiben. Jobs halten deshalb diese Sperre; der Viewer rendert
# nur, wenn er sie bekommt (sonst wird das Bild nach dem Job nachgeholt).
VTK_LOCK = threading.RLock()


class _Signals(QObject):
    finished = Signal(object)
    failed = Signal(str)
    progress = Signal(str)


class _Job(QRunnable):
    def __init__(self, fn, args, kwargs, signals):
        super().__init__()
        self.fn, self.args, self.kwargs, self.signals = fn, args, kwargs, signals
        self.setAutoDelete(True)

    def run(self):
        err = None
        with VTK_LOCK:
            try:
                res = self.fn(*self.args, **self.kwargs)
            except Exception as e:  # noqa: BLE001
                err = f"{e}\n\n{traceback.format_exc()}"
        if err is not None:
            self.signals.failed.emit(err)
            return
        self.signals.finished.emit(res)


class TaskRunner(QObject):
    """Startet genau eine Aufgabe zur Zeit; meldet Start/Ende an die GUI."""
    busy_changed = Signal(bool, str)
    progress = Signal(str)

    def report(self, text):
        """Aus dem Arbeits-Thread aufrufbar (Signal wird in den GUI-Thread übertragen)."""
        self.progress.emit(str(text))

    def __init__(self, parent=None):
        super().__init__(parent)
        self.pool = QThreadPool.globalInstance()
        self._active = None

    @property
    def busy(self):
        return self._active is not None

    def run(self, label, fn, on_done, *args, on_error=None, **kwargs):
        if self._active is not None:
            return False
        sig = _Signals()
        self._active = sig

        def done(res):
            self._active = None
            self.busy_changed.emit(False, "")
            on_done(res)

        def fail(msg):
            self._active = None
            self.busy_changed.emit(False, "")
            if on_error:
                on_error(msg)

        sig.finished.connect(done)
        sig.failed.connect(fail)
        self.busy_changed.emit(True, label)
        self.pool.start(_Job(fn, args, kwargs, sig))
        return True
