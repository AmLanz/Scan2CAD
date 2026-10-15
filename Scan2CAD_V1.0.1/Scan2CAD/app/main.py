"""Scan2CAD – Startpunkt."""
import os
import sys
import traceback
import warnings

os.environ.setdefault("QT_API", "pyside6")
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
LOG = os.path.join(os.path.dirname(HERE), "scan2cad.log")


def _excepthook(etype, value, tb):
    msg = "".join(traceback.format_exception(etype, value, tb))
    try:
        with open(LOG, "a", encoding="utf-8") as fh:
            fh.write(msg + "\n")
    except Exception:
        pass
    sys.__stderr__.write(msg)
    try:
        from qtpy.QtWidgets import QApplication, QMessageBox
        if QApplication.instance() is not None:
            from core.i18n import tr
            box = QMessageBox(QMessageBox.Critical, tr("Unerwarteter Fehler", "Unexpected error"), str(value))
            box.setDetailedText(msg)
            box.exec()
    except Exception:
        pass


def _choose_lang():
    from core.i18n import set_lang
    code = None
    for a in sys.argv[1:]:
        if a.startswith("--lang="):
            code = a.split("=", 1)[1].strip().lower()
    if code is None:
        try:
            from qtpy.QtCore import QSettings
            code = str(QSettings("Scan2CAD", "Scan2CAD").value("lang", "de") or "de")
        except Exception:
            code = "de"
    set_lang("en" if code.startswith("en") else "de")
    try:
        from qtpy.QtCore import QSettings
        QSettings("Scan2CAD", "Scan2CAD").setValue("lang", "en" if code.startswith("en") else "de")
    except Exception:
        pass


def main():
    sys.excepthook = _excepthook
    from qtpy.QtWidgets import QApplication
    import pyvista as pv
    pv.global_theme.allow_empty_mesh = True
    app = QApplication.instance() or QApplication(sys.argv)
    _choose_lang()
    from gui.main_window import MainWindow

    app.setApplicationName("Scan2CAD")
    app.setStyle("Fusion")
    w = MainWindow()
    w.show()
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if args and os.path.isfile(args[0]):
        if args[0].lower().endswith(".s2c"):
            w.load_project(args[0])
        else:
            w.open_mesh(args[0])
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
