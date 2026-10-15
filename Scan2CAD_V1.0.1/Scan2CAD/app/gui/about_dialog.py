"""Über Scan2CAD / Zitieren."""
from __future__ import annotations

import os

from qtpy.QtCore import Qt
from qtpy.QtGui import QFont, QIcon
from qtpy.QtWidgets import (QApplication, QDialog, QHBoxLayout, QLabel, QPlainTextEdit, QPushButton,
                            QVBoxLayout)

from core import about
from core.i18n import tr

LOGO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "assets", "vogel-orange.svg")
ORANGE = "#ff5609"


def logo_icon():
    return QIcon(LOGO) if os.path.exists(LOGO) else QIcon()


class AboutDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle(tr("Über Scan2CAD / Zitieren", "About Scan2CAD / Cite"))
        self.setWindowIcon(logo_icon())
        self.setMinimumWidth(660)
        v = QVBoxLayout(self)
        v.setSpacing(10)

        head = QHBoxLayout()
        logo = QLabel()
        pm = logo_icon().pixmap(80, 96)
        if not pm.isNull():
            logo.setPixmap(pm)
        logo.setAlignment(Qt.AlignTop)
        head.addWidget(logo)
        txt = QVBoxLayout()
        eyebrow = QLabel(tr("SCAN → STEP MIT KONSTRUKTIONSABSICHT", "SCAN → STEP WITH DESIGN INTENT"))
        eyebrow.setStyleSheet(f"color: {ORANGE}; font-size: 9pt; letter-spacing: 1px;")
        title = QLabel(about.NAME)
        f = QFont(title.font())
        f.setPointSizeF(f.pointSizeF() * 2.0)
        f.setBold(True)
        title.setFont(f)
        line = QLabel(tr(f"Version <b>{about.VERSION}</b> · von <b>{about.AUTHOR}</b>",
                         f"Version <b>{about.VERSION}</b> · by <b>{about.AUTHOR}</b>"))
        lic = QLabel(tr("Open Source · Apache License 2.0", "Open source · Apache License 2.0"))
        for w in (eyebrow, title, line, lic):
            txt.addWidget(w)
        head.addLayout(txt, 1)
        v.addLayout(head)

        tag = QLabel(f"<i>{about.tagline()}</i>")
        tag.setWordWrap(True)
        v.addWidget(tag)
        ai = QLabel(about.ai_disclosure())
        ai.setWordWrap(True)
        v.addWidget(ai)
        links = QLabel(f"<a href='{about.REPOSITORY}'>" + tr("Repository &amp; Downloads", "Repository &amp; downloads")
                       + f"</a> &nbsp;·&nbsp; <a href='{about.REPOSITORY}/issues'>"
                       + tr("Fehler melden", "Report an issue") + "</a> &nbsp;·&nbsp; "
                       + "<a href='https://www.apache.org/licenses/LICENSE-2.0'>Apache License 2.0</a>")
        links.setOpenExternalLinks(True)
        links.setTextInteractionFlags(Qt.TextBrowserInteraction)
        v.addWidget(links)

        h = QLabel(tr("Zitieren", "Cite this software"))
        hf = QFont(h.font())
        hf.setBold(True)
        hf.setPointSizeF(hf.pointSizeF() * 1.15)
        h.setFont(hf)
        v.addWidget(h)
        note = QLabel(tr("Zitieren ist freiwillig – freut aber den Autor und erklärt Gutachtern, woher das "
                         "STEP kommt.",
                         "Citation is optional – but it makes the author happy and tells reviewers where the "
                         "STEP came from."))
        note.setWordWrap(True)
        v.addWidget(note)
        self.cite_label = QLabel(tr("Zitat", "Citation"))
        self.cite_label.setStyleSheet("color: #9aa4ae;")
        v.addWidget(self.cite_label)
        self.cite_box = QPlainTextEdit(about.citation())
        self.cite_box.setReadOnly(True)
        mono = QFont("Consolas")
        mono.setStyleHint(QFont.Monospace)
        self.cite_box.setFont(mono)
        self.cite_box.setMinimumHeight(150)
        v.addWidget(self.cite_box, 1)
        row = QHBoxLayout()
        self.btn_cite = QPushButton(tr("Zitat kopieren", "Copy citation"))
        self.btn_cite.clicked.connect(lambda: self.copy("text"))
        self.btn_bib = QPushButton(tr("BibTeX kopieren", "Copy BibTeX"))
        self.btn_bib.clicked.connect(lambda: self.copy("bibtex"))
        self.status = QLabel("")
        self.status.setStyleSheet("color: #3dd68c;")
        row.addWidget(self.btn_cite)
        row.addWidget(self.btn_bib)
        row.addWidget(self.status, 1)
        v.addLayout(row)

        foot = QLabel(tr("Copyright © 2026 Amon P. Lanz. Bereitgestellt „AS IS“ ohne Gewährleistung "
                         "(Apache License 2.0). Maße vor der Fertigung selbst prüfen – auch Scanner haben "
                         "schlechte Tage. LICENSE, NOTICE, CITATION.cff und CITATION.bib liegen bei.",
                         "Copyright © 2026 Amon P. Lanz. Provided “AS IS”, without warranties (Apache "
                         "License 2.0). Check dimensions before manufacturing – scanners have bad days too. "
                         "LICENSE, NOTICE, CITATION.cff and CITATION.bib are included."))
        foot.setWordWrap(True)
        foot.setStyleSheet("color: #9aa4ae; font-size: 8.5pt;")
        v.addWidget(foot)
        close = QPushButton(tr("Schließen", "Close"))
        close.clicked.connect(self.accept)
        close.setDefault(True)
        r2 = QHBoxLayout()
        r2.addStretch(1)
        r2.addWidget(close)
        v.addLayout(r2)

    def copy(self, which):
        bib = which == "bibtex"
        text = about.bibtex() if bib else about.citation()
        self.cite_box.setPlainText(text)
        self.cite_label.setText(tr("BibTeX-Eintrag", "BibTeX entry") if bib else tr("Zitat", "Citation"))
        try:
            QApplication.clipboard().setText(text)
            self.status.setText(tr("BibTeX kopiert.", "BibTeX copied.") if bib else tr("Zitat kopiert.", "Citation copied."))
        except Exception:  # noqa: BLE001
            self.cite_box.selectAll()
            self.status.setText(tr("Bitte mit Strg+C kopieren.", "Please copy with Ctrl+C."))
