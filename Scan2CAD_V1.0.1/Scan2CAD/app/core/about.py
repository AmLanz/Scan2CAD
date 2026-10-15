"""Version, Autor, Lizenz und Zitierhinweis (eine Stelle für Fenster, Dialog und CITATION-Dateien)."""
from __future__ import annotations

from .i18n import tr

NAME = "Scan2CAD"
VERSION = "1.0.1"
YEAR = 2026
AUTHOR = "Amon P. Lanz"
AUTHOR_BIB = "Lanz, Amon P."
LICENSE = "Apache-2.0"
REPOSITORY = "https://github.com/AmLanz/Scan2CAD"
TITLE = "Scan2CAD: Guided Reverse Engineering of 3D Scans to STEP"
AI_NOTE = "Developed with Claude Opus 5.5 (Anthropic)"


def citation():
    return (f"{AUTHOR} ({YEAR}). {TITLE} (version {VERSION}). Computer software. {LICENSE}. "
            f"{REPOSITORY}. {AI_NOTE}.")


def bibtex():
    return (f"@software{{lanz_scan2cad_{YEAR},\n"
            f"  author  = {{{AUTHOR_BIB}}},\n"
            f"  title   = {{{TITLE}}},\n"
            f"  version = {{{VERSION}}},\n"
            f"  year    = {{{YEAR}}},\n"
            f"  license = {{{LICENSE}}},\n"
            f"  url     = {{{REPOSITORY}}},\n"
            f"  note    = {{{AI_NOTE}. No water blasters were harmed during testing.}}\n"
            f"}}")


def tagline():
    return tr("Macht aus verrauschten Dreiecken Flächen mit Konstruktionsabsicht – meistens.",
              "Turns noisy triangles into surfaces with design intent – most of the time.")


def ai_disclosure():
    return tr("Offen gesagt: Ein großer Teil des Codes wurde von <b>Claude Opus 5.5</b> (Anthropic) geschrieben. "
              "Idee, Scans, Testteile, Kritik, die Wasserpistolen-Feldversuche und jede Entscheidung, was gut "
              "genug ist: Amon P. Lanz.",
              "In the spirit of openness: much of the code was written by <b>Claude Opus 5.5</b> (Anthropic). "
              "Idea, scans, test parts, critique, the water-blaster field trials and every call on what is good "
              "enough: Amon P. Lanz.")
