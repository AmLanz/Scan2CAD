"""Zweisprachige Oberfläche (Deutsch / English).

Texte stehen direkt am Verwendungsort als Paar: tr("Deutsch", "English").
Die Sprache wird beim Start gesetzt (QSettings bzw. --lang=en) – ein Wechsel braucht einen
Neustart, weil alle Widgets ihre Beschriftung beim Aufbau erhalten.
"""
from __future__ import annotations

_STATE = {"lang": "de"}


def set_lang(lang):
    _STATE["lang"] = "en" if str(lang).lower().startswith("en") else "de"


def lang():
    return _STATE["lang"]


def tr(de, en=None):
    if _STATE["lang"] == "en" and en is not None:
        return en
    return de


class LangDict(dict):
    """dict key -> (de, en); Zugriff liefert den Text der aktuellen Sprache."""

    def __getitem__(self, k):
        v = dict.__getitem__(self, k)
        return tr(*v) if isinstance(v, tuple) else v

    def get(self, k, default=None):
        if k in self:
            return self[k]
        return default

    def items(self):
        return [(k, self[k]) for k in self.keys()]

    def values(self):
        return [self[k] for k in self.keys()]
