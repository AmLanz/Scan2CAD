"""Dokument: alles, was zu einem Projekt gehört (Netz, Features, Symmetrien, Bedingungen, Lage).

Rechenschritte (GUI-Aktionen und 'Auf gut Glück') arbeiten auf einer KOPIE (clone) und liefern
ein neues Dokument zurück; die GUI übernimmt es im Haupt-Thread (mit Rückgängig-Punkt). So
bleibt der angezeigte Zustand während einer Hintergrundrechnung unverändert.
"""
from __future__ import annotations

import copy

import numpy as np

from .features import Feature


def clone_feature(f):
    g = Feature.from_dict(f.to_dict(), np.asarray(f.face_ids, np.int64).copy())
    g.stats = dict(f.stats)
    return g


class Document:
    def __init__(self, model=None, features=None, symmetries=None, constraints=None, T_total=None,
                 mesh_path=None, tol=None):
        self.model = model
        self.features = list(features or [])
        self.symmetries = list(symmetries or [])
        self.constraints = list(constraints or [])
        self.T_total = np.eye(4) if T_total is None else np.asarray(T_total, float)
        self.mesh_path = mesh_path
        self.tol = dict(tol or {})
        self.preview = None          # Ergebnis der letzten Scan-Topologie (für die Vorschau)

    # ------------------------------------------------------------------ Kopie
    def clone(self):
        d = Document(self.model, [clone_feature(f) for f in self.features],
                     copy.deepcopy(self.symmetries), copy.deepcopy(self.constraints), self.T_total.copy(),
                     self.mesh_path, dict(self.tol))
        return d

    # ------------------------------------------------------------------ Werte
    @property
    def fit_tol(self):
        return float(self.tol.get("fit_tol", 0.05))

    @property
    def detect_eps(self):
        return float(self.tol.get("detect_eps", 0.1))

    def feature(self, fid):
        return next((f for f in self.features if f.id == fid), None)

    def assigned(self):
        n = self.model.n_faces if self.model is not None else 0
        a = np.zeros(n, np.int64)
        for f in self.features:
            ids = np.asarray(f.face_ids, np.int64)
            a[ids[(ids >= 0) & (ids < n)]] = f.id
        return a

    # ------------------------------------------------------------------ Netz ersetzen
    def replace_model(self, new_model, face_map=None, max_dist=None):
        """Neues Netz übernehmen; Flächenzuordnungen der Features mitnehmen
        (face_map alt->neu, -1 = entfernt; sonst über nächste Schwerpunkte)."""
        old = self.model
        self.model = new_model
        if old is None:
            return
        if face_map is not None:
            fm = np.asarray(face_map, np.int64)
            for f in self.features:
                ids = np.asarray(f.face_ids, np.int64)
                ids = ids[(ids >= 0) & (ids < len(fm))]
                nid = fm[ids]
                f.face_ids = nid[nid >= 0]
            return
        if new_model.n_faces >= old.n_faces and np.array_equal(new_model.F[:old.n_faces], old.F):
            return               # nur angehängt (Löcher gefüllt): Indizes bleiben gültig
        from .mesh_ops import transfer_face_labels
        owner = self.assigned_on(old)
        md = max_dist if max_dist is not None else 2.0 * old.mean_edge_length()
        new_owner = transfer_face_labels(old, new_model, owner, max_dist=md)
        for f in self.features:
            f.face_ids = np.nonzero(new_owner == f.id)[0]

    def assigned_on(self, model):
        a = np.zeros(model.n_faces, np.int64)
        for f in self.features:
            ids = np.asarray(f.face_ids, np.int64)
            a[ids[(ids >= 0) & (ids < model.n_faces)]] = f.id
        return a

    # ------------------------------------------------------------------ Lage
    def apply_transform(self, T):
        T = np.asarray(T, float)
        self.model = self.model.transformed(T)
        self.T_total = T @ self.T_total
        for f in self.features:
            f.transform(T)
        R, t = T[:3, :3], T[:3, 3]
        for s in self.symmetries:
            for k in ("o", "p"):
                if k in s and s[k] is not None:
                    s[k] = R @ np.asarray(s[k]) + t
            for k in ("nrm", "d"):
                if k in s and s[k] is not None:
                    s[k] = R @ np.asarray(s[k])
        self.preview = None
