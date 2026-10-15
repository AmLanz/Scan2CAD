"""Nicht zugeordnete Netzbereiche automatisch in Freiform-Flächen übersetzen.

Zerlegung: Zusammenhangskomponenten der freien Dreiecke werden so lange geteilt, bis jedes
Stück 'höhenfeldartig' ist (alle Normalen innerhalb eines Kegels um die mittlere Normale) und
einen Rand hat. Dann ist die Projektion auf die Ausgleichsebene eindeutig und ein getrimmter
B-Spline-Patch (Projektionsmodus) beschreibt das Stück stabil.
"""
from __future__ import annotations

import numpy as np

from .geom_utils import unit


def _split(model, ids, rng):
    """Teilt eine Region in zwei: k-Means (k=2) auf den Normalen, sonst entlang der
    längsten Ausdehnung. Rückgabe: Liste zusammenhängender Teilstücke."""
    N = model.fit_normals[ids]
    C = model.centroids[ids]
    A = model.areas[ids]
    # k-Means auf Normalen
    c0 = N[rng.integers(len(N))]
    c1 = N[np.argmin(N @ c0)]
    lab = None
    for _ in range(12):
        d0, d1 = N @ c0, N @ c1
        lab = d1 > d0
        if lab.all() or (~lab).all():
            break
        c0 = unit((N[~lab] * A[~lab, None]).sum(0))
        c1 = unit((N[lab] * A[lab, None]).sum(0))
    if lab is None or lab.all() or (~lab).all() or min(lab.sum(), (~lab).sum()) < 0.05 * len(ids):
        cen = (C * A[:, None]).sum(0) / A.sum()
        X = C - cen
        ev, evec = np.linalg.eigh((X * A[:, None]).T @ X)
        lab = X @ evec[:, -1] > 0
    parts = []
    for m in (lab, ~lab):
        parts.extend(model.components_of(ids[m]))
    return parts


def segment_heightfield_patches(model, face_ids, max_angle_deg=65.0, min_faces=40, max_depth=12, seed=0):
    rng = np.random.default_rng(seed)
    cos_lim = np.cos(np.radians(max_angle_deg))
    stack = [(c, 0) for c in model.components_of(np.asarray(face_ids, np.int64))]
    patches, leftovers = [], []
    while stack:
        ids, depth = stack.pop()
        if len(ids) < min_faces:
            leftovers.append(ids)
            continue
        N = model.fit_normals[ids]
        A = model.areas[ids]
        nm = unit((N * A[:, None]).sum(0))
        ok = np.percentile(N @ nm, 2) > cos_lim
        has_rim = len(model.boundary_loops(ids)) > 0
        if ok and has_rim:
            patches.append(ids)
        elif depth >= max_depth:
            leftovers.append(ids)
        else:
            for part in _split(model, ids, rng):
                stack.append((part, depth + 1))
    patches.sort(key=lambda p: -len(p))
    return patches, leftovers


def auto_freeform(model, face_ids, spacing=3.0, smooth=0.4, tol=0.05, max_angle_deg=65.0,
                  min_faces=40, progress=None):
    """Erzeugt Freiform-Features für freie Bereiche. Rückgabe: (features, nicht abgedeckte Dreiecke)."""
    from .features import Feature, _stats
    from .freeform import fit_freeform_patch
    patches, left = segment_heightfield_patches(model, face_ids, max_angle_deg, min_faces)
    feats = []
    rest = list(left)
    for k, ids in enumerate(patches):
        try:
            V, F, _ = model.submesh(ids)
            ext = float(np.linalg.norm(V.max(0) - V.min(0)))
            n_ctrl = int(np.clip(round(ext / max(spacing, 1e-3)), 5, 40))
            r = fit_freeform_patch(V, F, n_ctrl, smooth, "planar", trim_tol=tol)
            params = {"poles": r["poles"], "knots_u": r["knots_u"], "knots_v": r["knots_v"],
                      "degree": 3, "mode": r["mode"],
                      "trim_uv": None if r["trim_uv"] is None else np.asarray(r["trim_uv"]).tolist()}
            f = Feature("freeform", params, "surface", ids, stats=_stats(r["residuals"], r["weights"]),
                        auto=True)
            f.shape()  # Gültigkeit prüfen
            feats.append(f)
        except Exception:
            rest.append(ids)
        if progress:
            progress(k + 1, len(patches))
    rest = np.concatenate(rest) if rest else np.zeros(0, np.int64)
    return feats, rest
