"""Automatische Primitiverkennung (vereinfachtes 'Efficient RANSAC' nach Schnabel et al. 2007).

Arbeitet auf Dreiecksschwerpunkten + Normalen, mit lokalitätsbasierter Stichprobe,
Bewertung auf einer Teilmenge, Zusammenhangskomponente im Netz und
abschließendem Least-Squares-Refit.
"""
from __future__ import annotations

import numpy as np

from .fitting import (FIT_FUNCS, PRIMITIVE_ORDER, auto_fit, is_plausible, model_distance,
                      model_normals)
from .geom_utils import unit

COMPLEXITY = {"plane": 0, "sphere": 1, "cylinder": 1, "cone": 2, "torus": 3}


def _closest_between_lines(p1, d1, p2, d2):
    w0 = p1 - p2
    a, b, c = d1 @ d1, d1 @ d2, d2 @ d2
    d, e = d1 @ w0, d2 @ w0
    den = a * c - b * b
    if abs(den) < 1e-10:
        return None
    s = (b * e - c * d) / den
    t = (a * e - b * d) / den
    return 0.5 * ((p1 + s * d1) + (p2 + t * d2))


def _cand_plane(P, N):
    return {"origin": P[0].copy(), "normal": N[0].copy()}


def _cand_sphere(P, N):
    c = _closest_between_lines(P[0], N[0], P[1], N[1])
    if c is None:
        return None
    r = 0.5 * (np.linalg.norm(P[0] - c) + np.linalg.norm(P[1] - c))
    return {"center": c, "radius": float(r)}


def _cand_cylinder(P, N):
    a = np.cross(N[0], N[1])
    na = np.linalg.norm(a)
    if na < 0.05:
        return None
    a /= na
    p1 = P[0] - (P[0] @ a) * a
    p2 = P[1] - (P[1] @ a) * a
    c = _closest_between_lines(p1, N[0], p2, N[1])
    if c is None:
        return None
    r = 0.5 * (np.linalg.norm(p1 - c) + np.linalg.norm(p2 - c))
    return {"point": c, "axis": a, "radius": float(r)}


def _cand_cone(P, N):
    A = N[:3]
    if abs(np.linalg.det(A)) < 1e-3:
        return None
    apex = np.linalg.solve(A, np.einsum("ij,ij->i", N[:3], P[:3]))
    a = np.cross(N[1] - N[0], N[2] - N[0])
    if np.linalg.norm(a) < 1e-6:
        return None
    a = unit(a)
    if np.mean((P[:3] - apex) @ a) < 0:
        a = -a
    q = P[:3] - apex
    h = q @ a
    rho = np.linalg.norm(q - np.outer(h, a), axis=1)
    al = float(np.mean(np.arctan2(rho, np.maximum(h, 1e-9))))
    if not (0.02 < al < np.pi / 2 - 0.02):
        return None
    return {"apex": apex, "axis": a, "half_angle": al}


MINIMAL = {"plane": (1, _cand_plane), "sphere": (2, _cand_sphere),
           "cylinder": (2, _cand_cylinder), "cone": (3, _cand_cone)}


def _inliers(kind, params, P, N, eps, cos_a, T=None):
    """Dreiecke, die zum Modell passen: Abstand (Schwerpunkt bzw. bei CAD-Exporten alle drei
    Eckpunkte) < eps und Normale innerhalb des Winkels."""
    if T is None:
        d = np.abs(model_distance(kind, params, P))
    else:
        d = np.abs(model_distance(kind, params, T.reshape(-1, 3))).reshape(-1, 3).max(1)
    ok = d < eps
    if ok.any():
        g = model_normals(kind, params, P[ok])
        c = np.abs(np.einsum("ij,ij->i", g, N[ok]))
        ok[np.nonzero(ok)[0][c < cos_a]] = False
    return ok


def detect_primitives(model, eps=0.1, angle_deg=15.0, min_area_frac=0.005,
                      kinds=("plane", "cylinder", "sphere", "cone", "torus"),
                      max_shapes=40, trials=120, exclude_mask=None, seed=0,
                      progress=None, eps_coarse=None):
    """Erkennt Primitive auf dem Netz. Rückgabe: Liste (kind, FitResult, face_ids)."""
    rng = np.random.default_rng(seed)
    C, Nf, A = model.centroids, model.fit_normals, model.weights
    T = model.V[model.F] if model.exact else None      # CAD: Eckpunkte liegen exakt auf der Fläche
    # Kandidaten aus Schwerpunkten gekrümmter Facetten liegen um den Facettenfehler daneben:
    # Bewertung und erste Verfeinerung mit grober, danach mit enger Toleranz (exakte Eckpunkte)
    eps_c = max(eps, eps_coarse or eps)
    nF = len(C)
    remaining = np.ones(nF, bool)
    if exclude_mask is not None:
        remaining &= ~exclude_mask
    total_area = A[remaining].sum()
    min_area = max(min_area_frac * total_area, 1e-9)
    cos_a = np.cos(np.radians(angle_deg))
    tree = model.centroid_tree
    L = model.diag
    neigh_r = 0.06 * L
    results = []
    fails = 0
    kinds = [k for k in kinds if k in MINIMAL or k == "torus"]
    seedable = ~model.synthetic
    while len(results) < max_shapes and fails < 3:
        rem_idx = np.nonzero(remaining)[0]
        seed_idx = np.nonzero(remaining & seedable)[0]
        if len(seed_idx) < 10:
            break
        if A[rem_idx].sum() < min_area or len(rem_idx) < 10:
            break
        sub = rem_idx if len(rem_idx) <= 4000 else rng.choice(rem_idx, 4000, replace=False)
        cands_all = []
        for t in range(trials):
            s0 = rng.choice(seed_idx)
            nb = tree.query_ball_point(C[s0], neigh_r)
            nb = np.asarray(nb)
            nb = nb[remaining[nb]]
            if len(nb) < 4:
                continue
            for k in kinds:
                if k == "torus":
                    if t % 20 != 0 or len(nb) < 30:
                        continue
                    pick = nb if len(nb) <= 200 else rng.choice(nb, 200, replace=False)
                    try:
                        # nur wenn Ebene/Zylinder lokal nicht schon passen
                        pl = FIT_FUNCS["plane"](C[pick], Nf[pick], A[pick], None)
                        if pl.rms < 0.5 * eps:
                            continue
                        cy = FIT_FUNCS["cylinder"](C[pick], Nf[pick], A[pick], None)
                        if cy.rms < 0.5 * eps:
                            continue
                        fr = FIT_FUNCS["torus"](C[pick], Nf[pick], A[pick], None)
                        cand = {kk: v for kk, v in fr.params.items() if kk != "inside"}
                        if fr.rms > 0.5 * eps or cand["R"] > 5 * L or cand["r"] > 0.5 * L:
                            continue
                    except Exception:
                        continue
                else:
                    m, fn = MINIMAL[k]
                    pick = np.r_[s0, rng.choice(nb, m - 1, replace=False)] if m > 1 else np.array([s0])
                    cand = fn(C[pick], Nf[pick])
                    if cand is None:
                        continue
                if k in ("sphere", "cylinder") and cand["radius"] > 20 * L:
                    continue
                ok = _inliers(k, cand, C[sub], Nf[sub], eps_c, cos_a, None if T is None else T[sub])
                score = A[sub][ok].sum() * (1.0 - 0.03 * COMPLEXITY[k])
                if score > 0:
                    cands_all.append((score, k, cand))
        if not cands_all:
            fails += 1
            continue
        cands_all.sort(key=lambda c: -c[0])
        accepted = None
        for _, kind, cand in cands_all[:6]:
            accepted = _refine(model, kind, cand, rem_idx, C, Nf, A, eps, cos_a, nF, kinds, T, eps_c)
            if accepted is not None and A[accepted[1]].sum() >= min_area:
                break
            accepted = None
        if accepted is None:
            fails += 1
            continue
        fails = 0
        fr, comp = accepted
        remaining[comp] = False
        results.append((fr.kind, fr, comp))
        if progress:
            progress(len(results), fr.kind)
    return results


def _refine(model, kind, cand, rem_idx, C, Nf, A, eps, cos_a, nF, kinds=None, T=None, eps_c=None):
    """Volle Inlier-Menge, größte Zusammenhangskomponente, LSQ-Refit (3 Runden).
    In der ersten Runde wird per Modellauswahl ggf. auf einen besser passenden Typ gewechselt."""
    accepted = None
    params = cand
    if True:
        for it in range(3):
            ok = np.zeros(nF, bool)
            ok[rem_idx] = _inliers(kind, params, C[rem_idx], Nf[rem_idx], (eps_c or eps) if it == 0 else eps,
                                   cos_a, None if T is None else T[rem_idx])
            ids = np.nonzero(ok)[0]
            if len(ids) < 8:
                break
            comps = model.components_of(ids)
            comp = comps[0]
            fc = comp
            if len(fc) > 3000:
                fc = np.random.default_rng(it).choice(comp, 3000, replace=False)
            Pf, Nff, Af = model.fit_samples(fc) if T is not None else (C[fc], Nf[fc], A[fc])
            try:
                if it == 0 and kinds:
                    cheap = [k for k in PRIMITIVE_ORDER if k in kinds and k != "torus"]
                    fr, _ = auto_fit(Pf, Nff, Af, tol=eps / 3.0, kinds=cheap, robust=eps)
                    if "torus" in kinds and fr.rms > eps / 3.0:
                        fr, _ = auto_fit(Pf, Nff, Af, tol=eps / 3.0, kinds=cheap + ["torus"], robust=eps)
                    kind = fr.kind
                else:
                    fr = FIT_FUNCS[kind](Pf, Nff, Af, eps)
            except Exception:
                break
            if not is_plausible(fr, Pf):
                break
            params = {kk: v for kk, v in fr.params.items() if kk != "inside"}
            accepted = (fr, comp)
    return accepted if accepted is None else (accepted[0], accepted[1])
