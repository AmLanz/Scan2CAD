"""Symmetrie-Erkennung.

Global (ganzes Bauteil, über die Messpunkte):
  * Spiegelung an einer Ebene
  * Drehung um eine Achse um 360°/n (n = 2…12) und Rotationskörper (∞)
Kandidaten aus der Hauptachsenzerlegung (PCA) und aus Feature-Achsen/Ebenen, Bewertung =
Anteil der Punkte, deren Bild wieder auf dem Scan liegt; Verfeinerung per ICP-artiger
Neuschätzung (Ebene durch die Mittelpunkte der Punktpaare, Achse per Kabsch).

Lokal (Feature-Muster):
  * Lochkreise: ≥3 gleiche, parallele Bohrungen/Zylinder auf einem Kreis mit gleichen Abständen
  * Koaxiale Rotationsflächen -> Achse (∞)
  * Spiegelpaare: gleiche Features, gespiegelt an einer gemeinsamen Ebene
Jede Symmetrie ist ein dict:
  {id, type: 'mirror'|'rot', n (rot; 0 = ∞), o, nrm (mirror), p, d (rot), score, scope, features, name}
"""
from __future__ import annotations

import math

import numpy as np
from scipy.spatial import cKDTree

from .geom_utils import orthonormal_frame, unit
from .i18n import tr

_SID = {"next": 1}


def new_sym_id(existing=()):
    used = {s["id"] for s in existing}
    while _SID["next"] in used:
        _SID["next"] += 1
    i = _SID["next"]
    _SID["next"] += 1
    return i


def sym_label(s):
    if s["type"] == "mirror":
        return tr("Spiegelebene", "Mirror plane")
    n = int(s.get("n", 0))
    if n == 0:
        return tr("Rotationsachse (∞)", "Axis of revolution (∞)")
    return tr(f"{n}-zählige Drehachse ({360 / n:g}°)", f"{n}-fold axis ({360 / n:g}°)")


def sym_summary(s):
    if s["type"] == "mirror":
        g = f"o {np.round(s['o'], 3)}, n {np.round(s['nrm'], 4)}"
    else:
        g = f"p {np.round(s['p'], 3)}, d {np.round(s['d'], 4)}"
    sc = tr("Übereinstimmung", "match") + f" {100 * s.get('score', 0):.0f}%"
    fe = s.get("features") or []
    scope = tr("ganzes Bauteil", "whole part") if s.get("scope") == "global" else \
        tr(f"{len(fe)} Features", f"{len(fe)} features")
    return f"{s['name']}: {sym_label(s)} – {scope}, {sc}\n{g}"


# ===================================================================== Transformationen
def mirror_matrix(o, n):
    n = unit(n)
    R = np.eye(3) - 2 * np.outer(n, n)
    t = 2 * float(np.asarray(o) @ n) * n
    return R, t


def rot_matrix(p, d, ang):
    d = unit(d)
    K = np.array([[0, -d[2], d[1]], [d[2], 0, -d[0]], [-d[1], d[0], 0]])
    R = np.eye(3) + math.sin(ang) * K + (1 - math.cos(ang)) * K @ K
    p = np.asarray(p, float)
    t = p - R @ p
    return R, t


def sym_transforms(s):
    """Liste (R, t) der erzeugenden Abbildungen (für ∞: einige Winkel)."""
    if s["type"] == "mirror":
        return [mirror_matrix(s["o"], s["nrm"])]
    n = int(s.get("n", 0))
    angs = [2 * math.pi * k / n for k in range(1, n)] if n else [math.radians(a) for a in (37, 101, 199, 263)]
    return [rot_matrix(s["p"], s["d"], a) for a in angs]


# ===================================================================== Bewertung
def _sample(model, n=15000, seed=0):
    real = np.nonzero(~model.synthetic)[0]
    ids = real if len(real) else np.arange(model.n_faces)
    rng = np.random.default_rng(seed)
    if len(ids) > n:
        ids = rng.choice(ids, n, replace=False, p=model.areas[ids] / model.areas[ids].sum())
    return model.centroids[ids], model.fit_normals[ids]


def score_transform(P, N, tree, R, t, tau, NT=None, dmax=None):
    """Anteil der abgebildeten Punkte, die wieder auf dem Scan liegen: Punkt-zu-Ebene-Abstand
    zum nächsten Nachbarn < tau (unabhängig von der Punktdichte), Nachbar näher als dmax,
    Normalen passend."""
    Q = P @ R.T + t
    dmax = 4 * tau if dmax is None else dmax
    d, j = tree.query(Q, distance_upper_bound=dmax)
    ok = np.isfinite(d)
    if NT is not None and ok.any():
        jj = j[ok]
        dn = np.abs(np.einsum("ij,ij->i", Q[ok] - tree.data[jj], NT[jj]))
        cn = np.abs(np.einsum("ij,ij->i", N[ok] @ R.T, NT[jj]))
        good = (dn < tau) & (cn > 0.85)
        ok[np.nonzero(ok)[0][~good]] = False
    else:
        ok &= d < tau
    return float(ok.mean()), Q, j, ok


def _good_pairs(P, N, tree, NT, R, t, tau, dmax):
    Q = P @ R.T + t
    d, j = tree.query(Q, distance_upper_bound=dmax)
    ok = np.isfinite(d)
    idx = np.nonzero(ok)[0]
    jj = j[ok]
    dn = np.abs(np.einsum("ij,ij->i", Q[ok] - tree.data[jj], NT[jj]))
    cn = np.abs(np.einsum("ij,ij->i", N[ok] @ R.T, NT[jj]))
    g = (dn < tau) & (cn > 0.85)
    return idx[g], jj[g]


def _icp(P, N, tree, NT, make, theta0, tau, dmax, iters=8):
    """Punkt-zu-Ebene-ICP über wenige Parameter theta: make(theta) -> (R, t).
    Korrespondenzen je Runde neu (nur passende Paare), dann Gauß-Newton-Schritte."""
    th = np.asarray(theta0, float).copy()
    for it in range(iters):
        R, t = make(th)
        i, j = _good_pairs(P, N, tree, NT, R, t, (3.0 if it < 2 else 1.5) * tau, dmax)
        if len(i) < 30:
            break
        X, Y, NY = P[i], tree.data[j], NT[j]

        def res(thv):
            Rr, tt = make(thv)
            return np.einsum("ij,ij->i", X @ Rr.T + tt - Y, NY)
        for _ in range(3):
            r0 = res(th)
            J = np.empty((len(r0), len(th)))
            for k in range(len(th)):
                h = 1e-6 * max(1.0, abs(th[k]))
                tp = th.copy()
                tp[k] += h
                J[:, k] = (res(tp) - r0) / h
            step, *_ = np.linalg.lstsq(J, -r0, rcond=None)
            th = th + step
            if np.linalg.norm(step) < 1e-10:
                break
    return th


def _refine_mirror(P, N, tree, NT, o, n, tau, dmax, iters=8):
    n0 = unit(n)
    u, v, _ = orthonormal_frame(n0)
    o0 = np.asarray(o, float)

    def make(th):
        nn = unit(n0 + th[0] * u + th[1] * v)
        return mirror_matrix(o0 + th[2] * n0, nn)
    th = _icp(P, N, tree, NT, make, np.zeros(3), tau, dmax, iters)
    nn = unit(n0 + th[0] * u + th[1] * v)
    return o0 + th[2] * n0, nn


def _refine_rot(P, N, tree, NT, p, d, ang, tau, dmax, iters=8):
    d0 = unit(d)
    u, v, _ = orthonormal_frame(d0)
    p0 = np.asarray(p, float)

    def make(th):
        dd = unit(d0 + th[0] * u + th[1] * v)
        return rot_matrix(p0 + th[2] * u + th[3] * v, dd, ang)
    th = _icp(P, N, tree, NT, make, np.zeros(4), tau, dmax, iters)
    return p0 + th[2] * u + th[3] * v, unit(d0 + th[0] * u + th[1] * v)


def detect_global(model, tol=0.05, features=(), min_score=0.85, max_n=12, log=None):
    """Globale Symmetrien des Bauteils. Rückgabe: Liste von Symmetrie-dicts (nach Güte)."""
    log = log or (lambda s: None)
    P0, N0 = _sample(model, 20000)
    # Ziel (Baum) und Quelle sind verschiedene Stichproben; die Güte wird auf die Selbst-
    # übereinstimmung (Identität) normiert -> Abtastung und Kanten fallen heraus
    Pt, Nt = P0[0::2], N0[0::2]
    P, N = P0[1::2], N0[1::2]
    tree = cKDTree(Pt)
    el = model.mean_edge_length()
    tau = max(3 * tol, 0.02)                 # Punkt-zu-Ebene
    dmax = 3 * el + 3 * tol                  # Nachbarsuche
    base = score_transform(P, N, tree, np.eye(3), np.zeros(3), tau, Nt, dmax)[0]
    base = max(base, 0.5)

    def score(src, nsrc, R, t):
        return min(1.0, score_transform(src, nsrc, tree, R, t, tau, Nt, dmax)[0] / base)

    c = P0.mean(0)
    X = P0 - c
    _, _, Vt = np.linalg.svd(X, full_matrices=False)
    # Kandidaten: zuerst exakte Feature-Richtungen (große Ebenen-Normalen, Achsen), dann Hauptachsen
    from .analysis import feature_directions
    D, W, _ = feature_directions([f for f in features if f.role == "surface" or f.kind == "hole"])
    dirs = []
    for dv in list(D[np.argsort(-W)[:12]]) + [Vt[0], Vt[1], Vt[2]]:
        dv = unit(dv)
        if all(abs(float(dv @ u)) < math.cos(math.radians(3)) for u in dirs):
            dirs.append(dv)
    # globale Symmetrie-Elemente gehen durch den Flächenschwerpunkt
    centers = [c]
    Pc, Nc = P[:: max(1, len(P) // 4000)], N[:: max(1, len(P) // 4000)]
    found = []
    # Spiegelungen
    for dv in dirs:
        o, n = c - ((c - c) @ dv) * dv, unit(dv)
        o, n = _refine_mirror(Pc, Nc, tree, Nt, o, n, tau, dmax, iters=4)
        R, t = mirror_matrix(o, n)
        s0, *_ = (score(Pc, Nc, R, t),)
        if s0 < 0.5:
            continue
        o, n = _refine_mirror(P, N, tree, Nt, o, n, tau, dmax)
        R, t = mirror_matrix(o, n)
        s, *_ = (score(P, N, R, t),)
        if s >= min_score:
            found.append({"type": "mirror", "o": o, "nrm": n, "score": s, "scope": "global", "features": []})
    # Drehungen: für jede Achse alle n prüfen; höchstes n mit guter Übereinstimmung melden,
    # dazu den größten exakten Teiler, falls das höchste n nur näherungsweise gilt
    for dv in dirs:
        for pc in centers:
            cand = []
            for n in range(max_n, 1, -1):
                p0, d0 = _refine_rot(Pc, Nc, tree, Nt, pc, dv, 2 * math.pi / n, tau, dmax, iters=3)
                R, t = rot_matrix(p0, d0, 2 * math.pi / n)
                s0, *_ = (score(Pc, Nc, R, t),)
                if s0 < 0.5:
                    continue
                p, d = _refine_rot(P, N, tree, Nt, p0, d0, 2 * math.pi / n, tau, dmax)
                ss = [score(P, N, *rot_matrix(p, d, 2 * math.pi * k / n)) for k in range(1, n)]
                if min(ss) >= min_score:
                    cand.append((n, p, d, min(ss)))
            if not cand:
                continue
            n, p, d, s = cand[0]
            sinf = min(score(P, N, *rot_matrix(p, d, math.radians(a))) for a in (17, 53, 131))
            if sinf >= min_score:
                found.append({"type": "rot", "n": 0, "p": p, "d": d, "score": sinf, "scope": "global", "features": []})
                continue
            found.append({"type": "rot", "n": n, "p": p, "d": d, "score": s, "scope": "global", "features": []})
            if s < 0.997:
                exact = [c for c in cand if c[3] >= s + 0.005 and n % c[0] == 0]
                if exact:
                    n2, p2, d2, s2 = exact[0]
                    found.append({"type": "rot", "n": n2, "p": p2, "d": d2, "score": s2, "scope": "global",
                                  "features": []})
    return _dedupe(found, dmax)


def _dedupe(found, tau):
    out = []
    for s in sorted(found, key=lambda s: -s["score"]):
        dup = False
        for q in out:
            if q["type"] != s["type"]:
                continue
            if s["type"] == "mirror":
                if abs(float(unit(q["nrm"]) @ unit(s["nrm"]))) > math.cos(math.radians(2)) and \
                        abs(float((s["o"] - q["o"]) @ unit(q["nrm"]))) < 2 * tau:
                    dup = True
            else:
                w = s["p"] - q["p"]
                dq = unit(q["d"])
                if abs(float(dq @ unit(s["d"]))) > math.cos(math.radians(2)) and \
                        np.linalg.norm(w - (w @ dq) * dq) < 2 * tau:
                    if s.get("n", 0) == q.get("n", 0):
                        dup = True
            if dup:
                break
        if not dup:
            out.append(s)
    return out


# ===================================================================== Feature-Muster
def _feature_axis(f):
    p = f.params
    if f.kind == "hole":
        return np.asarray(p["point"], float), unit(p["axis"]), 0.5 * float(p["d"])
    if f.kind == "cylinder":
        return np.asarray(p["point"], float), unit(p["axis"]), float(p["radius"])
    if f.kind == "cone":
        return np.asarray(p["apex"], float), unit(p["axis"]), float(math.degrees(p["half_angle"]))
    if f.kind == "torus":
        return np.asarray(p["center"], float), unit(p["axis"]), float(p["R"])
    if f.kind == "sphere":
        return np.asarray(p["center"], float), None, float(p["radius"])
    return None


def _fit_circle_3d(C, d):
    x, y, _ = orthonormal_frame(d)
    Q = np.c_[C @ x, C @ y]
    A = np.c_[2 * Q, np.ones(len(Q))]
    sol, *_ = np.linalg.lstsq(A, (Q ** 2).sum(1), rcond=None)
    cc = sol[:2]
    r = math.sqrt(max(sol[2] + cc @ cc, 0))
    res = np.abs(np.linalg.norm(Q - cc, axis=1) - r)
    h = float(np.mean(C @ d))
    center = cc[0] * x + cc[1] * y + h * d
    ang = np.arctan2(Q[:, 1] - cc[1], Q[:, 0] - cc[0])
    return center, r, float(res.max()), ang


def detect_patterns(features, tol=0.05):
    """Lochkreise, koaxiale Gruppen, Spiegelpaare. Rückgabe: Liste Symmetrie-dicts (scope='features')."""
    out = []
    ax = [(f, *_feature_axis(f)) for f in features if _feature_axis(f) is not None and f.kind != "sphere"]
    lim = max(4 * tol, 0.15)
    # gleiche Art + gleicher "Radius" + parallel -> Muster-Kandidaten
    groups = []
    for f, p, d, r in ax:
        for g in groups:
            f0, p0, d0, r0 = g[0]
            if f.kind == f0.kind and abs(r - r0) < max(lim, 0.02 * r0) and abs(float(d @ d0)) > math.cos(math.radians(2)):
                g.append((f, p, d, r))
                break
        else:
            groups.append([(f, p, d, r)])
    for g in groups:
        if len(g) < 2:
            continue
        d = unit(sum(x[2] * np.sign(x[2] @ g[0][2]) for x in g))
        # Projektion der Achspunkte auf eine gemeinsame Ebene ⟂ d
        C = np.array([x[1] - (x[1] @ d) * d for x in g])
        # koaxial?
        if np.max(np.linalg.norm(C - C.mean(0), axis=1)) < lim:
            continue
        if len(g) >= 3:
            center, rr, err, ang = _fit_circle_3d(C, d)
            if err < max(lim, 0.01 * rr) and rr > lim:
                a = np.sort(np.mod(ang - ang.min(), 2 * math.pi))
                n = len(g)
                step = 2 * math.pi / n
                if np.max(np.abs(np.diff(np.r_[a, 2 * math.pi]) - step)) < math.radians(1.5):
                    out.append({"type": "rot", "n": n, "p": center, "d": d, "score": 1.0, "scope": "features",
                                "features": [x[0].id for x in g],
                                "info": (tr(f"Lochkreis Ø{2 * rr:.3f}", f"bolt circle Ø{2 * rr:.3f}")
                                         if all(x[0].kind == "hole" for x in g) else
                                         tr(f"Kreismuster Ø{2 * rr:.3f}", f"circular pattern Ø{2 * rr:.3f}"))})
        if len(g) == 2:
            mid = 0.5 * (C[0] + C[1])
            nrm = unit(C[1] - C[0])
            out.append({"type": "mirror", "o": mid, "nrm": nrm, "score": 1.0, "scope": "features",
                        "features": [x[0].id for x in g], "info": tr("Spiegelpaar", "mirror pair")})
    # koaxiale Rotationsflächen
    co = []
    for f, p, d, r in ax:
        if f.kind not in ("cylinder", "cone", "torus", "hole"):
            continue
        for g in co:
            f0, p0, d0, _ = g[0]
            w = p - p0
            if abs(float(d @ d0)) > math.cos(math.radians(2)) and np.linalg.norm(w - (w @ d0) * d0) < lim:
                g.append((f, p, d, r))
                break
        else:
            co.append([(f, p, d, r)])
    for g in co:
        if len(g) >= 2:
            d = unit(g[0][2])
            p = np.mean([x[1] - ((x[1] - g[0][1]) @ d) * d for x in g], axis=0)
            out.append({"type": "rot", "n": 0, "p": p, "d": d, "score": 1.0, "scope": "features",
                        "features": [x[0].id for x in g], "info": tr("koaxial", "coaxial")})
    return out


def detect_all(model, features, tol=0.05, existing=(), log=None, min_score=0.85):
    tau = max(3 * tol, 1.2 * model.mean_edge_length())
    pats, seen = [], set()
    for s in detect_patterns(features, tol):
        key = (s["type"], s.get("n", 0), frozenset(s["features"]))
        if key not in seen:
            seen.add(key)
            pats.append(s)
    res = detect_global(model, tol, features, min_score=min_score, log=log) + pats
    out = []
    for s in res:
        s["id"] = new_sym_id(list(existing) + out)
        s["name"] = f"S{s['id']}"
        out.append(s)
    return out


def sym_to_json(s):
    d = {}
    for k, v in s.items():
        d[k] = v.tolist() if isinstance(v, np.ndarray) else v
    return d


def sym_from_json(d):
    s = dict(d)
    for k in ("o", "nrm", "p", "d"):
        if k in s and s[k] is not None:
            s[k] = np.asarray(s[k], float)
    return s
