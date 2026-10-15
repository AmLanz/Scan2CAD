"""Drehprofil-Analyse: Flächen um eine gemeinsame Achse (Drehteile, Bohrungen mit Senkungen,
Einstichen, Ringnuten, Fasen und Rundungen).

Bei einer Rotationsfläche liegt jede Normale in der Meridianebene durch die Achse. Bildet man
alle passenden Messpunkte auf (r, z) ab (r = Abstand zur Achse, z = Lage entlang der Achse),
fällt die ganze Fläche auf ihre Profilkurve zusammen – und diese besteht bei technischen Teilen
aus Strecken und Kreisbögen:

    waagrechte Strecke  → Ebene ⟂ Achse        senkrechte Strecke → Zylinder
    schräge Strecke     → Kegel                Kreisbogen          → Torus (Rundung, Ringnut)
    Kreisbogen mit Mittelpunkt auf der Achse   → Kugel

In 2D ist die Erkennung viel robuster als in 3D: alle Facettenringe einer Rundung liegen auf
einem Profilkreis, kleine Ringnuten haben viele Messpunkte, Achse und Lage sind gemeinsam.
"""
from __future__ import annotations

import math

import numpy as np

from .geom_utils import unit


# ------------------------------------------------------------------ Stützpunkte
def profile_samples(model, faces, q, a, max_tilt_deg=12.0):
    """(r, z, Normale in (r, z), Gewicht, Dreieck je Stützpunkt). Nur Punkte, deren Normale
    (fast) in der Meridianebene liegt. CAD-Export: Eckpunkte (exakt), sonst Schwerpunkte."""
    faces = np.asarray(faces, np.int64)
    q = np.asarray(q, float)
    a = unit(np.asarray(a, float))
    if model.exact:
        F = model.F[faces]
        P = model.V[F].reshape(-1, 3)
        N = np.repeat(model.normals[faces], 3, axis=0)
        W = np.repeat(model.weights[faces] / 3.0, 3)
        owner = np.repeat(faces, 3)
    else:
        P = model.centroids[faces]
        N = model.fit_normals[faces]
        W = model.weights[faces]
        owner = faces
    rel = P - q
    z = rel @ a
    radial = rel - np.outer(z, a)
    r = np.linalg.norm(radial, axis=1)
    er = radial / np.maximum(r, 1e-12)[:, None]
    ephi = np.cross(a, er)
    tilt = np.abs(np.einsum("ij,ij->i", N, ephi))
    nr = np.einsum("ij,ij->i", N, er)
    nz = N @ a
    ln = np.maximum(np.hypot(nr, nz), 1e-12)
    keep = (tilt < math.sin(math.radians(max_tilt_deg))) | (r < 1e-9)
    return r, z, np.c_[nr / ln, nz / ln], W, owner, keep


# ------------------------------------------------------------------ 2D-Primitive
def _line_dist(prim, Q):
    p, d = prim["p"], prim["d"]
    v = Q - p
    return np.abs(v[:, 0] * d[1] - v[:, 1] * d[0])


def _line_normal(prim, Q):
    d = prim["d"]
    return np.tile([-d[1], d[0]], (len(Q), 1))


def _circ_dist(prim, Q):
    return np.abs(np.linalg.norm(Q - prim["c"], axis=1) - prim["rho"])


def _circ_normal(prim, Q):
    return unit(Q - prim["c"])


def prim_dist(prim, Q):
    return _line_dist(prim, Q) if prim["type"] == "line" else _circ_dist(prim, Q)


def prim_normal(prim, Q):
    return _line_normal(prim, Q) if prim["type"] == "line" else _circ_normal(prim, Q)


def _fit_line(Q, w):
    c = (Q * w[:, None]).sum(0) / w.sum()
    X = Q - c
    C = (X * w[:, None]).T @ X
    ev, evec = np.linalg.eigh(C)
    return {"type": "line", "p": c, "d": unit(evec[:, 1])}


def _fit_circle(Q, w):
    from .geom_utils import fit_circle_2d
    try:
        c, rho, _ = fit_circle_2d(Q, w)
    except Exception:
        return None
    if not np.isfinite(rho) or rho <= 0:
        return None
    return {"type": "circle", "c": np.asarray(c, float), "rho": float(rho)}


def _inl(prim, Q, N, eps, cos_a):
    d = prim_dist(prim, Q)
    ok = d < eps
    if ok.any():
        g = prim_normal(prim, Q[ok])
        c = np.abs(np.einsum("ij,ij->i", g, N[ok]))
        ok[np.nonzero(ok)[0][c < cos_a]] = False
    return ok


def _same_prim(a, b, eps):
    if a["type"] != b["type"]:
        return False
    if a["type"] == "line":
        da, db = unit(a["d"]), unit(b["d"])
        if abs(float(da @ db)) < math.cos(math.radians(1.0)):
            return False
        v = b["p"] - a["p"]
        return abs(v[0] * da[1] - v[1] * da[0]) < 0.5 * eps
    return float(np.linalg.norm(a["c"] - b["c"])) < eps and abs(a["rho"] - b["rho"]) < eps


def ransac_profile(Q, N, W, eps, min_weight, angle_deg=15.0, max_prims=40, trials=250, seed=0,
                   max_circle=None, fixed_cands=(), face_pts=None, face_w=None):
    """Strecken und Kreise im (r, z)-Profil, größte zuerst (Gewicht = Fläche).

    Punktmodus (Scan): Gewicht = Fläche der noch freien Punkte.
    Dreiecksmodus (CAD, face_pts = Indizes der drei Profilpunkte je Dreieck): ein Dreieck zählt,
    wenn alle drei Eckpunkte passen – Randkreise gehören so zu beiden Nachbarflächen.
    fixed_cands: zusätzlich in jeder Runde geprüfte Kandidaten."""
    rng = np.random.default_rng(seed)
    cos_a = math.cos(math.radians(angle_deg))
    prims = []
    span = float(np.ptp(Q, axis=0).max()) if len(Q) else 1.0
    max_circle = max_circle or 2.0 * span
    allidx = np.arange(len(Q))
    face_mode = face_pts is not None
    if face_mode:
        face_pts = np.asarray(face_pts, np.int64)
        face_w = np.asarray(face_w, float)
        f_rem = np.ones(len(face_pts), bool)
    remaining = np.ones(len(Q), bool)

    def gain(c):
        ok = _inl(c, Q, N, eps, cos_a)
        if face_mode:
            cov = ok[face_pts].all(1) & f_rem
            return float(face_w[cov].sum()), ok, cov
        return float(W[ok & remaining].sum()), ok, None

    for _ in range(max_prims):
        if face_mode:
            fidx = np.nonzero(f_rem)[0]
            if len(fidx) == 0 or face_w[fidx].sum() < min_weight:
                break
            pf = face_w[fidx] / face_w[fidx].sum()
        else:
            idx = np.nonzero(remaining)[0]
            if len(idx) < 1 or W[idx].sum() < min_weight:
                break
            p = W[idx] / W[idx].sum()
        best = None
        for c in fixed_cands:
            g, _, _ = gain(c)
            g *= 0.97 if c["type"] == "circle" else 1.0
            if best is None or g > best[0]:
                best = (g, c)
        for _t in range(trials):
            k3 = None
            if face_mode:
                fi = rng.choice(fidx, p=pf)
                pts = face_pts[fi]
                i = int(rng.choice(pts))
                others = [int(t) for t in pts if np.linalg.norm(Q[t] - Q[i]) > 2 * eps]
                j = others[0] if others else None
                if j is not None:
                    # dritter Punkt in der Nähe (nächster Facettenring) für Kreise durch 3 Punkte
                    lij = float(np.linalg.norm(Q[j] - Q[i]))
                    d3 = np.minimum(np.linalg.norm(Q - Q[i], axis=1), np.linalg.norm(Q - Q[j], axis=1))
                    near3 = allidx[(d3 > 2 * eps) & (d3 < 3 * lij)]
                    if len(near3):
                        k3 = int(rng.choice(near3))
            else:
                i = rng.choice(idx, p=p)
                d2 = np.linalg.norm(Q - Q[i], axis=1)
                near = allidx[(d2 > 2 * eps) & (d2 < 0.25 * span)]
                j = int(rng.choice(near)) if len(near) else None
            cands = [{"type": "line", "p": Q[i], "d": np.array([N[i, 1], -N[i, 0]])}]
            if k3 is not None:
                cc = _fit_circle(np.vstack([Q[i], Q[j], Q[k3]]), np.ones(3))
                if cc is not None and 3 * eps < cc["rho"] < max_circle:
                    cands.append(cc)
            if j is not None:
                # Strecke durch zwei Punkte (grob facettierte CAD-Kegel: Facettennormalen weichen
                # um einige Grad ab, die Punkte selbst sind exakt) und Kreis aus den Normalen
                cands.append({"type": "line", "p": Q[i], "d": unit(Q[j] - Q[i])})
                A = np.array([N[i], -N[j]]).T
                if abs(np.linalg.det(A)) > 1e-3:
                    t = np.linalg.solve(A, Q[j] - Q[i])
                    c = Q[i] + t[0] * N[i]
                    rho = 0.5 * (np.linalg.norm(Q[i] - c) + np.linalg.norm(Q[j] - c))
                    if 3 * eps < rho < max_circle:
                        cands.append({"type": "circle", "c": c, "rho": float(rho)})
            for c in cands:
                g, _, _ = gain(c)
                g *= 0.97 if c["type"] == "circle" else 1.0
                if best is None or g > best[0]:
                    best = (g, c)
        if best is None or best[0] < min_weight:
            break
        prim = best[1]
        # Verfeinern mit allen passenden Punkten (auch geteilten Randkreisen)
        for _r in range(3):
            ok = _inl(prim, Q, N, 2 * eps if _r == 0 else eps, cos_a)
            if ok.sum() < 2:
                break
            fit = _fit_line(Q[ok], W[ok]) if prim["type"] == "line" else _fit_circle(Q[ok], W[ok])
            if fit is None:
                break
            prim = fit
        g, ok, cov = gain(prim)
        if g < min_weight:
            # Verfeinerung ist weggelaufen → ursprünglichen Kandidaten nehmen
            prim = best[1]
            g, ok, cov = gain(prim)
            if g < min_weight:
                if face_mode:
                    f_rem[cov] = False
                else:
                    remaining[ok] = False
                continue
        if prim["type"] == "circle":
            # Kreis mit riesigem Radius ist eine Strecke
            lf = _fit_line(Q[ok], W[ok])
            if np.sqrt(np.average(_line_dist(lf, Q[ok]) ** 2, weights=W[ok])) < 0.5 * eps:
                prim = lf
                g, ok, cov = gain(prim)
        prim["weight"] = g
        if any(_same_prim(prim, q, eps) for q in prims):
            # Verfeinerung ist auf ein schon gefundenes Primitiv zurückgelaufen
            if face_mode:
                f_rem[cov] = False
            remaining[ok] = False
            continue
        prims.append(prim)
        if face_mode:
            f_rem[cov] = False
        remaining[ok] = False
    return prims


# ------------------------------------------------------------------ 3D-Features
def prim_to_feature_params(prim, q, a, eps, snap_deg=0.5):
    """2D-Primitiv → (kind, params) im Raum."""
    q = np.asarray(q, float)
    a = unit(np.asarray(a, float))
    if prim["type"] == "line":
        d = unit(prim["d"])
        p = prim["p"]
        ang = math.degrees(math.atan2(abs(d[1]), abs(d[0])))        # 0 = waagrecht (r-Richtung)
        if ang < snap_deg:                                            # z = const → Ebene ⟂ Achse
            return "plane", {"origin": q + p[1] * a, "normal": a.copy()}
        if ang > 90 - snap_deg:                                       # r = const → Zylinder
            return "cylinder", {"point": q + p[1] * a, "axis": a.copy(), "radius": float(abs(p[0]))}
        # Kegel: r = p_r + k (z − p_z)
        k = d[0] / d[1]
        z_apex = p[1] - p[0] / k
        half = math.atan(abs(k))
        axis = a.copy() if k > 0 else -a
        return "cone", {"apex": q + z_apex * a, "axis": axis, "half_angle": float(half)}
    c, rho = prim["c"], prim["rho"]
    if abs(c[0]) < max(2 * eps, 1e-3 * rho):
        return "sphere", {"center": q + c[1] * a, "radius": float(rho)}
    return "torus", {"center": q + c[1] * a, "axis": a.copy(), "R": float(abs(c[0])), "r": float(rho)}


def segment_rotational(model, faces, q, a, eps, min_area, angle_deg=15.0, max_tilt_deg=12.0, seed=0):
    """Rotationsflächen um die Achse (q, a) in den Dreiecken 'faces' finden.
    Rückgabe: Liste (kind, params, face_ids, rms)."""
    r, z, N2, W, owner, keep = profile_samples(model, faces, q, a, max_tilt_deg)
    if keep.sum() < 10:
        return []
    Q = np.c_[r, z]
    idx = np.nonzero(keep)[0]
    Qk, Nk, Wk = Q[idx], N2[idx], W[idx]
    # Exakte CAD-Punkte fallen ringweise auf denselben Profilpunkt → zusammenfassen
    face_pts = face_w = None
    if model.exact:
        key = np.round(Qk / max(1e-6, 0.05 * eps)).astype(np.int64)
        _, first, inv = np.unique(key[:, 0] * 1_000_003 + key[:, 1] * 7 + np.round(Nk[:, 1] * 1000).astype(np.int64),
                                  return_index=True, return_inverse=True)
        Wu = np.bincount(inv, Wk)
        Qu, Nu = Qk[first], Nk[first]
        kf = keep.reshape(-1, 3).all(1)
        pos = -np.ones(len(Q), np.int64)
        pos[idx] = inv
        face_pts = pos.reshape(-1, 3)[kf]
        face_w = model.weights[np.asarray(faces, np.int64)[kf]]
    else:
        Qu, Nu, Wu = Qk, Nk, Wk
        if len(Qu) > 20000:
            sel = np.random.default_rng(seed).choice(len(Qu), 20000, replace=False, p=Wu / Wu.sum())
            Qu, Nu, Wu = Qu[sel], Nu[sel], np.full(len(sel), Wu.sum() / len(sel))
    fixed = []
    if model.exact:
        # jedes Dreieck liefert exakte Kandidaten: Strecke durch 2, Kreis durch 3 verschiedene Profilpunkte
        T2 = Q.reshape(-1, 3, 2)
        kf = keep.reshape(-1, 3).all(1)
        seen = set()
        for tri in T2[kf]:
            u = np.unique(np.round(tri / max(1e-6, 0.1 * eps)), axis=0) * max(1e-6, 0.1 * eps)
            if len(u) == 2:
                d = unit(u[1] - u[0])
                if d[1] < 0 or (abs(d[1]) < 1e-12 and d[0] < 0):
                    d = -d
                c0 = float(u[0] @ np.array([-d[1], d[0]]))
                key = ("l", round(float(d[0]), 4), round(float(d[1]), 4), round(c0 / max(eps, 1e-6)))
                if key not in seen:
                    seen.add(key)
                    fixed.append({"type": "line", "p": u[0], "d": d})
            elif len(u) == 3:
                cc = _fit_circle(u, np.ones(3))
                if cc is not None and cc["rho"] < 2.0 * float(np.ptp(Qu, axis=0).max()):
                    key = ("c", round(float(cc["c"][0] / max(eps, 1e-6))), round(float(cc["c"][1] / max(eps, 1e-6))),
                           round(cc["rho"] / max(eps, 1e-6)))
                    if key not in seen:
                        seen.add(key)
                        fixed.append(cc)
            if len(fixed) > 3000:
                break
    prims = ransac_profile(Qu, Nu, Wu, eps, min_area, angle_deg=angle_deg, seed=seed, fixed_cands=fixed,
                           face_pts=face_pts, face_w=face_w)
    if not prims:
        return []
    # Dreiecke zuordnen: alle Stützpunkte eines Dreiecks müssen passen
    cos_a = math.cos(math.radians(angle_deg))
    best_d = np.full(len(Q), np.inf)
    best_k = -np.ones(len(Q), np.int64)
    for k, pr in enumerate(prims):
        d = prim_dist(pr, Q)
        ok = _inl(pr, Q, N2, eps, cos_a) & keep
        better = ok & (d < best_d)
        best_d[better] = d[better]
        best_k[better] = k
    lab_s = best_k
    # Dreieck → Primitiv: bei CAD müssen alle 3 Eckpunkte dasselbe Primitiv haben
    if model.exact:
        L3 = lab_s.reshape(-1, 3)
        same = (L3[:, 0] == L3[:, 1]) & (L3[:, 1] == L3[:, 2])
        flab = np.where(same, L3[:, 0], -1)
        # Punkte auf Kanten zwischen zwei Flächen passen zu beiden – dann Mehrheit über die
        # Abstände: Dreieck gehört zum Primitiv, auf dem alle drei Eckpunkte liegen
        for k, pr in enumerate(prims):
            d3 = prim_dist(pr, Q).reshape(-1, 3)
            n3 = _inl(pr, Q, N2, eps, cos_a).reshape(-1, 3)
            allin = n3.all(1) & (d3.max(1) < eps)
            flab = np.where((flab < 0) & allin, k, flab)
        face_lab = flab
    else:
        face_lab = lab_s
    out = []
    for k, pr in enumerate(prims):
        sel = np.nonzero(face_lab == k)[0]
        if len(sel) == 0:
            continue
        fids = np.asarray(faces, np.int64)[sel]
        if model.areas[fids].sum() < min_area:
            continue
        kind, params = prim_to_feature_params(pr, q, a, eps)
        dd = prim_dist(pr, Q[np.isin(owner, fids)])
        out.append((kind, params, fids, float(np.sqrt(np.mean(dd ** 2))) if len(dd) else 0.0))
    return out


# ------------------------------------------------------------------ Achse finden
def dominant_axis(features, model, ang_deg=1.0, tol=0.05):
    """Gemeinsame Achse der rotationssymmetrischen Features (Zylinder, Kegel, Tori, Kugeln,
    Bohrungen), flächengewichtet. Rückgabe (q, a, Flächenanteil) oder None."""
    items = []
    for f in features:
        p = f.params
        if f.kind in ("cylinder", "cone", "torus", "hole"):
            ax = unit(p.get("axis"))
            pt = np.asarray(p.get("point", p.get("apex", p.get("center"))), float)
        else:
            continue
        ids = np.asarray(f.face_ids, np.int64)
        w = float(model.areas[ids].sum()) if len(ids) else 0.0
        items.append((pt, ax, w))
    if not items:
        return None
    best = None
    for pt, ax, _ in items:
        tot = 0.0
        for pt2, ax2, w2 in items:
            if abs(float(ax @ ax2)) < math.cos(math.radians(ang_deg)):
                continue
            d = pt2 - pt
            if np.linalg.norm(d - (d @ ax) * ax) < max(10 * tol, 0.2):
                tot += w2
        if best is None or tot > best[2]:
            best = (pt, ax, tot)
    return best


def rotational_fraction(model, q, a, max_tilt_deg=5.0):
    """Flächenanteil, dessen Normalen in Meridianebenen um die Achse liegen."""
    faces = np.nonzero(~model.synthetic)[0]
    r, z, N2, W, owner, keep = profile_samples(model, faces, q, a, max_tilt_deg)
    return float(W[keep].sum() / max(W.sum(), 1e-300))
