"""2D-Profile für Extrusions- und Rotationskörper.

Profil-Darstellung (im lokalen 2D-Rahmen, mm):
  {"type": "circle", "c": [x, y], "r": r}
  {"type": "poly",   "pts": [[x, y], ...], "fillets": [r_i, ...]}   geschlossen, Ecken optional verrundet
  {"type": "spline", "pts": [[x, y], ...], "closed": bool}         Interpolationspunkte
  {"type": "polyline", "pts": [[x, y], ...], "closed": bool}       offene/geschlossene Linienzüge
"""
from __future__ import annotations

import numpy as np
from scipy.interpolate import splev, splprep
from scipy.spatial import cKDTree

from .geom_utils import (dist_points_to_polyline, douglas_peucker, fit_circle_2d,
                         intersect_lines_2d, line_fit_2d, min_area_rect, polygon_area_2d,
                         polyline_length, resample_polyline)


# ----------------------------------------------------------------- Hilfen
def _corner_geometry(prev, cur, nxt):
    d1 = prev - cur
    d2 = nxt - cur
    l1, l2 = np.linalg.norm(d1), np.linalg.norm(d2)
    if l1 < 1e-12 or l2 < 1e-12:
        return None
    u1, u2 = d1 / l1, d2 / l2
    theta = np.arccos(np.clip(u1 @ u2, -1, 1))  # Innenwinkel zwischen den Kanten
    return u1, u2, l1, l2, theta


def max_fillet_radius(pts, i):
    n = len(pts)
    g = _corner_geometry(pts[i - 1], pts[i], pts[(i + 1) % n])
    if g is None:
        return 0.0
    u1, u2, l1, l2, th = g
    if th < 1e-3 or th > np.pi - 1e-3:
        return 0.0
    return 0.49 * min(l1, l2) * np.tan(th / 2)


def clamp_fillets(pts, fillets):
    pts = np.asarray(pts, float)
    out = []
    for i, r in enumerate(fillets):
        out.append(float(max(0.0, min(r, max_fillet_radius(pts, i)))))
    return out


def poly_to_dense(pts, fillets=None, step=0.1, closed=True):
    """Verrundetes Polygon/Linienzug als dichte Punktfolge."""
    pts = np.asarray(pts, float)
    n = len(pts)
    fillets = fillets or [0.0] * n
    out = []
    for i in range(n):
        r = fillets[i]
        if not closed and (i == 0 or i == n - 1):
            out.append(pts[i])
            continue
        g = _corner_geometry(pts[i - 1], pts[i], pts[(i + 1) % n])
        if r <= 0 or g is None:
            out.append(pts[i])
            continue
        u1, u2, l1, l2, th = g
        t = r / np.tan(th / 2)
        a = pts[i] + u1 * t
        b = pts[i] + u2 * t
        bis = (u1 + u2)
        bis /= np.linalg.norm(bis)
        c = pts[i] + bis * (r / np.sin(th / 2))
        a0 = np.arctan2(*(a - c)[::-1])
        a1 = np.arctan2(*(b - c)[::-1])
        da = (a1 - a0 + np.pi) % (2 * np.pi) - np.pi
        m = max(2, int(np.ceil(abs(da) * r / max(step, 1e-6))))
        for s in np.linspace(0, 1, m + 1):
            ang = a0 + s * da
            out.append(c + r * np.array([np.cos(ang), np.sin(ang)]))
    return np.array(out)


def profile_to_dense(prof, step=0.1):
    t = prof["type"]
    if t == "circle":
        c = np.asarray(prof["c"], float)
        r = float(prof["r"])
        m = max(24, int(np.ceil(2 * np.pi * r / max(step, 1e-6))))
        a = np.linspace(0, 2 * np.pi, m, endpoint=False)
        return c + r * np.c_[np.cos(a), np.sin(a)], True
    if t == "poly":
        return poly_to_dense(prof["pts"], prof.get("fillets"), step), True
    if t == "poly_open":
        return poly_to_dense(prof["pts"], prof.get("fillets"), step, closed=False), False
    if t == "spline":
        pts = np.asarray(prof["pts"], float)
        closed = prof.get("closed", True)
        try:
            k = min(3, len(pts) - 1)
            tck, _ = splprep(pts.T, s=0, per=1 if closed else 0, k=k)
            L = polyline_length(pts, closed)
            m = max(50, int(np.ceil(L / max(step, 1e-6))))
            uu = np.linspace(0, 1, m, endpoint=not closed)
            return np.array(splev(uu, tck)).T, closed
        except Exception:
            return pts, closed
    if t == "polyline":
        return np.asarray(prof["pts"], float), prof.get("closed", False)
    raise ValueError(t)


def profile_residuals(prof, Q, step=None, exact=False):
    """Abstand Punkte -> Profil. Standard: KD-Baum auf dicht abgetastetem Profil
    (Fehler <= Abtastschritt/2, typ. viel kleiner); exact=True: Segmentabstand."""
    step = step or 0.05
    dense, closed = profile_to_dense(prof, step)
    if exact or len(dense) < 3:
        return dist_points_to_polyline(Q, dense, closed)
    L = polyline_length(dense, closed)
    dense = resample_polyline(dense, max(step, L / 20000), closed)
    d, _ = cKDTree(dense).query(np.asarray(Q, float))
    return d


# ----------------------------------------------------------------- Fits
def fit_circle_profile(Q, robust=None):
    c, r, d = fit_circle_2d(Q, None, robust)
    return {"type": "circle", "c": c.tolist(), "r": float(r)}


def _estimate_fillets(corners, loop_pts, snap=0.0, min_r=0.0):
    """Eckenradius aus Abstand Ecke -> Kontur: d = r (1/sin(θ/2) - 1)."""
    tree = cKDTree(loop_pts)
    n = len(corners)
    fil = []
    for i in range(n):
        g = _corner_geometry(corners[i - 1], corners[i], corners[(i + 1) % n])
        if g is None:
            fil.append(0.0)
            continue
        *_, th = g
        d, _ = tree.query(corners[i])
        k = 1.0 / np.sin(th / 2) - 1.0
        r = d / k if k > 1e-6 else 0.0
        if snap > 0:
            r = round(r / snap) * snap
        fil.append(float(r) if r >= min_r else 0.0)
    return clamp_fillets(corners, fil)


def fit_rect_profile(Q, loop_pts=None, fillets="auto", tol=0.05, snap=0.0):
    mid, ang, w, h = min_area_rect(Q)
    c, s = np.cos(ang), np.sin(ang)
    R = np.array([[c, s], [-s, c]])
    X = (Q - mid) @ R.T
    hw, hh = w / 2, h / 2
    # Punkte den Seiten zuordnen, Ecken-Nähe ausschließen
    dx = np.abs(np.abs(X[:, 0]) - hw)
    dy = np.abs(np.abs(X[:, 1]) - hh)
    side_x = dx < dy
    sel_x = side_x & (np.abs(X[:, 1]) < 0.7 * hh)
    sel_y = (~side_x) & (np.abs(X[:, 0]) < 0.7 * hw)
    xp = X[sel_x & (X[:, 0] > 0), 0]
    xn = X[sel_x & (X[:, 0] < 0), 0]
    yp = X[sel_y & (X[:, 1] > 0), 1]
    yn = X[sel_y & (X[:, 1] < 0), 1]
    x1 = np.median(xp) if len(xp) else hw
    x0 = np.median(xn) if len(xn) else -hw
    y1 = np.median(yp) if len(yp) else hh
    y0 = np.median(yn) if len(yn) else -hh
    corners_l = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])
    corners = corners_l @ R + mid
    lp = loop_pts if loop_pts is not None else Q
    if fillets == "auto":
        fil = _estimate_fillets(corners, lp, snap=snap, min_r=2 * tol)
        # Ecken ohne Messdaten (Teil-Scan) nicht schätzen
        side = min(x1 - x0, y1 - y0)
        tree = cKDTree(Q)
        supported = [len(tree.query_ball_point(c, 0.3 * side)) > 3 for c in corners]
        nz = [f for f, s in zip(fil, supported) if s and f > 0]
        if not all(supported):
            ref = float(np.median(nz)) if nz else 0.0
            fil = [f if s else ref for f, s in zip(fil, supported)]
            fil = clamp_fillets(corners, fil)
        # Rechteck: einheitlicher Radius ist meist Konstruktionsabsicht
        nz = [f for f, s in zip(fil, supported) if s and f > 0]
        if len(nz) >= 2 and np.std(nz) < 0.25 * np.mean(nz):
            fil = clamp_fillets(corners, [float(np.median(nz))] * 4)
    elif fillets in (None, "none"):
        fil = [0.0] * 4
    else:
        fil = clamp_fillets(corners, [float(fillets)] * 4)
    return {"type": "poly", "pts": corners.tolist(), "fillets": fil}


def _polygon_at(P, dp_tol, tol, fillets, snap):
    keep = douglas_peucker(P, dp_tol, closed=True)
    if len(keep) < 3:
        return None
    n = len(P)
    lines = []
    for j in range(len(keep)):
        a, b = keep[j], keep[(j + 1) % len(keep)]
        idx = np.arange(a, b + 1) if b > a else np.r_[np.arange(a, n), np.arange(0, b + 1)]
        m = len(idx)
        lo, hi = int(0.2 * m), int(np.ceil(0.8 * m))
        core = idx[lo:hi] if hi - lo >= 2 else idx
        c, d = line_fit_2d(P[core])
        # eine Nachiteration: nur Punkte nahe der Geraden
        nrm = np.array([-d[1], d[0]])
        dist = np.abs((P[idx] - c) @ nrm)
        near = idx[dist < max(2 * tol, 1e-6)]
        if len(near) >= 2:
            c, d = line_fit_2d(P[near])
        lines.append((c, d))
    changed = True
    while changed and len(lines) > 3:
        changed = False
        for j in range(len(lines)):
            c1, d1 = lines[j]
            c2, d2 = lines[(j + 1) % len(lines)]
            if abs(d1 @ d2) > np.cos(np.radians(4)):
                off = abs((c2 - c1) @ np.array([-d1[1], d1[0]]))
                if off > 3 * tol:      # parallel, aber versetzt (z. B. Langloch-Ende)
                    return None
                lines.pop((j + 1) % len(lines))
                changed = True
                break
    if len(lines) < 3:
        return None
    corners = []
    for j in range(len(lines)):
        x = intersect_lines_2d(lines[j - 1][0], lines[j - 1][1], lines[j][0], lines[j][1])
        if x is None:
            return None
        corners.append(x)
    corners = np.array(corners)
    if fillets == "auto":
        fil = _estimate_fillets(corners, P, snap=snap, min_r=2 * tol)
    elif fillets in (None, "none"):
        fil = [0.0] * len(corners)
    else:
        fil = clamp_fillets(corners, [float(fillets)] * len(corners))
    return {"type": "poly", "pts": corners.tolist(), "fillets": fil}


def fit_polygon_profile(loop_pts, tol=0.05, fillets="auto", snap=0.0, max_corners=24, cloud=None):
    """Polygon aus Kontur: mehrere DP-Stufen, je Kante Geradenfit, Ecken als
    Schnittpunkte, Eckenradien aus dem Abstand Ecke->Kontur. Gewählt wird die
    Variante mit den wenigsten Ecken, deren RMS nahe am besten liegt."""
    P = np.asarray(loop_pts, float)
    Lp = polyline_length(P, True)
    Q = P if cloud is None or len(cloud) == 0 else np.asarray(cloud, float)
    if len(Q) > 3000:
        Q = Q[np.random.default_rng(1).choice(len(Q), 3000, replace=False)]
    cands = []
    for dp in sorted(set([4 * tol, 0.005 * Lp, 0.01 * Lp, 0.02 * Lp, 0.04 * Lp])):
        try:
            p = _polygon_at(P, max(dp, 1e-6), tol, fillets, snap)
        except Exception:
            p = None
        if p is None or len(p["pts"]) > max_corners:
            continue
        r = float(np.sqrt(np.mean(profile_residuals(p, Q, step=max(tol / 4, 1e-3)) ** 2)))
        cands.append((len(p["pts"]), r, p))
    if not cands:
        return None
    best = min(c[1] for c in cands)
    ok = [c for c in cands if c[1] <= max(tol, 1.25 * best)]
    ok.sort(key=lambda c: (c[0], c[1]))
    return ok[0][2]


def _open_polygon_at(P, dp_tol, tol, fillets, snap):
    keep = douglas_peucker(P, dp_tol, closed=False)
    if len(keep) < 2:
        return None
    lines = []
    for j in range(len(keep) - 1):
        idx = np.arange(keep[j], keep[j + 1] + 1)
        m = len(idx)
        lo, hi = int(0.2 * m), int(np.ceil(0.8 * m))
        core = idx[lo:hi] if hi - lo >= 2 else idx
        c, d = line_fit_2d(P[core])
        nrm = np.array([-d[1], d[0]])
        near = idx[np.abs((P[idx] - c) @ nrm) < max(2 * tol, 1e-6)]
        if len(near) >= 2:
            c, d = line_fit_2d(P[near])
        lines.append((c, d))
    j = 0
    while j < len(lines) - 1:
        (c1, d1), (c2, d2) = lines[j], lines[j + 1]
        if abs(d1 @ d2) > np.cos(np.radians(4)):
            if abs((c2 - c1) @ np.array([-d1[1], d1[0]])) > 3 * tol:
                return None
            lines.pop(j + 1)
            continue
        j += 1
    pts = []
    c, d = lines[0]
    pts.append(c + ((P[0] - c) @ d) * d)
    for j in range(1, len(lines)):
        x = intersect_lines_2d(lines[j - 1][0], lines[j - 1][1], lines[j][0], lines[j][1])
        if x is None:
            return None
        pts.append(x)
    c, d = lines[-1]
    pts.append(c + ((P[-1] - c) @ d) * d)
    pts = np.array(pts)
    fil = [0.0] * len(pts)
    if fillets == "auto" and len(pts) > 2:
        tree = cKDTree(P)
        for i in range(1, len(pts) - 1):
            g = _corner_geometry(pts[i - 1], pts[i], pts[i + 1])
            if g is None:
                continue
            dd, _ = tree.query(pts[i])
            k = 1.0 / np.sin(g[4] / 2) - 1.0
            r = dd / k if k > 1e-6 else 0.0
            if snap > 0:
                r = round(r / snap) * snap
            fil[i] = r if r >= 2 * tol else 0.0
        for i in range(1, len(pts) - 1):
            fil[i] = float(min(fil[i], max_fillet_radius(pts, i)))
    return {"type": "poly_open", "pts": pts.tolist(), "fillets": fil}


def fit_open_polygon_profile(P, tol=0.05, fillets="auto", snap=0.0, cloud=None, max_corners=20):
    """Offener Linienzug aus Geraden + Eckenradien (z. B. Rotationsprofil Zapfen+Rundung)."""
    P = np.asarray(P, float)
    Lp = polyline_length(P, False)
    Q = P if cloud is None or len(cloud) == 0 else np.asarray(cloud, float)
    if len(Q) > 3000:
        Q = Q[np.random.default_rng(1).choice(len(Q), 3000, replace=False)]
    cands = []
    for dp in sorted(set([4 * tol, 0.005 * Lp, 0.01 * Lp, 0.02 * Lp, 0.04 * Lp])):
        try:
            p = _open_polygon_at(P, max(dp, 1e-6), tol, fillets, snap)
        except Exception:
            p = None
        if p is None or len(p["pts"]) > max_corners + 2:
            continue
        r = float(np.sqrt(np.mean(profile_residuals(p, Q, step=max(tol / 4, 1e-3)) ** 2)))
        cands.append((len(p["pts"]), r, p))
    if not cands:
        return None
    best = min(c[1] for c in cands)
    ok = [c for c in cands if c[1] <= max(tol, 1.25 * best)]
    ok.sort(key=lambda c: (c[0], c[1]))
    return ok[0][2]


def fit_spline_profile(loop_pts, closed=True, smooth=0.5, tol=0.05, step=2.0):
    """Glättender Spline; 'smooth' 0..1 skaliert den scipy-Glättungsfaktor."""
    P = np.asarray(loop_pts, float)
    if closed and np.linalg.norm(P[0] - P[-1]) < 1e-9:
        P = P[:-1]
    P = resample_polyline(P, max(tol, polyline_length(P, closed) / 1500), closed)
    m = len(P)
    sigma = max(tol, 1e-4) * (0.2 + 3.0 * smooth)
    try:
        tck, _ = splprep(P.T, s=m * sigma ** 2, per=1 if closed else 0, k=3)
        L = polyline_length(P, closed)
        k = int(np.clip(np.ceil(L / max(step, 1e-3)), 12, 400))
        uu = np.linspace(0, 1, k, endpoint=not closed)
        pts = np.array(splev(uu, tck)).T
    except Exception:
        idx = douglas_peucker(P, tol, closed)
        pts = P[idx]
    return {"type": "spline", "pts": pts.tolist(), "closed": closed}


def fit_profile(loop_pts, cloud, kind="auto", tol=0.05, fillets="auto", snap=0.0,
                spline_smooth=0.5, spline_step=2.0, closed=True, simplicity=1.25, max_poly_corners=16):
    """Passt ein Profil an eine geordnete Kontur (loop_pts) + Punktwolke (cloud) an."""
    cloud = np.asarray(cloud, float) if cloud is not None and len(cloud) else np.asarray(loop_pts)
    cands = {}
    if not closed:
        if kind in ("auto", "spline"):
            cands["spline"] = fit_spline_profile(loop_pts, False, spline_smooth, tol, spline_step)
        if kind in ("auto", "polygon"):
            idx = douglas_peucker(np.asarray(loop_pts), tol, False)
            cands["polygon"] = {"type": "polyline", "pts": np.asarray(loop_pts)[idx].tolist(), "closed": False}
        if kind == "circle":
            cands["circle"] = fit_circle_profile(cloud, tol)
    else:
        if kind in ("auto", "circle"):
            cands["circle"] = fit_circle_profile(cloud, tol)
        if kind in ("auto", "rect"):
            cands["rect"] = fit_rect_profile(cloud, loop_pts, fillets, tol, snap)
        if kind in ("auto", "polygon"):
            p = fit_polygon_profile(loop_pts, tol, fillets, snap, cloud=cloud)
            if p is not None:
                cands["polygon"] = p
        if kind in ("auto", "spline"):
            cands["spline"] = fit_spline_profile(loop_pts, True, spline_smooth, tol, spline_step)
    rms = {}
    sub = cloud if len(cloud) <= 2000 else cloud[np.random.default_rng(0).choice(len(cloud), 2000, replace=False)]
    for k, p in cands.items():
        try:
            d = profile_residuals(p, sub, step=max(tol / 4, 1e-3))
            # robustes Maß (80. Perzentil): einzelne Zacken/Ausreißer am Rand sollen
            # eine einfache Form nicht disqualifizieren
            rms[k] = float(np.percentile(d, 80))
        except Exception:
            pass
    if not rms:
        raise RuntimeError("Profil konnte nicht gefittet werden.")
    if kind != "auto":
        k = next(iter(rms))
        return cands[k], k, rms[k]
    # Flächenplausibilität: Profilfläche muss zur Kontur passen
    if closed and kind == "auto":
        a_raw = abs(polygon_area_2d(np.asarray(loop_pts, float)))
        for k in list(rms):
            try:
                a_k = abs(polygon_area_2d(profile_to_dense(cands[k], max(tol, 0.02))[0]))
                if a_raw > 0 and not (0.75 < a_k / a_raw < 1.33):
                    rms.pop(k)
            except Exception:
                rms.pop(k)
        if not rms:
            cands["spline"] = fit_spline_profile(loop_pts, True, spline_smooth, tol, spline_step)
            rms["spline"] = 0.0
    best = min(rms.values())
    for k in ("circle", "rect", "polygon", "spline"):
        if k in rms and rms[k] <= max(tol, simplicity * best):
            if k == "polygon" and len(cands[k]["pts"]) > max_poly_corners:
                continue
            return cands[k], k, rms[k]
    k = min(rms, key=rms.get)
    return cands[k], k, rms[k]


def refine_polyline_with_cloud(poly, cloud, h, closed=True, iters=1):
    """Verschiebt Polylinienpunkte entlang ihrer Normalen auf den Median der Wolke."""
    poly = np.asarray(poly, float).copy()
    cloud = np.asarray(cloud, float)
    if len(cloud) == 0 or len(poly) < 3:
        return poly
    tree = cKDTree(cloud)
    for _ in range(iters):
        if closed:
            t = np.roll(poly, -1, 0) - np.roll(poly, 1, 0)
        else:
            t = np.gradient(poly, axis=0)
        t /= np.maximum(np.linalg.norm(t, axis=1), 1e-12)[:, None]
        nrm = np.c_[-t[:, 1], t[:, 0]]
        nbrs = tree.query_ball_point(poly, h)
        for i, nb in enumerate(nbrs):
            if len(nb) < 3:
                continue
            d = cloud[nb] - poly[i]
            tt = d @ t[i]
            nn = d @ nrm[i]
            m = np.abs(tt) < 0.6 * h
            if m.sum() >= 3:
                poly[i] += np.median(nn[m]) * nrm[i]
    return poly


def order_loops(loops_2d):
    """Sortiert geschlossene 2D-Konturen: (außen, [innen...]), Restliche außen."""
    items = [(abs(polygon_area_2d(np.asarray(l))), np.asarray(l)) for l in loops_2d]
    items.sort(key=lambda t: -t[0])
    return [l for _, l in items]


def offset_profile(prof, d):
    """Versetzt eine geschlossene Kontur um d (d > 0 vergrößert die umschlossene Fläche)."""
    t = prof["type"]
    if abs(d) < 1e-12:
        return dict(prof)
    if t == "circle":
        return {"type": "circle", "c": list(prof["c"]), "r": float(max(prof["r"] + d, 1e-3))}
    if t == "poly":
        pts = np.asarray(prof["pts"], float)
        fil = list(prof.get("fillets") or [0.0] * len(pts))
        s = 1.0 if polygon_area_2d(pts) > 0 else -1.0
        n = len(pts)
        lines = []
        for i in range(n):
            a, b = pts[i], pts[(i + 1) % n]
            e = b - a
            L = np.linalg.norm(e)
            if L < 1e-12:
                continue
            nrm = s * np.array([e[1], -e[0]]) / L      # nach außen
            lines.append((a + d * nrm, e / L, i))
        out = []
        for j in range(len(lines)):
            p1, d1, _ = lines[j - 1]
            p2, d2, i = lines[j]
            x = intersect_lines_2d(p1, d1, p2, d2)
            out.append(p2 if x is None else x)
        newfil = [max(0.0, r + d) if r > 0 else 0.0 for r in fil]
        out = np.asarray(out)
        return {"type": "poly", "pts": out.tolist(), "fillets": clamp_fillets(out, newfil[:len(out)])}
    dense, closed = profile_to_dense(prof, 0.2)
    if not closed:
        return dict(prof)
    s = 1.0 if polygon_area_2d(dense) > 0 else -1.0
    tng = np.roll(dense, -1, 0) - np.roll(dense, 1, 0)
    tng /= np.maximum(np.linalg.norm(tng, axis=1), 1e-12)[:, None]
    nrm = s * np.c_[tng[:, 1], -tng[:, 0]]
    moved = dense + d * nrm
    L = polyline_length(moved, True)
    pts = resample_polyline(moved, max(L / 200, 0.2), True)
    return {"type": "spline", "pts": pts.tolist(), "closed": True}
