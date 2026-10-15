"""Extrusions- und Rotationskörper aus einer Netzauswahl ableiten."""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from .fitting import fit_rotation_axis
from .geom_utils import (dist_points_to_polyline, douglas_peucker, orthonormal_frame,
                         point_in_polygon, polygon_area_2d, polyline_length,
                         resample_polyline, unit, weighted_mean)
from .mesh_model import slice_mesh
from .profiles import (fit_open_polygon_profile, fit_polygon_profile, fit_profile,
                       fit_spline_profile, profile_residuals,
                       profile_to_dense, refine_polyline_with_cloud)


def _faces_data(model, face_ids):
    return model.centroids[face_ids], model.fit_normals[face_ids], model.weights[face_ids]


# ======================================================================= Extrusion
def chain_pieces(pieces, max_gap_frac=0.95):
    """Verbindet offene Kontur-Teilstücke (Teil-Auswahl) gierig zu einer geschlossenen Kontur.
    Lücken werden geradlinig überbrückt. None, wenn die Lücken zu groß sind."""
    pieces = sorted(pieces, key=lambda q: -polyline_length(q))
    cur = np.asarray(pieces[0], float)
    rest = [np.asarray(p, float) for p in pieces[1:]]
    gaps = 0.0
    while rest:
        e = cur[-1]
        best = None
        for i, p in enumerate(rest):
            for rev in (False, True):
                s = p[-1] if rev else p[0]
                d = np.linalg.norm(s - e)
                if best is None or d < best[0]:
                    best = (d, i, rev)
        d, i, rev = best
        p = rest.pop(i)
        cur = np.vstack([cur, p[::-1] if rev else p])
        gaps += d
    gaps += np.linalg.norm(cur[-1] - cur[0])
    total = polyline_length(cur, True)
    if gaps > max_gap_frac * total:
        return None
    return cur


def extrusion_direction(N, w, fixed_dir=None):
    if fixed_dir is not None:
        return unit(fixed_dir)
    M = (N * w[:, None]).T @ N
    ev, evec = np.linalg.eigh(M)
    return evec[:, 0]


def fit_extrusion(model, face_ids, tol=0.05, profile_kind="auto", fillets="auto", snap=0.0,
                  spline_smooth=0.5, spline_step=2.0, fixed_dir=None):
    face_ids = np.asarray(face_ids)
    C, N, A = _faces_data(model, face_ids)
    d = extrusion_direction(N, A, fixed_dir)
    side = np.abs(N @ d) < 0.5
    for _ in range(2):
        if side.sum() < 6:
            break
        if fixed_dir is None:
            d = extrusion_direction(N[side], A[side])
        side = np.abs(N @ d) < 0.5
    side_ids = face_ids[side]
    if len(side_ids) < 6:
        raise RuntimeError("Zu wenige Seitenflächen für eine Extrusion – Mantelflächen markieren.")
    o = weighted_mean(C[side], A[side])
    x, y, _ = orthonormal_frame(d)
    vid = np.unique(model.F[side_ids])
    Pv = model.V[vid]
    zs = (Pv - o) @ d
    z0, z1 = float(zs.min()), float(zs.max())
    # Profil-Punktwolke nur aus klar senkrechten Mantelflächen und ohne die Deckelkanten
    # (dort liegen oft Brücken-Dreiecke, deren Ecken schon auf dem Deckel liegen)
    strict = side_ids[np.abs(model.fit_normals[side_ids] @ d) < 0.25]
    if len(strict) < 6:
        strict = side_ids
    Pc = model.V[np.unique(model.F[strict])]
    zc = (Pc - o) @ d
    hgt = z1 - z0
    keep = (zc > z0 + 0.03 * hgt) & (zc < z1 - 0.03 * hgt)
    if keep.sum() >= 10:
        Pc = Pc[keep]
    cloud = np.c_[(Pc - o) @ x, (Pc - o) @ y]
    # Schnittpolylinien in mehreren Höhen; die längste Gesamtlänge gewinnt
    Vs, Fs, _ = model.submesh(side_ids)
    best = None
    for frac in (0.5, 0.35, 0.65, 0.2, 0.8):
        zm = z0 + frac * (z1 - z0)
        polys = slice_mesh(Vs, Fs, o + zm * d, d)
        loops2 = []
        for pts, closed in polys:
            q = np.c_[(pts - o) @ x, (pts - o) @ y]
            if polyline_length(q, closed) < 5 * tol:
                continue
            loops2.append((q, closed))
        tot = sum(polyline_length(q, c) for q, c in loops2)
        if best is None or tot > best[0] * 1.05:
            best = (tot, loops2)
    loops2 = best[1] if best else []
    if not loops2:
        raise RuntimeError("Kein Querschnitt gefunden.")
    h = max(3 * tol, 1.5 * model.mean_edge_length())
    refined = []
    for q, closed in loops2:
        step = max(tol, polyline_length(q, closed) / 1500)
        q2 = resample_polyline(q, step, closed)
        q2 = refine_polyline_with_cloud(q2, cloud, h, closed)
        refined.append((q2, closed))
    any_closed = any(c for _, c in refined)
    if any_closed:
        refined = [(q, c) for q, c in refined if c]
    elif len(refined) > 1:
        # nur Teile der Kontur gewählt (z. B. nur die Eckenradien): zu einer Kontur verketten
        ch = chain_pieces([q for q, _ in refined])
        if ch is not None:
            refined = [(resample_polyline(ch, max(tol, polyline_length(ch, True) / 1500), True), True)]
            any_closed = True
    prof_kind = profile_kind
    is_open = False
    if not any_closed:
        if profile_kind in ("circle", "rect"):
            # Teil-Scan: geschlossene Grundform aus der Wolke ergänzen
            q = max(refined, key=lambda t: polyline_length(t[0]))[0]
            refined = [(q, True)]
        elif profile_kind == "auto":
            prof_kind = "auto"
            is_open = True
        else:
            is_open = True
    # Wolke den Schleifen zuordnen
    dense = [resample_polyline(q, max(tol, 0.2), c) for q, c in refined]
    lab_pts = np.vstack(dense)
    lab = np.concatenate([np.full(len(dd), i) for i, dd in enumerate(dense)])
    dist, nearest = cKDTree(lab_pts).query(cloud)
    # Punkte weit weg von jeder Kontur gehören zu anderen Flächen (z. B. mitgemalte Bohrung)
    near = dist < max(10 * tol, 3 * model.mean_edge_length())
    if near.sum() >= 10:
        cloud, nearest, dist = cloud[near], nearest[near], dist[near]
    owner = lab[nearest]
    profiles, kinds, rmss = [], [], []
    for i, (q, closed) in enumerate(refined):
        sub = cloud[owner == i]
        prof, kind, rms = fit_profile(q, sub, prof_kind, tol, fillets, snap, spline_smooth,
                                      spline_step, closed=bool(closed and not is_open))
        profiles.append(prof)
        kinds.append(kind)
        rmss.append(rms)
    if not is_open:
        # Außenkontur zuerst, Innenkonturen nur wenn innerhalb
        areas = [abs(polygon_area_2d(profile_to_dense(p, 0.5)[0])) for p in profiles]
        order = np.argsort(areas)[::-1]
        profiles = [profiles[i] for i in order]
        kinds = [kinds[i] for i in order]
        outer_dense = profile_to_dense(profiles[0], 0.5)[0]
        keep = [0]
        for i in range(1, len(profiles)):
            c = profile_to_dense(profiles[i], 0.5)[0].mean(0)
            if point_in_polygon(c, outer_dense):
                keep.append(i)
        profiles = [profiles[i] for i in keep]
        kinds = [kinds[i] for i in keep]
    # Residuen
    res = np.full(len(cloud), np.nan)
    for i, p in enumerate(profiles):
        m = owner == i
        if m.any():
            res[m] = profile_residuals(p, cloud[m], step=max(tol / 4, 1e-3))
    res = res[np.isfinite(res)]
    # Hohlform? (Normalen zeigen zur Profilmitte -> Tasche/Bohrung)
    inside = False
    if not is_open and len(profiles) == 1:
        cen = profile_to_dense(profiles[0], 0.5)[0].mean(0)
        Cs = C[side]
        Ns = N[side]
        rel = np.c_[(Cs - o) @ x, (Cs - o) @ y] - cen
        n2 = np.c_[Ns @ x, Ns @ y]
        inside = bool(np.sum(A[side] * np.einsum("ij,ij->i", rel, n2)) < 0)
    params = {"origin": o, "dir": d, "xdir": x, "z0": z0, "z1": z1,
              "loops": profiles, "open": is_open, "inside": inside,
              "profile_kinds": kinds}
    return params, res, side_ids


# ======================================================================= Rotation
def fit_revolve(model, face_ids, tol=0.05, profile_mode="auto", smooth=0.4, spline_step=2.0,
                fixed_axis=None, fixed_point=None, snap=0.0):
    face_ids = np.asarray(face_ids)
    C, N, A = _faces_data(model, face_ids)
    if fixed_axis is None:
        q, a, pitch, _ = fit_rotation_axis(C, N, A)
    else:
        a = unit(fixed_axis)
        if fixed_point is not None:
            q = np.asarray(fixed_point, float)
        else:
            # Achslage bei fester Richtung: alle Normalenlinien sollen die Achse schneiden
            # -> (n × a)·(p − q) = 0, lineares Ausgleichsproblem für q
            c0 = weighted_mean(C, A)
            G = np.cross(N, a)
            sw = np.sqrt(A / A.mean())
            qq, *_ = np.linalg.lstsq(G * sw[:, None], np.einsum("ij,ij->i", G, C - c0) * sw, rcond=None)
            q = c0 + qq - (qq @ a) * a
    vid = np.unique(model.F[face_ids])
    Pv = model.V[vid]
    x0, y0, _ = orthonormal_frame(a)
    rel = Pv - q
    zz = rel @ a
    q = q + a * zz.min()  # Achspunkt am unteren Ende
    rel = Pv - q
    zz = rel @ a
    radial = rel - np.outer(zz, a)
    rr = np.linalg.norm(radial, axis=1)
    ang = np.arctan2(radial @ y0, radial @ x0)
    phi = np.arctan2(np.mean(np.sin(ang)), np.mean(np.cos(ang)))
    xdir = np.cos(phi) * x0 + np.sin(phi) * y0
    cloud = np.c_[rr, zz]
    # Schnitt mit der Halbebene durch die Achse
    Vs, Fs, _ = model.submesh(face_ids)
    nrm = unit(np.cross(a, xdir))
    polys = slice_mesh(Vs, Fs, q, nrm)
    pieces = []
    for pts, closed in polys:
        side = (pts - q) @ xdir > 0
        if side.all():
            pieces.append((pts, closed))
            continue
        # an Vorzeichenwechseln aufteilen
        idx = np.nonzero(side)[0]
        if len(idx) < 2:
            continue
        splits = np.split(idx, np.nonzero(np.diff(idx) > 1)[0] + 1)
        for s in splits:
            if len(s) >= 2:
                pieces.append((pts[s], False))
    if not pieces:
        raise RuntimeError("Kein Profilschnitt gefunden – Achse plausibel?")
    pts, closed = max(pieces, key=lambda t: polyline_length(t[0], t[1]))
    rel = pts - q
    pz = rel @ a
    pr = np.linalg.norm(rel - np.outer(pz, a), axis=1)
    prof = np.c_[pr, pz]
    step = max(tol, polyline_length(prof, closed) / 1500)
    prof = resample_polyline(prof, step, closed)
    h = max(3 * tol, 1.5 * model.mean_edge_length())
    prof = refine_polyline_with_cloud(prof, cloud, h, closed)
    prof[:, 0] = np.maximum(prof[:, 0], 0.0)
    # Profil: Geraden+Radien (Konstruktionsabsicht) oder glättender Spline
    sub = cloud if len(cloud) < 4000 else cloud[np.random.default_rng(0).choice(len(cloud), 4000, replace=False)]
    if closed:
        poly = fit_polygon_profile(prof, tol, "auto", snap, cloud=sub)
    else:
        poly = fit_open_polygon_profile(prof, tol, "auto", snap, cloud=sub)
    spl = fit_spline_profile(prof, closed, smooth, tol, spline_step)
    spl["pts"] = np.c_[np.maximum(np.asarray(spl["pts"])[:, 0], 0.0), np.asarray(spl["pts"])[:, 1]].tolist()
    r_spl = float(np.sqrt(np.mean(profile_residuals(spl, sub, step=max(tol / 4, 1e-3)) ** 2)))
    r_poly = np.inf
    if poly is not None:
        pts_ = np.asarray(poly["pts"])
        pts_[:, 0] = np.maximum(pts_[:, 0], 0.0)
        poly["pts"] = pts_.tolist()
        r_poly = float(np.sqrt(np.mean(profile_residuals(poly, sub, step=max(tol / 4, 1e-3)) ** 2)))
    if poly is not None and (profile_mode == "polyline" or
                             (profile_mode == "auto" and r_poly <= max(tol, 1.25 * r_spl))):
        chosen = poly
    else:
        chosen = spl
    res = profile_residuals(chosen, cloud, step=max(tol / 4, 1e-3))
    radial_n = np.einsum("ij,ij->i", N, unit(C - q - np.outer((C - q) @ a, a)))
    inside = bool(np.sum(A * radial_n) < 0)
    params = {"point": q, "axis": a, "xdir": xdir, "profile": chosen, "angle": 2 * np.pi,
              "inside": inside}
    return params, res
