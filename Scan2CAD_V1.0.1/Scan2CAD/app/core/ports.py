"""Bohrungsumgebungen: lokale Drehprofil-Analyse um Bohrungen und runde Vertiefungen.

Um eine Bohrung liegen oft weitere rotationssymmetrische Formelemente – Ringnuten (O-Ring),
Einstiche, Senkungen, Fasen. Einzeln per RANSAC gefittet werden daraus kurze, schiefe
Zylinderstücke, Tori und Schein-Bohrungen (die Nutwände sehen aus wie kurze Bohrungen).

Ablauf je Stelle:
  1. Saat: Bohrungen und kleine Zylinder/Kegel/Tori, Eintrittsebene = angrenzende Ebene,
     unter der die Saat liegt. Achse = Normale dieser Ebene (Bohrungen stehen senkrecht).
  2. Achslage: alle Normalenlinien einer Rotationsfläche schneiden die Achse → lineares
     Ausgleichsproblem mit robuster Gewichtung.
  3. Profil (r, z) der Umgebung → Strecken/Kreise (rotational.ransac_profile); nur Primitive,
     die rundum (≥ ~180°) Messpunkte haben, zählen.
  4. Luft/Material im Profil aus den orientierten Profilpunkten; Schein-Bohrungen (Material
     in ihrem Volumen) fallen weg, echte Bohrungen bekommen Achse und Ø aus dem Profil.
  5. Jede übrige Luftzone unter der Eintrittsebene wird ein abgezogener Rotationskörper
     (Polygon mit Eckenradien), Fasen direkt an der Bohrung werden zur Senkung.
"""
from __future__ import annotations

import math

import numpy as np

from scipy.spatial import cKDTree

from . import rotational as rot
from .geom_utils import orthonormal_frame, polygon_area_2d, unit
from .i18n import tr

DEFAULTS = {
    "max_seed_frac": 0.2,      # Saat-Radius höchstens dieser Anteil der Bauteildiagonale
    "min_seed_r": 0.45,        # kleinere Saaten (Kantenrundungen, Ø0,8-Scheinbohrungen) ignorieren
    "coverage": 0.5,           # Anteil des Umfangs mit Messpunkten
    "min_size": 0.35,          # kleinere Luftzonen sind Scan-Verrundung, keine Formelemente
    "max_tilt_deg": 10.0,
}


# ------------------------------------------------------------------ Hilfsfunktionen
def _faces(f, nF):
    ids = np.asarray(f.face_ids, np.int64)
    return ids[(ids >= 0) & (ids < nF)]


def _plane_out(model, f):
    ids = _faces(f, model.n_faces)
    n = unit(np.asarray(f.params["normal"], float))
    if len(ids) and float((model.normals[ids] @ n).sum()) < 0:
        n = -n
    return n, np.asarray(f.params["origin"], float)


def _seed_info(model, f, diag, o):
    nF = model.n_faces
    ids = _faces(f, nF)
    if len(ids) < 6:
        return None
    p = f.params
    if f.kind == "hole":
        a, pt, r = unit(p["axis"]), np.asarray(p["point"], float), 0.5 * float(p["d"])
    elif f.kind == "cylinder" and f.role == "surface":
        a, pt, r = unit(p["axis"]), np.asarray(p["point"], float), float(p["radius"])
    elif f.kind == "cone" and f.role == "surface":
        a, pt = unit(p["axis"]), np.asarray(p["apex"], float)
        rel = model.centroids[ids] - pt
        r = float(np.median(np.linalg.norm(rel - np.outer(rel @ a, a), axis=1)))
    elif f.kind == "torus" and f.role == "surface":
        a, pt, r = unit(p["axis"]), np.asarray(p["center"], float), float(p["R"])
    else:
        return None
    if not (o["min_seed_r"] <= r <= o["max_seed_frac"] * diag):
        return None
    return {"f": f, "a": a, "p": pt, "r": r, "ids": ids, "area": float(model.areas[ids].sum())}


def _point_in_poly(Q, poly):
    """Gerade-ungerade-Regel, vektorisiert über die Punkte."""
    x, y = Q[:, 0], Q[:, 1]
    inside = np.zeros(len(Q), bool)
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        if y1 == y2:
            continue
        cond = (y1 > y) != (y2 > y)
        xi = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
        inside ^= cond & (x < xi)
    return inside


def _dist_to_polyline(Q, P, closed=True):
    A = P
    B = np.roll(P, -1, axis=0) if closed else P[1:]
    if not closed:
        A = P[:-1]
    best = np.full(len(Q), np.inf)
    for a, b in zip(A, B):
        ab = b - a
        L2 = float(ab @ ab)
        t = np.clip(((Q - a) @ ab) / L2, 0, 1) if L2 > 1e-18 else np.zeros(len(Q))
        d = np.linalg.norm(Q - (a + np.outer(t, ab)), axis=1)
        best = np.minimum(best, d)
    return best


# ------------------------------------------------------------------ Achse
def refine_center(model, faces, x0, a, R, tol, iters=12):
    """Achslage bei fester Richtung a: Normalenlinien sollen die Achse treffen.
    (n × a)·(c − q) = 0 je Dreieck, robust gewichtet (Tukey)."""
    faces = np.asarray(faces, np.int64)
    C = model.centroids[faces]
    N = model.fit_normals[faces]
    w = model.areas[faces]
    e1, e2, _ = orthonormal_frame(a)
    G = np.cross(N, a)
    b0 = np.einsum("ij,ij->i", G, C - x0)
    A = np.c_[G @ e1, G @ e2]
    u = np.zeros(2)
    c_t = max(0.3 * R, 0.5)
    for it in range(iters):
        res = b0 - A @ u
        ww = w * np.where(np.abs(res) < c_t, (1 - (res / c_t) ** 2) ** 2, 0.0)
        if ww.sum() <= 0:
            break
        M = (A * ww[:, None]).T @ A
        if np.linalg.cond(M) > 1e8:
            break
        u_new = np.linalg.solve(M, (A * ww[:, None]).T @ b0)
        done = np.linalg.norm(u_new - u) < 1e-5
        u = u_new
        c_t = max(0.6 * c_t, 6 * tol, 0.08)
        if done and c_t <= max(6 * tol, 0.08) + 1e-9:
            break
    if np.linalg.norm(u) > 0.5 * R + 0.3:
        return np.asarray(x0, float), False
    return x0 + u[0] * e1 + u[1] * e2, True


# ------------------------------------------------------------------ Profil
def _coverage(phi, nbins=36):
    b = np.floor((phi + math.pi) / (2 * math.pi) * nbins).astype(int) % nbins
    return len(np.unique(b)) / float(nbins)


def _prim_inliers(pr, Q, N, eps, cos_a=math.cos(math.radians(20))):
    return rot._inl(pr, Q, N, eps, cos_a)


def _project(pr, Q):
    if pr["type"] == "line":
        p, d = pr["p"], unit(pr["d"])
        return p + np.outer((Q - p) @ d, d)
    c, rho = pr["c"], pr["rho"]
    return c + unit(Q - c) * rho


def _trace_loops(M):
    """Rand einer Pixelmaske als geschlossene Kantenzüge (Inneres links). Jede Kante:
    (Startecke, Endecke, Nachbarzelle (i, j))."""
    H, W = M.shape
    edges = {}
    ii, jj = np.nonzero(M)
    for i, j in zip(ii.tolist(), jj.tolist()):
        # x = j (r), y = i (z); Zelle [j, j+1] × [i, i+1]
        if i == 0 or not M[i - 1, j]:
            edges.setdefault((j, i), []).append(((j + 1, i), (i - 1, j)))
        if j == W - 1 or not M[i, j + 1]:
            edges.setdefault((j + 1, i), []).append(((j + 1, i + 1), (i, j + 1)))
        if i == H - 1 or not M[i + 1, j]:
            edges.setdefault((j + 1, i + 1), []).append(((j, i + 1), (i + 1, j)))
        if j == 0 or not M[i, j - 1]:
            edges.setdefault((j, i + 1), []).append(((j, i), (i, j - 1)))
    loops = []
    while edges:
        start = next(iter(edges))
        cur = start
        loop = []
        prev_dir = None
        guard = 0
        while True:
            outs = edges.get(cur)
            if not outs:
                break
            if len(outs) > 1 and prev_dir is not None:
                # Engstelle: am stärksten nach links abbiegen (Inneres zusammenhalten)
                def turn(e):
                    d = (e[0][0] - cur[0], e[0][1] - cur[1])
                    return -(prev_dir[0] * d[1] - prev_dir[1] * d[0])
                outs.sort(key=turn)
            nxt, nb = outs.pop(0)
            if not outs:
                del edges[cur]
            loop.append((cur, nxt, nb))
            prev_dir = (nxt[0] - cur[0], nxt[1] - cur[1])
            cur = nxt
            guard += 1
            if cur == start or guard > 4 * M.size:
                break
        if len(loop) >= 4:
            loops.append(loop)
    return loops


def _zone_polygon(loop, h, z_lo, nz, nr, gz, in_hole, hole_at, acc_tree, Qacc, Nacc, eps, log, L):
    """Randzug einer Luftzone → Polygon mit Eckenradien. Echte Randstücke folgen den
    Profilpunkten, künstliche (Eintrittsfläche, Bohrung) werden als Werkzeugüberstand
    gerade fortgesetzt. Rückgabe (P, Radien, echte Punkte, Kategorien der Läufe) oder None."""
    from .geom_utils import line_fit_2d
    from .profiles import fit_polygon_profile
    dz_top = max(0.3, 5 * eps)
    dh_ov = max(0.05, 3 * eps)
    d_round = max(0.25, 4 * eps)
    cats, pts = [], []
    rho = 2 * h + eps
    for (c0, c1, nbc) in loop:
        mid = np.array([0.5 * (c0[0] + c1[0]) * h, z_lo + 0.5 * (c0[1] + c1[1]) * h])
        i, j = nbc
        if i < 0 or i >= nz or j < 0 or j >= nr:
            cats.append("bad")
            pts.append(mid)
        elif gz[i] <= 0:
            cats.append("top")
            pts.append(mid)
        elif in_hole[i, j]:
            cats.append(("hole", int(hole_at[i, j]), "v" if c0[1] != c1[1] else "h"))
            pts.append(mid)
        else:
            nb = acc_tree.query_ball_point(mid, rho)
            if not nb:
                cats.append("bad")
                pts.append(mid)
                continue
            Pn, Nn = Qacc[nb], Nacc[nb]
            proj = mid - np.einsum("ij,ij->i", mid - Pn, Nn)[:, None] * Nn
            cats.append("real")
            pts.append(np.median(proj, axis=0))
    pts = np.asarray(pts)
    nbad = sum(1 for c in cats if c == "bad")
    if nbad > 0.15 * len(cats):
        log(f"port dbg: zone {L} unexplained boundary {nbad}/{len(cats)}")
        return None
    key = lambda c: "real" if c in ("real", "bad") else (c if c == "top" else (c[0], c[1], c[2]))
    runs = []
    for k, c in enumerate(cats):
        kc = key(c)
        if runs and runs[-1][0] == kc:
            runs[-1][1].append(k)
        else:
            runs.append([kc, [k]])
    if len(runs) > 1 and runs[0][0] == runs[-1][0]:
        runs[0][1] = runs[-1][1] + runs[0][1]
        runs.pop()
    # sehr kurze Läufe dem Vorgänger zuschlagen
    k = 0
    while len(runs) > 2 and k < len(runs):
        if len(runs[k][1]) < 3:
            runs[k - 1][1] += runs[k][1]
            runs.pop(k)
            k = 0
            continue
        j = (k + 1) % len(runs)
        if j != k and runs[k][0] == runs[j][0]:
            runs[k][1] += runs[j][1]
            runs.pop(j)
            k = 0
            continue
        k += 1
    if len(runs) < 2 or not any(r[0] == "real" for r in runs):
        log(f"port dbg: zone {L} runs {[r[0] for r in runs]}")
        return None

    def art_line(r):
        c, ids = r
        P = pts[ids]
        if c == "top":
            return np.array([0.0, -dz_top]), np.array([1.0, 0.0]), ("z", 0.0)
        if c[2] == "v":
            rv = float(np.median(P[:, 0]))
            return np.array([rv - dh_ov, 0.0]), np.array([0.0, 1.0]), ("r", rv)
        zv = float(np.median(P[:, 1]))
        return np.array([0.0, zv + dh_ov]), np.array([1.0, 0.0]), ("z", zv)

    def isect(p1, d1, p2, d2):
        den = d1[0] * d2[1] - d1[1] * d2[0]
        if abs(den) < 0.2:
            return None
        w = p2 - p1
        t = (w[0] * d2[1] - w[1] * d2[0]) / den
        return p1 + t * d1

    out = []
    real_all = []
    n = len(runs)
    for k in range(n):
        c, ids = runs[k]
        if c != "real":
            prv = runs[k - 1]
            if prv[0] != "real":
                p1, d1, _ = art_line(prv)
                p2, d2, _ = art_line(runs[k])
                X = isect(p1, d1, p2, d2)
                if X is None:
                    return None
                out.append(X)
            continue
        good = [i for i in ids if cats[i] == "real"]
        R = pts[good]
        # Rundungen an künstlichen Rändern abschneiden (Kante wird scharf)
        prv, nxt = runs[k - 1], runs[(k + 1) % n]
        for side, nb_run in (("start", prv), ("end", nxt)):
            if nb_run[0] == "real" or len(R) < 2:
                continue
            _, _, ref = art_line(nb_run)
            dist = np.abs(R[:, 0] - ref[1]) if ref[0] == "r" else np.abs(R[:, 1] - ref[1])
            far = dist > d_round
            if far.sum() < 4:
                continue
            if side == "start":
                i0 = int(np.argmax(far))
                R = R[i0:]
            else:
                i1 = len(far) - int(np.argmax(far[::-1]))
                R = R[:i1]
        if len(R) < 4:
            log(f"port dbg: zone {L} real run too short")
            return None
        real_all.append(R)
        seg = [R]
        for side, nb_run in (("start", prv), ("end", nxt)):
            if nb_run[0] == "real":
                continue
            pa, da, _ = art_line(nb_run)
            # Tangente über ~0,3 mm am Ende
            arc = np.r_[0, np.cumsum(np.linalg.norm(np.diff(R, axis=0), axis=1))]
            if side == "start":
                sel = R[arc <= max(0.3, arc[min(4, len(arc) - 1)])]
                E = R[0]
            else:
                sel = R[arc >= arc[-1] - max(0.3, arc[-1] - arc[max(-5, -len(arc))])]
                E = R[-1]
            cc, dd = line_fit_2d(sel)
            X = isect(E, dd, pa, da)
            if X is None or np.linalg.norm(X - E) > 3 * d_round + dz_top + dh_ov:
                X = pa + ((E - pa) @ da) * da          # Lot auf die künstliche Linie
            if side == "start":
                seg.insert(0, X[None])
            else:
                seg.append(X[None])
        out.extend(np.vstack(seg))
    loop_pts = np.asarray(out)
    if len(loop_pts) < 6:
        return None
    from .geom_utils import resample_polyline
    lp = resample_polyline(loop_pts, max(h, 0.01), closed=True)
    cloud = np.vstack(real_all)
    poly = fit_polygon_profile(lp, tol=eps, fillets="auto", cloud=cloud, max_corners=16)
    if poly is None:
        log(f"port dbg: zone {L} polygon fit failed")
        return None
    P = np.asarray(poly["pts"], float)
    fl = list(poly["fillets"])
    try:
        P, fl = _refine_polygon(P, fl, cloud, eps)
    except Exception as e:  # noqa: BLE001
        log(f"port dbg: refine failed {e}")
    return P, fl, cloud, [r[0] for r in runs]


def _snap_dir(d, fine=3.0, coarse=12.0):
    """Richtung (r, z) auf 0°/90° (bis coarse) bzw. 30°/45°/60° (bis fine) ziehen."""
    ang = math.degrees(math.atan2(d[1], d[0])) % 180.0
    for t, lim in ((0, coarse), (90, coarse), (180, coarse), (30, fine), (45, fine), (60, fine),
                   (120, fine), (135, fine), (150, fine)):
        if abs(ang - t) <= lim:
            a = math.radians(t)
            v = np.array([math.cos(a), math.sin(a)])
            return v if v @ d >= 0 else -v
    return d


def _refine_polygon(P, fl, cloud, eps):
    """Kanten des Profilpolygons an den echten Profilpunkten neu ausgleichen (ohne die
    Eckenrundungen), Richtungen auf Konstruktionswinkel ziehen, Ecken neu schneiden.
    Kanten ohne Messpunkte (Werkzeugüberstand) bleiben wie sie sind."""
    from .geom_utils import intersect_lines_2d, line_fit_2d
    from .profiles import _estimate_fillets
    n = len(P)
    lines, real = [], []
    tree = cKDTree(cloud)
    for j in range(n):
        A, B = P[j], P[(j + 1) % n]
        L = float(np.linalg.norm(B - A))
        d = (B - A) / max(L, 1e-12)
        # Punkte nahe der Kante, ohne die Eckbereiche (Radien)
        cut_a = (fl[j] if j < len(fl) else 0.0) + 0.05
        cut_b = (fl[(j + 1) % n] if (j + 1) % n < len(fl) else 0.0) + 0.05
        mid = 0.5 * (A + B)
        idx = tree.query_ball_point(mid, 0.5 * L + 2 * eps)
        pts = cloud[idx] if idx else np.zeros((0, 2))
        if len(pts):
            t = (pts - A) @ d
            dist = np.abs((pts - A) @ np.array([-d[1], d[0]]))
            pts = pts[(dist < 2 * eps) & (t > cut_a) & (t < L - cut_b)]
        if len(pts) >= 6 and np.ptp((pts - A) @ d) > 0.25 * L:
            c, dd = line_fit_2d(pts)
            dd = _snap_dir(dd if dd @ d >= 0 else -dd)
            c = np.array(c, float)
            # glatte Maße: Ø auf 0,1 mm, Tiefen auf 0,05 mm, wenn innerhalb der Messunsicherheit
            if abs(dd[0]) < 1e-9:
                dia = 2 * c[0]
                ds = round(dia * 10) / 10
                if abs(ds - dia) <= max(0.06, 1.5 * eps):
                    c[0] = 0.5 * ds
            elif abs(dd[1]) < 1e-9:
                zs = round(c[1] * 20) / 20
                if abs(zs - c[1]) <= max(0.03, eps):
                    c[1] = zs
            lines.append((c, dd))
            real.append(True)
        else:
            lines.append((mid, _snap_dir(d) if L > 1e-9 else d))
            real.append(False)
    Q = []
    for j in range(n):
        (c1, d1), (c2, d2) = lines[j - 1], lines[j]
        x = intersect_lines_2d(c1, d1, c2, d2)
        if x is None or np.linalg.norm(x - P[j]) > 0.5:
            x = P[j]
        Q.append(x)
    Q = np.asarray(Q)
    # Radien nur an Ecken zwischen zwei gemessenen Kanten
    fil = _estimate_fillets(Q, cloud, min_r=2 * eps)
    fil = [(round(f * 20) / 20 if abs(round(f * 20) / 20 - f) < 0.03 and f >= 0.1 else f)
           if (real[j - 1] and real[j]) else 0.0 for j, f in enumerate(fil)]
    return Q, fil


# ------------------------------------------------------------------ Hauptfunktion
def analyze_ports(model, features, tol, opts=None, log=None, labels=None, near=None):
    """Rückgabe dict(new=[Feature], remove={ids}, update={id: (params, face_ids)},
    shrink={id: face_ids_to_remove}, report=[str])."""
    from scipy import ndimage
    from .bores import hole_face_ids, snap_diameter
    from .features import Feature
    from .profiles import poly_to_dense
    o = dict(DEFAULTS, **(opts or {}))
    log = log or (lambda s: None)
    nF = model.n_faces
    diag = float(np.linalg.norm(model.bbox[1] - model.bbox[0]))
    C = model.centroids
    tree = cKDTree(C)
    if labels is None:
        labels = np.zeros(nF, np.int64)
        for f in features:
            labels[_faces(f, nF)] = f.id
    fid = {f.id: f for f in features}
    planes = {f.id: f for f in features if f.kind == "plane" and f.role == "surface" and len(f.face_ids) >= 30}
    pinfo = {i: _plane_out(model, f) for i, f in planes.items()}
    ptree = {i: cKDTree(C[_faces(f, nF)]) for i, f in planes.items()}
    eps = max(2.5 * tol, 0.03)
    out = {"new": [], "remove": set(), "update": {}, "shrink": {}, "report": []}

    # ---- 1. Saaten und Eintrittsebenen
    seeds = []
    for f in features:
        s = _seed_info(model, f, diag, o)
        if s is None:
            continue
        if near is not None:
            cpt, rad = near
            if np.linalg.norm(C[s["ids"]] - np.asarray(cpt, float), axis=1).min() > rad:
                continue
        best = None
        cs = np.average(C[s["ids"]], axis=0, weights=model.areas[s["ids"]])
        for pid in planes:
            if pid == f.id:
                continue
            n, po = pinfo[pid]
            al = abs(float(s["a"] @ n))
            surrounded = False
            if al < math.cos(math.radians(30)):
                # schief gefittete Saat (kurze Wand, Delle): zählt, wenn die Ebene sie rundum umgibt
                xc = cs - ((cs - po) @ n) * n
                nbp = ptree[pid].query_ball_point(xc, s["r"] + 2.5)
                if len(nbp) < 20:
                    continue
                Pp = C[_faces(planes[pid], nF)][nbp] - xc
                e1, e2, _ = orthonormal_frame(n)
                if _coverage(np.arctan2(Pp @ e2, Pp @ e1)) < 0.75:
                    continue
                surrounded = True
            h = (C[s["ids"]] - po) @ n
            if np.median(h) >= -2 * tol or np.percentile(h, 95) > 0.3 + 3 * tol:
                continue
            if np.percentile(h, 98) < -max(0.5, 0.25 * s["r"]):
                continue                     # reicht nicht bis an die Ebene
            # Ebene muss um die Stelle herum vorhanden sein
            if surrounded:
                xh = cs - ((cs - po) @ n) * n
            else:
                xh = s["p"] + ((po - s["p"]) @ n) / float(s["a"] @ n) * s["a"]
                if ptree[pid].query(xh)[0] > s["r"] + 3.0:
                    continue
            A = float(model.areas[_faces(planes[pid], nF)].sum())
            key = (0.0 if surrounded else round(al, 2), A)
            if best is None or key > best[0]:
                best = (key, pid, n, po, h, xh)
        if best is None:
            continue
        _, pid, n, po, h, x0 = best
        s["surrounded"] = best[0][0] == 0.0
        s.update(plane=pid, n=n, x0=x0, depth=float(max(0.0, -h.min())))
        seeds.append(s)

    # ---- 2. Gruppen je Ebene und Lage
    groups = []
    for s in sorted(seeds, key=lambda s: -s["area"]):
        for g in groups:
            if g["plane"] != s["plane"]:
                continue
            thr = max(1.0, 0.25 * min(s["r"], g["rmin"]))
            if np.linalg.norm(g["x0"] - s["x0"]) < thr:
                g["seeds"].append(s)
                w = np.array([t["area"] for t in g["seeds"]])
                g["x0"] = np.average([t["x0"] for t in g["seeds"]], axis=0, weights=w)
                g["rmin"] = min(g["rmin"], s["r"])
                break
        else:
            groups.append({"plane": s["plane"], "x0": s["x0"].copy(), "rmin": s["r"], "seeds": [s]})

    used_faces = np.zeros(nF, bool)
    for g in groups:
        try:
            res = _analyze_group(model, g, pinfo, tree, eps, tol, o, fid, used_faces, ndimage,
                                 Feature, hole_face_ids, snap_diameter, poly_to_dense, log)
        except Exception as e:  # noqa: BLE001
            log(f"port: {e}")
            res = None
        if not res:
            continue
        for k in ("new",):
            out[k] += res[k]
        out["remove"] |= res["remove"]
        out["update"].update(res["update"])
        for k, v in res["shrink"].items():
            out["shrink"].setdefault(k, set()).update(v)
        out["report"] += res["report"]
    return out


def _analyze_group(model, g, pinfo, tree, eps, tol, o, fid, used_faces, ndimage, Feature,
                   hole_face_ids, snap_diameter, poly_to_dense, log):
    nF = model.n_faces
    C = model.centroids
    n_out, po = pinfo[g["plane"]]
    a = -n_out                                    # Bohrrichtung (ins Material)
    seeds = g["seeds"]
    R_seed = max(s["r"] for s in seeds)
    D = float(np.clip(max(s["depth"] for s in seeds), 0.5, 4 * R_seed + 2))
    x0 = g["x0"] - ((g["x0"] - po) @ n_out) * n_out
    R_nb = 1.25 * R_seed + 1.5
    ctr = x0 + a * 0.5 * D
    cand = np.asarray(tree.query_ball_point(ctr, math.hypot(R_nb, 0.5 * D + 0.6)), np.int64)
    if len(cand) < 30:
        return None
    rel = C[cand] - x0
    z = rel @ a
    r = np.linalg.norm(rel - np.outer(z, a), axis=1)
    nb = cand[(r < R_nb) & (z > -0.6) & (z < D + 0.6)]
    nb = nb[~used_faces[nb]]
    real = nb[~model.synthetic[nb]]
    if len(real) < 30:
        return None
    # ---- Achslage
    q, ok = refine_center(model, real, x0, a, R_seed, tol)
    # ---- Profil
    rr, zz, N2, W, owner, keep = rot.profile_samples(model, real, q, a, o["max_tilt_deg"])
    if keep.sum() < 30:
        return None
    Q = np.c_[rr, zz][keep]
    N = N2[keep]
    Wk = W[keep]
    own = owner[keep]
    e1, e2, _ = orthonormal_frame(a)
    relc = C[own] - q
    phi = np.arctan2(relc @ e2, relc @ e1)
    min_w = max(2 * math.pi * max(R_seed, 1.0) * 0.12 * 0.5, 1e-3 * float(Wk.sum()))
    prims = rot.ransac_profile(Q, N, Wk, eps, min_w, angle_deg=20.0, max_prims=30, trials=300, seed=0,
                               max_circle=R_nb)
    acc = []
    for pr in prims:
        inl = _prim_inliers(pr, Q, N, eps)
        if inl.sum() < 8:
            continue
        cov = _coverage(phi[inl])
        pr["cov"] = cov
        pr["inl"] = inl
        if cov >= o["coverage"]:
            acc.append(pr)
    log("port dbg: q=%s prims=%s" % (np.round(q, 3).tolist(), [
        (pr["type"], np.round(pr["p"] if pr["type"] == "line" else pr["c"], 3).tolist(),
         np.round(pr["d"], 3).tolist() if pr["type"] == "line" else round(pr["rho"], 3),
         round(pr.get("cov", 0), 2)) for pr in prims]))
    if not acc:
        return None
    # ---- Luft/Material im Profil
    h = max(0.5 * eps, max(R_nb, D + 1.0) / 350.0)
    z_lo, z_hi = -0.3, D + 0.4
    nr, nz = int(math.ceil(R_nb / h)), int(math.ceil((z_hi - z_lo) / h))
    gr = (np.arange(nr) + 0.5) * h
    gz = z_lo + (np.arange(nz) + 0.5) * h
    GR, GZ = np.meshgrid(gr, gz)            # [iz, ir]
    G = np.c_[GR.ravel(), GZ.ravel()]
    SP, SN, SW = [], [], []
    for pr in acc:
        inl = pr["inl"]
        P = _project(pr, Q[inl])
        nn = rot.prim_normal(pr, P)
        sgn = np.sign(np.einsum("ij,ij->i", nn, N[inl]))
        sgn[sgn == 0] = 1
        SP.append(P)
        SN.append(nn * sgn[:, None])
        SW.append(Wk[inl])
    SP, SN = np.vstack(SP), np.vstack(SN)
    st = cKDTree(SP)
    kk = min(6, len(SP))
    dist, idx = st.query(G, k=kk)
    if kk == 1:
        dist, idx = dist[:, None], idx[:, None]
    s = np.einsum("ikj,ikj->ik", G[:, None, :] - SP[idx], SN[idx])
    wgt = 1.0 / (dist + h)
    air = ((np.sign(s) * wgt).sum(1) > 0).reshape(nz, nr)
    # Gewicht der Profilpunkte im Inneren eines Volumens (Belege für Material)
    allP, allN = SP, SN
    allW = np.concatenate(SW)

    report = []
    res = {"new": [], "remove": set(), "update": {}, "shrink": {}, "report": report}
    # ---- Bohrungen prüfen
    holes = [s["f"] for s in seeds if s["f"].kind == "hole"]
    real_holes = []
    for hf in holes:
        p = hf.params
        rh = 0.5 * float(p["d"])
        dh = float(p.get("seen_depth") or p["depth"]) if p.get("bottom") != "through" else float(p["depth"])
        dh = min(dh, D)
        # Belege für einen Materialkern im Bohrungsvolumen: Flächen, deren Normale von der Achse
        # weg zeigt (Innenwand einer Ringnut, Zapfen). Böden und Absätze zeigen nach oben/innen.
        inside = ((allP[:, 0] < rh - 3 * eps) & (allP[:, 1] > 3 * eps) & (allP[:, 1] < dh - 3 * eps)
                  & (allN[:, 0] > 0.5))
        ev = float(allW[inside].sum())
        A_wall = 2 * math.pi * rh * max(dh, 0.2)
        if ev > 0.1 * A_wall:
            res["remove"].add(hf.id)
            report.append(tr(f"{hf.name}: Schein-Bohrung (Wand eines Drehprofils) entfernt",
                             f"{hf.name}: spurious hole (wall of a turned profile) removed"))
            continue
        real_holes.append(hf)
    # doppelte echte Bohrungen: größte (tiefste) behalten
    real_holes.sort(key=lambda f: -float(f.params["depth"]) * float(f.params["d"]))
    kept = []
    for hf in real_holes:
        if any(abs(float(hf.params["d"]) - float(k.params["d"])) < 0.3 for k in kept):
            res["remove"].add(hf.id)
            report.append(tr(f"{hf.name}: doppelt – entfernt", f"{hf.name}: duplicate – removed"))
            continue
        kept.append(hf)
    hole_regions = []
    for hf in kept:
        p = dict(hf.params)
        rh = 0.5 * float(p["d"])
        # Ø aus senkrechter Profilstrecke nahe rh
        best = None
        for pr in acc:
            if pr["type"] != "line":
                continue
            d = unit(pr["d"])
            if abs(d[1]) < math.cos(math.radians(1.0)):
                continue
            rp = abs(float(pr["p"][0]))
            if abs(rp - rh) < max(0.15, 0.05 * rh) and (best is None or pr["weight"] > best[1]):
                best = (rp, pr["weight"])
        old_axis = unit(p["axis"])
        tilt = math.degrees(math.acos(min(1.0, abs(float(old_axis @ a)))))
        if tilt > 3.0 and best is None:
            # neue Achse nicht durch eine rundum erfasste Bohrwand bestätigt → Bohrung unverändert
            log(f"port dbg: {hf.name} tilt {tilt:.1f} not verified")
            continue
        p["axis"] = a.copy()
        p["point"] = q.copy()
        if best is not None:
            sn = snap_diameter(2 * best[0], meas=min(tol, 0.02))
            d_new = sn[0] if sn is not None else round(2 * best[0], 2)
            if abs(d_new - p["d"]) > 1e-9:
                report.append(tr(f"{hf.name}: Ø {p['d']:g} → {d_new:g} (aus dem Drehprofil)",
                                 f"{hf.name}: Ø {p['d']:g} → {d_new:g} (from turned profile)"))
                p["d"] = d_new
        if tilt > 1.0:
            report.append(tr(f"{hf.name}: Achse um {tilt:.1f}° auf die Flächennormale korrigiert",
                             f"{hf.name}: axis corrected by {tilt:.1f}° to the face normal"))
        hole_regions.append((hf, p))
    # ---- Luftzonen außerhalb der Bohrungen
    in_hole = np.zeros_like(air)
    hole_at = -np.ones(air.shape, np.int64)
    for k, (hf, p) in enumerate(hole_regions):
        rh = 0.5 * float(p["d"])
        if p.get("bottom") == "through":
            reg = GR < rh
        else:
            dep = float(p["depth"])
            if p.get("bottom") == "drill":
                ta = math.tan(math.radians(float(p.get("point_angle", 118.0)) / 2))
                reg = (GR < rh) & (GZ < dep + (rh - GR) / max(ta, 1e-6))
            else:
                reg = (GR < rh) & (GZ < dep)
        hd = p.get("head", "none")
        if hd == "cbore":
            reg |= (GR < 0.5 * float(p["head_d"])) & (GZ < float(p["head_depth"]))
        elif hd == "csk":
            ta = math.tan(math.radians(float(p.get("head_angle", 90.0)) / 2))
            reg |= GR < 0.5 * float(p["head_d"]) - GZ * ta
        in_hole |= reg
        hole_at[reg & (hole_at < 0)] = k
    below = GZ > 0
    zone = air & below & ~in_hole
    lab, nlab = ndimage.label(zone)
    min_size = o["min_size"]
    edge_len = model.mean_edge_length()
    claimed = np.zeros(nF, bool)
    rel_nb = C[nb] - q
    z_nb = rel_nb @ a
    r_nb = np.linalg.norm(rel_nb - np.outer(z_nb, a), axis=1)
    Qnb = np.c_[r_nb, z_nb]
    n_made = 0
    Qacc = np.vstack([Q[pr["inl"]] for pr in acc])
    Nacc = np.vstack([N[pr["inl"]] for pr in acc])
    acc_tree = cKDTree(Qacc)
    for L in range(1, nlab + 1):
        M = lab == L
        cells = int(M.sum())
        iz, ir = np.nonzero(M)
        ext_r = (ir.max() - ir.min() + 1) * h
        ext_z = (iz.max() - iz.min() + 1) * h
        if cells * h * h < min_size ** 2 or ext_r < min_size or ext_z < min_size:
            log(f"port dbg: zone {L} too small {ext_r:.2f}x{ext_z:.2f}")
            continue
        if ir.max() >= nr - 1 or iz.max() >= nz - 1:
            log(f"port dbg: zone {L} open (r {ir.max() * h:.2f}, z {gz[iz.max()]:.2f})")
            continue                                  # nicht geschlossen im Ausschnitt
        top_row = int(np.searchsorted(gz, 0.0))       # erste Zeile unter der Fläche
        touches_top = bool((iz == top_row).any())
        nbr_hole = ndimage.binary_dilation(M) & in_hole
        touches_hole = bool(nbr_hole.any())
        if not touches_top and not touches_hole:
            log(f"port dbg: zone {L} enclosed")
            continue
        loops = _trace_loops(M)
        if len(loops) != 1:
            log(f"port dbg: zone {L} loops {len(loops)}")
            continue
        loop = loops[0]
        zp = _zone_polygon(loop, h, z_lo, nz, nr, gz, in_hole, hole_at, acc_tree, Qacc, Nacc, eps, log, L)
        if zp is None:
            continue
        P, fl, real_pts, cats = zp
        P[:, 0] = np.maximum(P[:, 0], 0.0)
        A_poly = abs(polygon_area_2d(P))
        A_cells = cells * h * h
        dz_top = max(0.3, 5 * eps)
        A_top = dz_top * (ext_r + 0.2) if touches_top else 0.0
        if not (0.5 * A_cells < A_poly < 1.6 * A_cells + A_top + 4 * h * (ext_r + ext_z)):
            log(f"port dbg: zone {L} area {A_poly:.3f} vs {A_cells:.3f}")
            continue
        from .mesh_ops import polygon_self_intersections
        if polygon_self_intersections([P]):
            log(f"port dbg: zone {L} self-intersecting")
            continue
        dense = poly_to_dense(P.tolist(), fl, step=max(h, 0.02))
        # reine Fase an der Bohrung → Senkung der Bohrung
        n_real = sum(1 for c in cats if c == "real")
        if touches_top and touches_hole and n_real == 1 and len(hole_regions):
            from .geom_utils import line_fit_2d
            c0, d2 = line_fit_2d(real_pts)
            nrm = np.array([-d2[1], d2[0]])
            rms = float(np.sqrt(np.mean(((real_pts - c0) @ nrm) ** 2)))
            ang = math.degrees(math.atan2(abs(d2[0]), abs(d2[1])))   # zur Achse
            hk = int(np.bincount([int(c[1]) for c in cats if isinstance(c, tuple)] or [0]).argmax())
            hf, p = hole_regions[min(hk, len(hole_regions) - 1)]
            if rms < eps and 15 < ang < 75 and p.get("head", "none") == "none" and abs(d2[1]) > 1e-9:
                r_top = float(c0[0] + (0.0 - c0[1]) * d2[0] / d2[1])
                p["head"] = "csk"
                p["head_angle"] = float(round(2 * ang))
                p["head_d"] = float(round(2 * r_top, 2))
                report.append(tr(f"{hf.name}: Fase/Senkung Ø{2 * r_top:.2f} × {2 * ang:.0f}° erkannt",
                                 f"{hf.name}: chamfer/countersink Ø{2 * r_top:.2f} × {2 * ang:.0f}° detected"))
                inside = _point_in_poly(Qnb, dense) & (Qnb[:, 1] > -eps)
                near = _dist_to_polyline(Qnb, dense) < max(2 * eps, 0.5 * edge_len)
                claimed[nb[inside | (near & (Qnb[:, 1] > eps))]] = True
                continue
        # Drehkontur als abgezogener Rotationskörper
        kind_name = (tr("Ringnut", "Ring groove") if touches_top and not touches_hole else
                     tr("Einstich", "Undercut") if touches_hole and not touches_top else
                     tr("Drehkontur", "Turned contour"))
        params = {"point": q.copy(), "axis": a.copy(), "xdir": e1.copy(), "angle": 2 * math.pi,
                  "profile": {"type": "poly", "pts": P.tolist(), "fillets": [float(x) for x in fl], "closed": True},
                  "port": True}
        inside = _point_in_poly(Qnb, dense) & (Qnb[:, 1] > -eps)
        near_all = _dist_to_polyline(Qnb, dense) < max(2 * eps, 0.5 * edge_len)
        ids = nb[inside | (near_all & (Qnb[:, 1] > eps))]
        f = Feature("revolve", params, "sub", ids, auto=True)
        f.name = f"{kind_name} {f.id}"
        try:
            f.shape()
        except Exception as e:  # noqa: BLE001
            log(f"port: revolve failed {e}")
            continue
        finally:
            f.invalidate()
        res["new"].append(f)
        claimed[ids] = True
        n_made += 1
        dims = (f"Ø{2 * float(P[:, 0].min()):.2f}…Ø{2 * float(P[:, 0].max()):.2f}, "
                + tr("Tiefe", "depth") + f" {float(P[:, 1].max()):.2f}")
        fr = [x for x in fl if x > 0]
        report.append(f"{f.name}: {dims}" + ((tr(", Radien ", ", radii ") + ", ".join(f"{x:.2f}" for x in fr)) if fr else ""))
    # ---- Kantenverrundungen an den Öffnungen (Scan) dem Element zuschlagen: beim Verstopfen
    # wird so die ganze Öffnung bis an die ebene Fläche entfernt und eben gedeckelt
    w_round = max(0.5, 6 * eps)
    band = (Qnb[:, 1] > -3 * eps) & (Qnb[:, 1] < w_round)
    for f in res["new"]:
        Pp = np.asarray(f.params["profile"]["pts"], float)
        top = Pp[Pp[:, 1] < 0]
        if len(top) < 2:
            continue
        r1, r2 = float(top[:, 0].min()), float(top[:, 0].max())
        sel = band & (Qnb[:, 0] > r1 - w_round) & (Qnb[:, 0] < r2 + w_round) & ~claimed[nb]
        if sel.any():
            f.face_ids = np.union1d(f.face_ids, nb[sel])
            claimed[nb[sel]] = True
    # ---- Bohrungen aktualisieren
    for hf, p in hole_regions:
        ids = hole_face_ids(model, p, tol=tol)
        ids = ids[~claimed[ids]] if len(ids) else ids
        if len(ids) < 0.3 * len(_faces(hf, nF)):
            ids = np.union1d(ids, _faces(hf, nF)[~claimed[_faces(hf, nF)]])
        rh = 0.5 * float(p["d"])
        if p.get("head") == "csk":
            rh = max(rh, 0.5 * float(p["head_d"]))
        elif p.get("head") == "cbore":
            rh = max(rh, 0.5 * float(p["head_d"]))
        sel = band & (Qnb[:, 0] < rh + w_round) & ~claimed[nb]
        ids = np.union1d(ids, nb[sel])
        res["update"][hf.id] = (p, ids)
        claimed[ids] = True
    if not n_made and not res["remove"] and not hole_regions:
        # nichts Drehsymmetrisches gefunden: schief gefittete Saaten in einer Ebene (Dellen,
        # Fehlstellen) sind Fehlfits – entfernen, die Lücke schließt später eben
        bad = [s_["f"] for s_ in seeds if s_.get("surrounded") and s_["f"].kind != "hole"]
        if not bad:
            return None
        for bf in bad:
            res["remove"].add(bf.id)
            report.append(tr(f"{bf.name}: schief gefittete Delle in {fid[g['plane']].name} – entfernt",
                             f"{bf.name}: misfitted dent in {fid[g['plane']].name} – removed"))
        return res
    # ---- verbrauchte Flächen-Features
    seed_ids = {s["f"].id for s in seeds}
    for f2 in fid.values():
        if f2.role != "surface" or f2.kind == "freeform":
            continue
        ids = _faces(f2, nF)
        if len(ids) == 0:
            continue
        frac = float(claimed[ids].mean())
        if frac == 0.0:
            continue
        if frac >= 0.6 or (f2.id in seed_ids and frac >= 0.3):
            res["remove"].add(f2.id)
        else:
            res["shrink"].setdefault(f2.id, set()).update(ids[claimed[ids]].tolist())
    used_faces |= claimed
    gone = [fid[i].name for i in res["remove"] if i in fid and fid[i].kind != "hole"]
    if gone:
        report.append(tr(f"in der Bohrungsumgebung aufgegangen: {', '.join(sorted(gone))}",
                         f"absorbed into the hole surroundings: {', '.join(sorted(gone))}"))
    if n_made or res["remove"]:
        report.insert(0, tr(f"Stelle bei {np.round(q, 2).tolist()}: Achse ⟂ {fid[g['plane']].name}",
                            f"Site at {np.round(q, 2).tolist()}: axis ⟂ {fid[g['plane']].name}"))
    return res


def apply_ports(doc, res):
    """Ergebnis von analyze_ports auf das Dokument anwenden."""
    nF = doc.model.n_faces
    keep = []
    for f in doc.features:
        if f.id in res["remove"]:
            continue
        if f.id in res["update"]:
            p, ids = res["update"][f.id]
            f.params = p
            f.face_ids = np.asarray(ids, np.int64)
            f.invalidate()
        elif f.id in res["shrink"]:
            ids = _faces(f, nF)
            f.face_ids = np.setdiff1d(ids, np.fromiter(res["shrink"][f.id], np.int64))
        keep.append(f)
    doc.features = keep + list(res["new"])
