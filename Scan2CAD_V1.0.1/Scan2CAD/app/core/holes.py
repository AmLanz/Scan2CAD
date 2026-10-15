"""Löcher im Netz finden und schließen (nur anhängend: bestehende Indizes bleiben gültig).

Modi
  'flat'   eben:   Kappe in der Ausgleichsebene des Lochrands
  'plane'  Ebene:  wie 'flat', aber in einer vorgegebenen Ebene (plane=(o, n))
  'tight'  straff: Membranfläche (harmonisch, ΔX = 0) – kürzeste Überspannung
  'smooth' glatt:  biharmonisch (Δ²X = 0) mit dem Ring um das Loch als Randbedingung –
                   setzt die Krümmung der Umgebung stetig fort
Neue Dreiecke werden als künstlich markiert und im Füllungsprotokoll vermerkt.
"""
from __future__ import annotations

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve

from .geom_utils import orthonormal_frame, unit
from .mesh_model import MeshModel


def find_holes(model):
    """Randschleifen mit Kennwerten. 'diameter' = flächengleicher Durchmesser der in die
    Ausgleichsebene projizierten Schleife (robust gegen zackige Scan-Ränder, anders als
    Umfang/π)."""
    from .geom_utils import polygon_area_2d
    out = []
    for loop in model.boundary_loops():
        P = model.V[loop]
        per = float(np.linalg.norm(np.diff(np.vstack([P, P[:1]]), axis=0), axis=1).sum())
        X = P - P.mean(0)
        ev, evec = np.linalg.eigh(X.T @ X / max(len(P), 1))
        Q = np.c_[X @ evec[:, 2], X @ evec[:, 1]]
        area = abs(polygon_area_2d(Q))
        d_eq = 2.0 * np.sqrt(area / np.pi) if area > 0 else per / np.pi
        out.append({"loop": loop, "perimeter": per, "diameter": float(d_eq), "area": float(area),
                    "center": P.mean(0), "n": len(loop), "flatness": float(np.sqrt(max(ev[0], 0)))})
    out.sort(key=lambda h: h["perimeter"])
    return out


def _ear_clip(Q):
    n = len(Q)
    idx = list(range(n))
    area = 0.5 * np.sum(Q[:, 0] * np.roll(Q[:, 1], -1) - Q[:, 1] * np.roll(Q[:, 0], -1))
    if area < 0:
        idx = idx[::-1]
    tris = []
    guard = 0
    while len(idx) > 3 and guard < 4 * n * n:
        m = len(idx)
        best = None
        for k in range(m):
            guard += 1
            i0, i1, i2 = idx[k - 1], idx[k], idx[(k + 1) % m]
            a, b, c = Q[i0], Q[i1], Q[i2]
            cr = (b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0])
            if cr <= 1e-14:
                continue
            others = [j for j in idx if j not in (i0, i1, i2)]
            if others:
                R = Q[others]
                d1 = (b[0] - a[0]) * (R[:, 1] - a[1]) - (b[1] - a[1]) * (R[:, 0] - a[0])
                d2 = (c[0] - b[0]) * (R[:, 1] - b[1]) - (c[1] - b[1]) * (R[:, 0] - b[0])
                d3 = (a[0] - c[0]) * (R[:, 1] - c[1]) - (a[1] - c[1]) * (R[:, 0] - c[0])
                inside = (d1 >= -1e-12) & (d2 >= -1e-12) & (d3 >= -1e-12)
                if inside.any():
                    continue
            e = [np.linalg.norm(b - a), np.linalg.norm(c - b), np.linalg.norm(a - c)]
            q = cr / max(e[0] * e[1] * e[2], 1e-300)
            if best is None or q > best[0]:
                best = (q, k)
            if m > 200:
                break
        if best is None:
            return None
        k = best[1]
        tris.append((idx[k - 1], idx[k], idx[(k + 1) % m]))
        idx.pop(k)
    if len(idx) == 3:
        tris.append(tuple(idx))
    return tris


def vertex_angle_sums(model, vids):
    """Summe der Dreieckswinkel an den Punkten vids (nur Dreiecke an diesen Punkten rechnen)."""
    vids = np.asarray(vids, np.int64)
    vm = np.zeros(model.n_verts, bool)
    vm[vids] = True
    F = model.F[vm[model.F].any(1)]
    V = model.V
    sums = np.zeros(model.n_verts)
    for k in range(3):
        a, b, c = V[F[:, k]], V[F[:, (k + 1) % 3]], V[F[:, (k + 2) % 3]]
        u, v = b - a, c - a
        den = np.linalg.norm(u, axis=1) * np.linalg.norm(v, axis=1)
        cs = np.einsum("ij,ij->i", u, v) / np.maximum(den, 1e-300)
        sums += np.bincount(F[:, k], np.arccos(np.clip(cs, -1, 1)), minlength=model.n_verts)
    return sums[vids]


def existing_chords(model, loop):
    """Paare (i, j), i < j, von Schleifenpositionen, die im Netz schon eine Kante bilden, aber
    keine Randkante der Schleife sind."""
    loop = np.asarray(loop, np.int64)
    n = len(loop)
    vm = np.zeros(model.n_verts, bool)
    vm[loop] = True
    Fl = model.F[vm[model.F].all(1)]
    if len(Fl) == 0:
        return set()
    ex = np.sort(np.vstack([Fl[:, [0, 1]], Fl[:, [1, 2]], Fl[:, [2, 0]]]), axis=1)
    pos = -np.ones(model.n_verts, np.int64)
    pos[loop] = np.arange(n)
    out = set()
    for e0, e1 in ex:
        i, j = sorted((int(pos[e0]), int(pos[e1])))
        if j - i not in (1, n - 1):
            out.add((i, j))
    return out


def loop_follows_faces(model, loop):
    """True, wenn die Schleife in Richtung der Halbkanten der angrenzenden Dreiecke läuft."""
    a, b = int(loop[0]), int(loop[1])
    F = model.F
    rows = np.nonzero((F == a).any(1) & (F == b).any(1))[0]
    for r in rows:
        t = list(F[r])
        i = t.index(a)
        return t[(i + 1) % 3] == b
    return True


def front_triangulate(P, hole_ang, forbidden=(), max_chord=None, max_angle=None, normal=None,
                      max_normal_dot=None):
    """Lochrand in 3D triangulieren, indem immer die spitzeste Ecke abgeschnitten wird
    (vereinfachte Advancing-Front-Methode; braucht keine Projektion, kommt also auch mit
    gefalteten, sich in jeder Ebene überlappenden Rändern zurecht).

    hole_ang  Innenwinkel des Lochs an jedem Randpunkt (2π − Winkelsumme der Netzdreiecke)
    forbidden Sehnen (i, j) mit i < j, die schon als Netzkante existieren (sonst nicht-mannigfaltig)
    max_chord, max_angle  nur schneiden, solange Sehne kurz und Winkel spitz (Reißverschluss
              für schmale Schlitze); normal/max_normal_dot: nur Dreiecke, deren Normale nicht zu
              parallel zu 'normal' ist (Schlitze in Wänden, nicht Stege am Boden)
    Rückgabe (Dreiecke als Indexdreier in Schleifenrichtung a→i→b, Restschleife als Indexliste)
    bzw. (None, None), wenn eine vollständige Triangulierung nicht gelingt."""
    import heapq
    P = np.asarray(P, float)
    n = len(P)
    prv = list(range(-1, n - 1))
    prv[0] = n - 1
    nxt = list(range(1, n)) + [0]
    ang = np.asarray(hole_ang, float).copy()
    alive = np.ones(n, bool)
    stamp = [0] * n
    forb = set(forbidden)
    heap = [(float(ang[i]), 0, i) for i in range(n)]
    heapq.heapify(heap)
    tris = []
    count = n
    nrm = None if normal is None else unit(np.asarray(normal, float))

    def corner(at, p, q):
        u, v = P[p] - P[at], P[q] - P[at]
        d = np.linalg.norm(u) * np.linalg.norm(v)
        return float(np.arccos(np.clip(u @ v / max(d, 1e-300), -1, 1)))

    while count > 3 and heap:
        aval, st, i = heapq.heappop(heap)
        if not alive[i] or st != stamp[i]:
            continue
        if max_angle is not None and aval > max_angle:
            break
        a, b = prv[i], nxt[i]
        if max_chord is not None and np.linalg.norm(P[a] - P[b]) > max_chord:
            continue
        if (min(a, b), max(a, b)) in forb:
            continue
        if nrm is not None and max_normal_dot is not None:
            tn = np.cross(P[i] - P[a], P[b] - P[a])
            ln = np.linalg.norm(tn)
            if ln > 1e-300 and abs(float(tn @ nrm)) / ln > max_normal_dot:
                continue
        tris.append((a, i, b))
        alive[i] = False
        count -= 1
        nxt[a] = b
        prv[b] = a
        ang[a] -= corner(a, i, b)
        ang[b] -= corner(b, a, i)
        for v in (a, b):
            stamp[v] += 1
            heapq.heappush(heap, (float(ang[v]), stamp[v], v))
    start = int(np.nonzero(alive)[0][0])
    rest = [start]
    v = nxt[start]
    while v != start and len(rest) <= n:
        rest.append(v)
        v = nxt[v]
    if max_chord is None and max_angle is None:
        if count > 3:
            return None, None
        tris.append(tuple(rest))
        rest = []
    return tris, rest


def _refine(Vn, tris, target, max_rounds=10, max_new=80000):
    """Rot-Grün-Verfeinerung: innere Kanten > target halbieren (Randkanten nie -> keine T-Stöße)."""
    V = [np.asarray(v, float) for v in Vn]
    T = [tuple(t) for t in tris]

    def key(a, b):
        return (a, b) if a < b else (b, a)

    for _ in range(max_rounds):
        cnt = {}
        for t in T:
            for k in range(3):
                e = key(t[k], t[(k + 1) % 3])
                cnt[e] = cnt.get(e, 0) + 1
        mid = {}
        for e, c in cnt.items():
            if c == 2 and np.linalg.norm(V[e[0]] - V[e[1]]) > target:
                mid[e] = len(V)
                V.append(0.5 * (V[e[0]] + V[e[1]]))
        if not mid or len(V) - len(Vn) > max_new:
            break
        out = []
        for (a, b, c) in T:
            m = [mid.get(key(a, b)), mid.get(key(b, c)), mid.get(key(c, a))]
            k = sum(x is not None for x in m)
            if k == 0:
                out.append((a, b, c))
                continue
            vs = [a, b, c]
            for rot in range(3):
                mm = m[rot:] + m[:rot]
                vv = vs[rot:] + vs[:rot]
                if k == 1 and mm[0] is not None:
                    break
                if k == 2 and mm[0] is not None and mm[1] is not None:
                    break
                if k == 3:
                    break
            p, q, r = vv
            mpq, mqr, mrp = mm
            if k == 1:
                out += [(p, mpq, r), (mpq, q, r)]
            elif k == 2:
                out += [(mpq, q, mqr), (p, mpq, mqr), (p, mqr, r)]
            else:
                out += [(p, mpq, mrp), (mpq, q, mqr), (mrp, mqr, r), (mpq, mqr, mrp)]
        T = out
    return np.asarray(V), np.asarray(T, np.int64)


def _laplacian(nV, F):
    e = np.vstack([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
    e = np.unique(np.sort(e, axis=1), axis=0)
    A = sparse.csr_matrix((np.ones(2 * len(e)), (np.r_[e[:, 0], e[:, 1]], np.r_[e[:, 1], e[:, 0]])), shape=(nV, nV))
    A.data[:] = 1.0
    deg = np.asarray(A.sum(1)).ravel()
    return (sparse.diags(deg) - A).tocsr()


def fill_hole(model, loop, mode="smooth", target_edge=None, plane=None, kind="hole", fill_id=None):
    """Schließt ein Loch. Rückgabe: (neues MeshModel, Indizes der neuen Dreiecke)."""
    from .mesh_ops import new_fill_id, triangulate_polygon
    V, F = model.V, model.F
    loop = np.asarray(loop, np.int64)
    P = V[loop]
    n = len(loop)
    c = P.mean(0)
    X = P - c
    ev, evec = np.linalg.eigh(X.T @ X)
    nrm = evec[:, 0]
    if plane is not None:
        po, pn = np.asarray(plane[0], float), unit(plane[1])
        mode = "plane"
    else:
        po, pn = c, nrm
    ax_u, ax_v, _ = orthonormal_frame(pn)
    Q = np.c_[X @ ax_u, X @ ax_v]
    rim_e = np.linalg.norm(np.diff(np.vstack([P, P[:1]]), axis=0), axis=1)
    # Zielkantenlänge: wie der Rand, aber nicht feiner als ~1/40 der Lochgröße (sonst entstehen bei
    # fein aufgelösten Scans für große Löcher Hunderttausende künstlicher Dreiecke)
    size = float(np.sqrt(max(ev[1], 0.0) + max(ev[2], 0.0)) / np.sqrt(max(n, 1))) * 2.0
    target = max(float(target_edge or 1.2 * np.mean(rim_e)), 0.6 * float(rim_e.max()), size / 40.0)
    tris = None
    if n > 60:
        try:
            tris = triangulate_polygon(Q)
            if len(tris) < n - 2 - 2:
                tris = None
        except Exception:
            tris = None
    if tris is None:
        tris = _ear_clip(Q) if n <= 1500 else None
        if tris is not None:
            from .mesh_ops import triangulation_ok
            if not triangulation_ok(np.asarray(tris, np.int64), [n]):
                tris = None
    # Sehnen zwischen Randpunkten, die im Netz schon als Kante existieren (zackige Ränder,
    # Brückendreiecke), würden nicht-mannigfaltige Kanten erzeugen
    forb = existing_chords(model, loop)
    if tris is not None and forb:
        tt = np.asarray(tris, np.int64)
        ch = np.sort(np.vstack([tt[:, [0, 1]], tt[:, [1, 2]], tt[:, [2, 0]]]), axis=1)
        if any((int(a), int(b)) in forb for a, b in ch):
            tris = None
    if tris is None:
        # 3D-Reißverschluss (spitzeste Ecke zuerst) – kommt mit gefalteten Rändern zurecht
        try:
            ftris, _ = front_triangulate(P, 2 * np.pi - vertex_angle_sums(model, loop), forbidden=forb)
        except Exception:
            ftris = None
        tris = ftris
    local_V = list(P)
    if tris is None:  # Fächer um den Schwerpunkt
        local_V.append(c)
        ci = n
        tris = [(ci, i, (i + 1) % n) for i in range(n)]
    tris = np.asarray(tris, np.int64)
    rim_dir = {(i, (i + 1) % n) for i in range(n)}
    same = sum(1 for t in tris for k in range(3) if (t[k], t[(k + 1) % 3]) in rim_dir)
    opp = sum(1 for t in tris for k in range(3) if (t[(k + 1) % 3], t[k]) in rim_dir)
    if same > opp:
        tris = tris[:, ::-1]
    Vl, Fl = _refine(local_V, tris.tolist(), target)
    n_new_v = len(Vl) - n
    gmap = np.r_[loop, len(V) + np.arange(n_new_v)]
    Fg = gmap[Fl]
    V2 = np.vstack([V, Vl[n:]])
    F2 = np.vstack([F, Fg])
    unknown = len(V) + np.arange(n_new_v)
    if n_new_v > 0:
        if mode in ("flat", "plane"):
            r = V2[unknown] - po
            V2[unknown] = V2[unknown] - np.outer(r @ pn, pn)
        else:
            # nur lokal rechnen: neue Dreiecke + alle Dreiecke am Rand (reicht für L und L·L)
            vm = np.zeros(len(V2), bool)
            vm[loop] = True
            ring = np.nonzero(vm[F].any(1))[0]
            vm[np.unique(F[ring])] = True
            ring2 = np.nonzero(vm[F].any(1))[0]
            Floc = np.vstack([F[ring2], Fg])
            L = _laplacian(len(V2), Floc)
            known = np.setdiff1d(np.unique(Floc), unknown)
            if mode == "smooth" and n_new_v <= 30000:
                LU = L[unknown] @ L
                M = LU[:, unknown].tocsc()
                rhs = -(LU[:, known] @ V2[known])
            else:
                M = L[unknown][:, unknown].tocsc()
                rhs = -(L[unknown][:, known] @ V2[known])
            try:
                sol = np.column_stack([spsolve(M, rhs[:, d]) for d in range(3)])
                if np.all(np.isfinite(sol)):
                    V2[unknown] = sol
            except Exception:
                pass
    new_faces = np.arange(len(F), len(F2))
    syn = np.r_[model.synthetic, np.ones(len(new_faces), bool)]
    fills = list(model.fills)
    fills.append({"id": fill_id or new_fill_id(model.fills), "kind": kind, "mode": mode,
                  "faces": new_faces, "rim": [P.copy()],
                  "plane": (po, pn) if mode == "plane" else None, "feature": None, "extend": False})
    return MeshModel(V2, F2, syn, fills), new_faces
