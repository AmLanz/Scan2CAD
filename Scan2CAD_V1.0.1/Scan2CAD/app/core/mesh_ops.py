"""Netz-Werkzeuge für typische Scan-Probleme.

Alle Funktionen geben ein NEUES MeshModel zurück (Rückgängig funktioniert über die Historie).
Wo Dreiecke entfernt werden, wird zusätzlich `face_map` (alt -> neu, -1 = entfernt) geliefert,
damit Feature-Zuordnungen nachgezogen werden können.

Füllungsprotokoll (MeshModel.fills): jede Füllung ist ein dict
    id, kind ('hole' | 'base' | 'replace'), mode, faces (Dreiecks-Indizes),
    rim (Liste Nx3-Randpunkte zum Wiederfinden), plane ((o, n) oder None),
    feature (id eines festgelegten Ebenen-Features oder None), extend (bool)
"""
from __future__ import annotations

import itertools

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.spatial import cKDTree

from .geom_utils import orthonormal_frame, pca_frame, polygon_area_2d, unit
from .i18n import tr
from .mesh_model import MeshModel, boundary_loops_of

_FILL_IDS = itertools.count(1)


def new_fill_id(fills=()):
    m = max([f.get("id", 0) for f in fills], default=0)
    while True:
        i = next(_FILL_IDS)
        if i > m:
            return i


# ======================================================================== Grundoperationen
def compact(model, keep_mask):
    """Dreiecke mit keep_mask=False entfernen, unbenutzte Punkte löschen."""
    keep_mask = np.asarray(keep_mask, bool)
    F = model.F[keep_mask]
    used = np.unique(F)
    vmap = -np.ones(model.n_verts, np.int64)
    vmap[used] = np.arange(len(used))
    face_map = -np.ones(model.n_faces, np.int64)
    face_map[keep_mask] = np.arange(int(keep_mask.sum()))
    fills = []
    for fl in model.fills:
        g = dict(fl)
        ids = remap_ids(fl.get("faces", []), face_map)
        if len(ids) == 0:
            continue
        g["faces"] = ids
        fills.append(g)
    return MeshModel(model.V[used], vmap[F], model.synthetic[keep_mask], fills), face_map


def remap_ids(ids, face_map):
    ids = np.asarray(ids, np.int64)
    ids = ids[(ids >= 0) & (ids < len(face_map))]
    out = face_map[ids]
    return out[out >= 0]


def append_geometry(model, V_new, F_new, synthetic=True, fill=None):
    """Punkte/Dreiecke anhängen (F_new in globalen Indizes, neue Punkte ab model.n_verts)."""
    V2 = np.vstack([model.V, np.asarray(V_new, float).reshape(-1, 3)])
    F_new = np.asarray(F_new, np.int64).reshape(-1, 3)
    F2 = np.vstack([model.F, F_new])
    syn = np.r_[model.synthetic, np.full(len(F_new), bool(synthetic))]
    new_ids = np.arange(model.n_faces, len(F2))
    fills = list(model.fills)
    if fill is not None:
        fill = dict(fill)
        fill["faces"] = np.r_[np.asarray(fill.get("faces", []), np.int64), new_ids]
        fills.append(fill)
    return MeshModel(V2, F2, syn, fills), new_ids


def transfer_face_labels(old, new, labels, max_dist=None):
    """Label je Dreieck vom alten auf ein neues Netz übertragen (nächster Schwerpunkt)."""
    d, j = old.centroid_tree.query(new.centroids)
    out = np.asarray(labels)[j]
    if max_dist is not None:
        out = np.where(d <= max_dist, out, -1 if np.issubdtype(np.asarray(labels).dtype, np.integer) else False)
    return out


def transfer_model_state(old, new):
    """Künstlich-Maske und Füllungen auf ein umgebautes Netz (Glätten/Vereinfachen) übertragen."""
    syn = transfer_face_labels(old, new, old.synthetic)
    owner = -np.ones(old.n_faces, np.int64)
    for i, fl in enumerate(old.fills):
        owner[np.asarray(fl["faces"], np.int64)] = i
    own_new = transfer_face_labels(old, new, owner)
    fills = []
    for i, fl in enumerate(old.fills):
        g = dict(fl)
        g["faces"] = np.nonzero(own_new == i)[0]
        if len(g["faces"]):
            fills.append(g)
    return MeshModel(new.V, new.F, syn, fills)


def vertex_adjacency(model):
    F = model.F
    e = np.vstack([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
    nV = model.n_verts
    A = sparse.csr_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(nV, nV))
    A = ((A + A.T) > 0).astype(np.float64)
    return A.tocsr()


def uniform_laplacian(nV, F):
    e = np.vstack([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
    A = sparse.csr_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(nV, nV))
    A = ((A + A.T) > 0).astype(np.float64)
    deg = np.asarray(A.sum(1)).ravel()
    return (sparse.diags(deg) - A).tocsr()


def boundary_vertex_mask(model):
    loops = model.boundary_loops()
    m = np.zeros(model.n_verts, bool)
    for l in loops:
        m[l] = True
    return m


# ======================================================================== Rauschen
def estimate_noise(model, n_samples=500, k=36, seed=0):
    """Scanrauschen σ aus lokalen Quadrik-Fits (robust gegen Krümmung und Kanten).

    In zufälligen Nachbarschaften (k Punkte) wird z = ax²+bxy+cy²+dx+ey+f gefittet; die
    Residuen-RMS ist Rauschen + Restkrümmung. Kanten/Ecken erhöhen sie stark, daher das
    35. Perzentil statt Mittelwert. Rückgabe: σ und daraus abgeleitete Toleranzvorschläge.
    """
    rng = np.random.default_rng(seed)
    V = model.V
    real_v = np.ones(model.n_verts, bool)
    if model.synthetic.any():
        real_v[:] = False
        real_v[np.unique(model.F[~model.synthetic])] = True
    cand = np.nonzero(real_v)[0]
    tree = cKDTree(V[cand])
    picks = rng.choice(len(cand), size=min(n_samples, len(cand)), replace=False)
    rms = []
    for i in picks:
        _, nb = tree.query(V[cand[i]], k=k)
        Q = V[cand[nb]]
        c, ev, evec = pca_frame(Q)
        X = (Q - c) @ evec
        x, y, z = X[:, 0], X[:, 1], X[:, 2]
        A = np.c_[x * x, x * y, y * y, x, y, np.ones_like(x)]
        sol, *_ = np.linalg.lstsq(A, z, rcond=None)
        r = z - A @ sol
        rms.append(np.sqrt((r @ r) / max(len(r) - 6, 1)))
    rms = np.asarray(rms)
    sigma = float(np.percentile(rms, 35))
    edge = model.mean_edge_length()
    sigma = max(sigma, 1e-4)
    return {"sigma": sigma, "edge": edge,
            "fit_tol": round(max(2.5 * sigma, 0.01), 4),
            "detect_eps": round(max(4.0 * sigma, 0.02), 4),
            "simplify_tol": round(max(3.0 * sigma, 0.01), 4),
            "p90": float(np.percentile(rms, 90))}


# ======================================================================== Mannigfaltigkeit
def manifold_report(model):
    F = model.F
    nV = model.n_verts
    e = np.sort(np.vstack([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]), axis=1)
    key = e[:, 0] * nV + e[:, 1]
    _, cnt = np.unique(key, return_counts=True)
    # Orientierung: gerichtete Halbkanten doppelt -> inkonsistent
    h = np.vstack([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
    hk = h[:, 0] * nV + h[:, 1]
    _, hc = np.unique(hk, return_counts=True)
    from scipy.sparse.csgraph import connected_components
    ncomp, _ = connected_components(model.adjacency, directed=False)
    return {"faces": model.n_faces, "nonmanifold_edges": int((cnt > 2).sum()),
            "boundary_edges": int((cnt == 1).sum()), "flipped_edges": int((hc > 1).sum()),
            "components": int(ncomp), "closed": bool((cnt == 1).sum() == 0 and (cnt > 2).sum() == 0)}


def repair_manifold(model):
    """Nicht-mannigfaltige Kanten auftrennen (überzählige Dreiecke entfernen), doppelte
    Dreiecke löschen, Orientierung je Zusammenhangskomponente vereinheitlichen
    (geschlossene Teile nach außen). Rückgabe: (Netz, Bericht, face_map)."""
    F = model.F.copy()
    nF = len(F)
    nV = model.n_verts
    report = []
    keep = np.ones(nF, bool)
    # doppelte Dreiecke: gleich orientiert -> eines behalten; gegensinnig (Nullvolumen-"Flosse",
    # typisch nach Dezimierung) -> beide entfernen
    sk = np.sort(F, axis=1)
    key3 = sk[:, 0] * nV * nV + sk[:, 1] * nV + sk[:, 2]
    order3 = np.argsort(key3, kind="stable")
    ks3 = key3[order3]
    st3 = np.r_[0, np.nonzero(np.diff(ks3))[0] + 1]
    cn3 = np.diff(np.r_[st3, len(ks3)])
    # Parität der zyklischen Reihenfolge relativ zur sortierten
    rot = np.argmin(F, axis=1)
    Fr = np.take_along_axis(F, np.c_[rot, (rot + 1) % 3, (rot + 2) % 3], axis=1)
    par = Fr[:, 1] < Fr[:, 2]
    dup = np.zeros(nF, bool)
    for b in np.nonzero(cn3 > 1)[0]:
        grp = order3[st3[b]:st3[b] + cn3[b]]
        pos = list(grp[par[grp]])
        neg = list(grp[~par[grp]])
        k = min(len(pos), len(neg))
        drop = pos[:k] + neg[:k]
        rest = pos[k:] + neg[k:]
        drop += rest[1:]
        dup[drop] = True
    if dup.any():
        report.append(tr(f"{int(dup.sum())} doppelte Dreiecke entfernt", f"{int(dup.sum())} duplicate triangles removed"))
    keep &= ~dup
    # degenerierte
    deg = (model.areas < 1e-12) | (F[:, 0] == F[:, 1]) | (F[:, 1] == F[:, 2]) | (F[:, 0] == F[:, 2])
    keep &= ~deg
    # nicht-mannigfaltige Kanten: nur die zwei größten Dreiecke behalten
    for _ in range(3):
        idx = np.nonzero(keep)[0]
        Fk = F[idx]
        e = np.sort(np.vstack([Fk[:, [0, 1]], Fk[:, [1, 2]], Fk[:, [2, 0]]]), axis=1)
        fi = np.r_[idx, idx, idx]
        key = e[:, 0] * nV + e[:, 1]
        order = np.argsort(key, kind="stable")
        ks = key[order]
        starts = np.r_[0, np.nonzero(np.diff(ks))[0] + 1]
        cnts = np.diff(np.r_[starts, len(ks)])
        bad = np.nonzero(cnts > 2)[0]
        if len(bad) == 0:
            break
        removed = 0
        for b in bad:
            faces = fi[order[starts[b]:starts[b] + cnts[b]]]
            faces = faces[np.argsort(-model.areas[faces])]
            keep[faces[2:]] = False
            removed += len(faces) - 2
        report.append(tr(f"{len(bad)} nicht-mannigfaltige Kanten aufgetrennt ({removed} Dreiecke entfernt)",
                         f"{len(bad)} non-manifold edges split ({removed} triangles removed)"))
    m2, face_map = compact(model, keep)
    F = m2.F.copy()
    # Orientierung per BFS über Kanten
    nF = len(F)
    nV = m2.n_verts
    h = np.vstack([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
    fi = np.r_[np.arange(nF), np.arange(nF), np.arange(nF)]
    es = np.sort(h, axis=1)
    key = es[:, 0] * nV + es[:, 1]
    order = np.argsort(key, kind="stable")
    ks = key[order]
    same = np.nonzero(ks[1:] == ks[:-1])[0]
    i0, i1 = order[same], order[same + 1]
    f0, f1 = fi[i0], fi[i1]
    consistent = h[i0, 0] != h[i1, 0]           # gleiche Richtung -> inkonsistent
    nbrs = [[] for _ in range(nF)]
    for a, b, c in zip(f0, f1, consistent):
        nbrs[a].append((b, c))
        nbrs[b].append((a, c))
    flip = np.zeros(nF, bool)
    seen = np.zeros(nF, bool)
    comp = -np.ones(nF, np.int64)
    ncomp = 0
    for s in range(nF):
        if seen[s]:
            continue
        stack = [s]
        seen[s] = True
        comp[s] = ncomp
        while stack:
            a = stack.pop()
            for b, c in nbrs[a]:
                if not seen[b]:
                    seen[b] = True
                    comp[b] = ncomp
                    flip[b] = flip[a] if c else not flip[a]
                    stack.append(b)
        ncomp += 1
    nflip = 0
    if flip.any():
        F[flip] = F[flip][:, ::-1]
    # geschlossene Komponenten nach außen (Volumen positiv), offene: so wenig wie möglich drehen
    tri = m2.V[F]
    vol6 = np.einsum("ij,ij->i", tri[:, 0], np.cross(tri[:, 1], tri[:, 2]))
    tmp = MeshModel(m2.V, F)
    _, open_cnt = np.unique(np.sort(np.vstack([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]), axis=1),
                            axis=0, return_counts=True)
    bnd_faces = np.zeros(nF, bool)
    if tmp.boundary_edge_count:
        for l in tmp.boundary_loops():
            vm = np.zeros(nV, bool)
            vm[l] = True
            bnd_faces |= vm[F].any(1)
    for c in range(ncomp):
        mm = comp == c
        is_open = bnd_faces[mm].any()
        if (not is_open and vol6[mm].sum() < 0) or (is_open and flip[mm].sum() > 0.5 * mm.sum()):
            F[mm] = F[mm][:, ::-1]
            flip[mm] = ~flip[mm]
    nflip = int(flip.sum())
    if nflip:
        report.append(tr(f"{nflip} Dreiecke umorientiert", f"{nflip} triangles re-oriented"))
    m3 = MeshModel(m2.V, F, m2.synthetic, m2.fills)
    rep = manifold_report(m3)
    report.append(tr(f"Ergebnis: {rep['boundary_edges']} Randkanten, {rep['nonmanifold_edges']} nicht-mannigfaltig, "
                     f"{rep['components']} Teile", f"result: {rep['boundary_edges']} boundary edges, "
                     f"{rep['nonmanifold_edges']} non-manifold, {rep['components']} parts"))
    return m3, report, face_map


# ======================================================================== Glätten / Ausreißer
def remove_spikes(model, sigma, k=4.0, iters=2):
    """Punkte, die mehr als k·σ über ihrer 1-Ring-Tangentialebene liegen, auf diese ziehen."""
    V = model.V.copy()
    A = vertex_adjacency(model)
    deg = np.asarray(A.sum(1)).ravel()
    moved = np.zeros(len(V), bool)
    bnd = boundary_vertex_mask(model)
    for _ in range(iters):
        cen = (A @ V) / np.maximum(deg, 1)[:, None]
        m = MeshModel(V, model.F)
        n = m.vertex_normals
        d = np.einsum("ij,ij->i", V - cen, n)
        bad = (np.abs(d) > k * sigma) & ~bnd & (deg > 2)
        V[bad] -= d[bad, None] * n[bad]
        moved |= bad
    return model.with_vertices(V), int(moved.sum())


def region_vertices(model, face_ids):
    face_ids = np.asarray(face_ids, np.int64)
    inner = np.unique(model.F[face_ids])
    mask = np.zeros(model.n_faces, bool)
    mask[face_ids] = True
    outside = np.unique(model.F[~mask])
    fixed = np.intersect1d(inner, outside)
    bnd = boundary_vertex_mask(model)
    free = np.setdiff1d(inner, fixed)
    free = free[~bnd[free]]
    return free


def smooth_region(model, face_ids, iters=10, mode="taubin"):
    """Bereich glätten. 'taubin': volumenerhaltende Laplace-Glättung (Rauschen/Riffel),
    'fair': biharmonische Fairing-Lösung Δ²x = 0 mit festem Rand (Falten, Beulen, Murks) –
    ergibt die glatteste Fläche, die stetig an die Umgebung anschließt."""
    free = region_vertices(model, face_ids)
    if len(free) == 0:
        return model
    V = model.V.copy()
    nV = len(V)
    L = uniform_laplacian(nV, model.F)
    if mode == "fair":
        LU = L[free] @ L
        known = np.setdiff1d(np.arange(nV), free)
        M = LU[:, free].tocsc()
        rhs = -(LU[:, known] @ V[known])
        try:
            sol = np.column_stack([spsolve(M, rhs[:, d]) for d in range(3)])
            if np.all(np.isfinite(sol)):
                V[free] = sol
        except Exception:
            mode = "taubin"
    if mode == "taubin":
        A = vertex_adjacency(model)
        deg = np.maximum(np.asarray(A.sum(1)).ravel(), 1)
        for _ in range(int(iters)):
            for lam in (0.5, -0.53):
                d = (A @ V) / deg[:, None] - V
                V[free] += lam * d[free]
    return model.with_vertices(V)


def project_region(model, face_ids, feature):
    """Punkte des Bereichs exakt auf die Feature-Fläche ziehen."""
    from .carriers import project_points_to_feature
    vid = np.unique(model.F[np.asarray(face_ids, np.int64)])
    V = model.V.copy()
    V[vid] = project_points_to_feature(feature, V[vid])
    return model.with_vertices(V)


# ======================================================================== Triangulierung
def _lawson_flips(P, tris, loop_sizes, max_rounds=20):
    """Kanten im Inneren so lange umklappen, bis die Triangulierung (bei festen Randkanten)
    Delaunay ist – verbessert die schmalen Dreiecke der Ohr-Abschneide-Triangulierung."""
    tris = [list(t) for t in np.asarray(tris, np.int64)]
    rim = set()
    off = 0
    for n in loop_sizes:
        for i in range(n):
            a, b = off + i, off + (i + 1) % n
            rim.add((min(a, b), max(a, b)))
        off += n

    def incircle(a, b, c, d):
        m = np.array([[P[a][0] - P[d][0], P[a][1] - P[d][1], (P[a][0] - P[d][0]) ** 2 + (P[a][1] - P[d][1]) ** 2],
                      [P[b][0] - P[d][0], P[b][1] - P[d][1], (P[b][0] - P[d][0]) ** 2 + (P[b][1] - P[d][1]) ** 2],
                      [P[c][0] - P[d][0], P[c][1] - P[d][1], (P[c][0] - P[d][0]) ** 2 + (P[c][1] - P[d][1]) ** 2]])
        return np.linalg.det(m)

    def orient(a, b, c):
        return (P[b][0] - P[a][0]) * (P[c][1] - P[a][1]) - (P[b][1] - P[a][1]) * (P[c][0] - P[a][0])

    for _ in range(max_rounds):
        emap = {}
        for ti, t in enumerate(tris):
            for k in range(3):
                a, b = t[k], t[(k + 1) % 3]
                emap.setdefault((min(a, b), max(a, b)), []).append(ti)
        flipped = 0
        done = set()
        for e, ts in emap.items():
            if len(ts) != 2 or e in rim:
                continue
            t1, t2 = ts
            if t1 in done or t2 in done:
                continue
            A, B = e
            c = [v for v in tris[t1] if v not in e]
            d = [v for v in tris[t2] if v not in e]
            if len(c) != 1 or len(d) != 1:
                continue
            c, d = c[0], d[0]
            # Dreieck t1 gegen den Uhrzeigersinn ordnen
            if orient(A, B, c) < 0:
                A, B = B, A
            if incircle(A, B, c, d) <= 1e-12:
                continue
            # neues Paar (c, d) nur, wenn das Viereck konvex ist
            if orient(c, d, A) * orient(c, d, B) >= 0 or orient(A, B, c) * orient(A, B, d) >= 0:
                continue
            tris[t1] = [A, d, c] if orient(A, d, c) > 0 else [A, c, d]
            tris[t2] = [B, c, d] if orient(B, c, d) > 0 else [B, d, c]
            done.add(t1)
            done.add(t2)
            flipped += 1
        if not flipped:
            break
    return np.asarray(tris, np.int64)


def _earcut(loops):
    import mapbox_earcut as ec
    verts = np.vstack(loops).astype(np.float64)
    rings = np.cumsum([len(l) for l in loops]).astype(np.uint32)
    idx = np.asarray(ec.triangulate_float64(verts, rings), np.int64)
    return idx.reshape(-1, 3)


def triangulate_polygon(outer2d, inners2d=()):
    """Triangulierung eines einfachen Polygons mit Löchern. Bevorzugt Ohr-Abschneiden (mapbox
    earcut, robust auch bei zackigen Rändern) + Delaunay-Kantenflips; sonst eingeschränktes
    Delaunay (VTK). Rückgabe: Dreiecke als Indizes in die Punktliste [outer, inner1, ...]."""
    loops = [np.asarray(outer2d, float)] + [np.asarray(l, float) for l in inners2d]
    pts = np.vstack(loops)
    # sich selbst schneidende oder entartete Ränder ablehnen (Aufrufer haben Ausweichverfahren);
    # vtkDelaunay2D kann daran Gigabytes an Speicher verbrauchen
    span = float(np.ptp(pts, axis=0).max()) if len(pts) else 0.0
    for l in loops:
        if len(l) < 3:
            raise RuntimeError("Delaunay: Rand zu kurz")
        seg = np.linalg.norm(np.roll(l, -1, 0) - l, axis=1)
        if seg.min() <= 1e-9 * max(span, 1e-9):
            raise RuntimeError("Delaunay: doppelte Randpunkte")
    if polygon_self_intersections(loops):
        raise RuntimeError("Delaunay: Rand schneidet sich selbst")
    sizes = [len(l) for l in loops]
    try:
        tris = _earcut(loops)
        if triangulation_ok(tris, sizes):
            if len(tris) <= 20000:
                t2 = _lawson_flips(pts, tris, sizes)
                if triangulation_ok(t2, sizes):
                    tris = t2
            return tris
    except ImportError:
        pass
    except Exception:
        pass
    import pyvista as pv
    P3 = np.c_[pts, np.zeros(len(pts))]
    cells = []
    off = 0
    for k, l in enumerate(loops):
        n = len(l)
        ids = np.arange(off, off + n)
        ccw = polygon_area_2d(l) > 0
        if (k == 0) != ccw:   # außen gegen, innen im Uhrzeigersinn
            ids = ids[::-1]
        cells.append(np.r_[n, ids])
        off += n
    src = pv.PolyData(P3, faces=np.concatenate(cells))
    cloud = pv.PolyData(P3)
    out = cloud.delaunay_2d(edge_source=src, tol=1e-9)
    if out.n_points != len(pts):
        raise RuntimeError("Delaunay hat Punkte verändert")
    tris = np.asarray(out.regular_faces) if out.n_cells else np.zeros((0, 3), int)
    if not triangulation_ok(tris, sizes):
        raise RuntimeError("Delaunay: Rand nicht vollständig erhalten")
    return tris


def triangulation_ok(tris, loop_sizes):
    """Prüft, ob jede Randkante genau einmal und jede innere Kante genau zweimal vorkommt."""
    if len(tris) == 0:
        return False
    e = np.sort(np.vstack([tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]]]), axis=1)
    keys, cnt = np.unique(e[:, 0] * 10_000_000 + e[:, 1], return_counts=True)
    rim = []
    off = 0
    for n in loop_sizes:
        ids = np.arange(off, off + n)
        a, b = ids, np.roll(ids, -1)
        rim.append(np.minimum(a, b) * 10_000_000 + np.maximum(a, b))
        off += n
    rim = np.concatenate(rim)
    pos = np.searchsorted(keys, rim)
    pos = np.clip(pos, 0, len(keys) - 1)
    if not np.all(keys[pos] == rim) or not np.all(cnt[pos] == 1):
        return False
    inner = np.ones(len(keys), bool)
    inner[pos] = False
    return bool(np.all(cnt[inner] == 2))


def orient_to_rim(tris, rim_halfedges):
    """Dreiecke so drehen, dass jede vorhandene Randhalbkante (a->b) als (b->a) auftaucht."""
    tris = np.asarray(tris, np.int64)
    rim = set(rim_halfedges)
    same = sum(1 for t in tris for k in range(3) if (t[k], t[(k + 1) % 3]) in rim)
    opp = sum(1 for t in tris for k in range(3) if (t[(k + 1) % 3], t[k]) in rim)
    return tris[:, ::-1] if same > opp else tris


# ======================================================================== Ränder / Boden
def rim_info(model, loop):
    P = model.V[loop]
    c, ev, evec = pca_frame(P)
    n = evec[:, 2]
    flat = float(np.sqrt(max(ev[2], 0.0)))
    per = float(np.linalg.norm(np.diff(np.vstack([P, P[:1]]), axis=0), axis=1).sum())
    # Normale nach außen (weg vom Material): Netz liegt auf der anderen Seite
    m = np.zeros(model.n_faces, bool)
    vmask = np.zeros(model.n_verts, bool)
    vmask[loop] = True
    near = vmask[model.F].any(1)
    inner_c = model.centroids[near].mean(0)
    if (inner_c - c) @ n > 0:
        n = -n
    return {"center": c, "normal": n, "flatness": flat, "perimeter": per, "diameter": per / np.pi,
            "n": len(loop)}


def principal_axes_rotation(model):
    """Rotation in die Hauptachsen (größte Ausdehnung = X, kleinste = Z), Rückgabe (R, Mitte).
    Das Vorzeichen von Z ist bei PCA beliebig; gewählt wird so, dass das Teil „steht“:
    ein großer offener Rand (Scan vom Drehteller) nach unten, sonst die größte ebene Fläche nach
    unten, bei Gleichstand der Flächenschwerpunkt in der unteren Hälfte."""
    c, ev, evec = pca_frame(model.V, model.vertex_areas)
    R = evec.T.copy()
    if np.linalg.det(R) < 0:
        R[2] = -R[2]
    ez = R[2]
    z = model.centroids @ ez
    zmin, zmax = float(z.min()), float(z.max())
    mid = 0.5 * (zmin + zmax)
    flip = None
    try:
        loops = model.boundary_loops()
    except Exception:
        loops = []
    if loops:
        big = max(loops, key=len)
        P = model.V[big]
        per = float(np.linalg.norm(np.diff(np.vstack([P, P[:1]]), axis=0), axis=1).sum())
        if per > 0.5 * model.diag:
            flip = float((P @ ez).mean()) > mid
    if flip is None:
        nz = model.normals @ ez
        A = model.areas
        bw = max(0.3, 0.003 * model.diag)

        def best_flat(mask):
            if not mask.any():
                return 0.0
            zz = z[mask]
            k = np.floor((zz - zz.min()) / bw).astype(np.int64)
            s = np.bincount(k, A[mask])
            return float((s + np.r_[s[1:], 0.0]).max())   # Nachbar-Bins (Rauschen an Grenzen)
        down, up = best_flat(nz < -0.97), best_flat(nz > 0.97)
        if up > 1.15 * down:
            flip = True
        elif down > 1.15 * up:
            flip = False
        else:
            flip = float((z * A).sum() / A.sum()) > mid
    if flip:
        R = np.diag([1.0, -1.0, -1.0]) @ R
    return R, c


def wall_directions(model, loop, smooth=3):
    """Verlängerungsrichtung der Wand an jedem Randpunkt: tangential zur Wand, senkrecht zum
    Rand, vom Material weg (lineare Extrapolation der Wand)."""
    V = model.V
    loop = np.asarray(loop, np.int64)
    P = V[loop]
    # Tangente mit breiterer Schablone (Treppen-/Zackenränder von Voxel-Scans)
    k = 2 if len(loop) > 12 else 1
    t = unit(np.roll(P, -k, 0) - np.roll(P, k, 0))
    # Wandnormale: Punktnormalen über zwei Ringe gemittelt, dann entlang des Rands geglättet.
    # Netz und Randschleife sind konsistent orientiert → cross(Normale, Tangente) zeigt überall
    # zur selben Seite; nur das Vorzeichen insgesamt wird per Mehrheitsentscheid bestimmt
    # (punktweise Entscheidungen kippen bei Rauschen und zerstören die Glättung).
    A = vertex_adjacency(model)
    vn = model.vertex_normals
    vn2 = A @ (A @ vn + vn) + vn
    nrm = unit(vn2[loop])
    for _ in range(2):
        nrm = unit(nrm + 0.5 * (np.roll(nrm, 1, 0) + np.roll(nrm, -1, 0)))
    d = unit(np.cross(nrm, t))
    rimset = np.zeros(model.n_verts, bool)
    rimset[loop] = True
    votes = 0.0
    for i, v in enumerate(loop):
        nb = A.indices[A.indptr[v]:A.indptr[v + 1]]
        nb = nb[~rimset[nb]]
        if len(nb):
            votes += float(d[i] @ (V[nb].mean(0) - P[i]))
    out = -d if votes > 0 else d
    # glätten (Rauschen der Normalen)
    for _ in range(int(smooth)):
        out = unit(out + 0.5 * (np.roll(out, 1, 0) + np.roll(out, -1, 0)))
    return out


def polygon_self_intersections(loops2d, max_count=1):
    """Anzahl sich echt schneidender, nicht benachbarter Kantenpaare aller Schleifen (2D).
    Bricht nach max_count Treffern ab. Schutz vor vtkDelaunay2D, das bei sich selbst
    schneidenden Rändern sehr lange laufen und viel Speicher belegen kann."""
    A, B, nxt_id, prv_id = [], [], [], []
    off = 0
    for l in loops2d:
        l = np.asarray(l, float)
        m = len(l)
        A.append(l)
        B.append(np.roll(l, -1, 0))
        ids = np.arange(off, off + m)
        nxt_id.append(np.roll(ids, -1))
        prv_id.append(np.roll(ids, 1))
        off += m
    A, B = np.vstack(A), np.vstack(B)
    nxt_id, prv_id = np.concatenate(nxt_id), np.concatenate(prv_id)
    N = len(A)
    lo, hi = np.minimum(A, B), np.maximum(A, B)
    R = B - A
    hits = 0
    chunk = 256
    for s0 in range(0, N, chunk):
        idx = np.arange(s0, min(N, s0 + chunk))
        # Bounding-Box-Vorfilter
        bb = ((lo[idx, None, 0] <= hi[None, :, 0]) & (hi[idx, None, 0] >= lo[None, :, 0]) &
              (lo[idx, None, 1] <= hi[None, :, 1]) & (hi[idx, None, 1] >= lo[None, :, 1]))
        bb[np.arange(len(idx)), idx] = False
        bb[np.arange(len(idx)), nxt_id[idx]] = False
        bb[np.arange(len(idx)), prv_id[idx]] = False
        ii, jj = np.nonzero(bb)
        if len(ii) == 0:
            continue
        i, j = idx[ii], jj
        r, sq = R[i], R[j]
        den = r[:, 0] * sq[:, 1] - r[:, 1] * sq[:, 0]
        w = A[j] - A[i]
        with np.errstate(all="ignore"):
            t = (w[:, 0] * sq[:, 1] - w[:, 1] * sq[:, 0]) / den
            u = (w[:, 0] * r[:, 1] - w[:, 1] * r[:, 0]) / den
        eps = 1e-12
        hit = np.isfinite(t) & (t > eps) & (t < 1 - eps) & (u > eps) & (u < 1 - eps)
        hits += int(hit.sum())
        if hits >= 2 * max_count:
            break
    return hits // 2


def plane_from_rims(model, loops, mode="fit", value=None, ref=None):
    """Bodenebene bestimmen.
    mode 'fit'    Ausgleichsebene durch alle Randpunkte
         'lowest' Normale wie 'fit', Lage am äußersten Randpunkt (alle Wände werden verlängert)
         'z'      Ebene z = value
         'offset' parallel zu ref=(o, n) im Abstand value (Seite des Rands)
         'plane'  exakt ref=(o, n)
    Rückgabe (o, n) mit n vom Material weg."""
    P = np.vstack([model.V[l] for l in loops])
    info = rim_info(model, max(loops, key=len))
    n_out = info["normal"]
    if mode == "z":
        n = np.array([0, 0, 1.0]) * (1 if n_out[2] >= 0 else -1)
        return np.array([0, 0, float(value)]), n
    if mode in ("offset", "plane"):
        o, n = np.asarray(ref[0], float), unit(ref[1])
        if n @ n_out < 0:
            n = -n
        if mode == "plane":
            return o, n
        side = 1.0 if ((P.mean(0) - o) @ n) >= 0 else -1.0
        return o + side * float(value) * n, n
    # robust: Randpunkte weit weg von der Ebene (Schlitze, Einrisse, Fetzen) iterativ ausschließen
    keep = np.ones(len(P), bool)
    c, ev, evec = pca_frame(P)
    n = evec[:, 2]
    for _ in range(4):
        d = (P - c) @ n
        med = float(np.median(d[keep]))
        s = 1.4826 * float(np.median(np.abs(d[keep] - med))) + 1e-9
        k2 = np.abs(d - med) < max(4.0 * s, 1e-6)
        if k2.sum() < max(10, 0.5 * len(P)) or np.array_equal(k2, keep):
            break
        keep = k2
        c, ev, evec = pca_frame(P[keep])
        n = evec[:, 2]
    if n @ n_out < 0:
        n = -n
    if mode == "lowest":
        c = c + n * float(((P[keep] - c) @ n).max())
    return c, n


def close_rim_notches(model, loop, normal, max_width=None, min_ratio=3.0, max_iter=50):
    """Schmale Einrisse/Schlitze im offenen Rand (Scan-Lücken in einer Wand, z. B. an Ecken)
    schließen, bevor der Boden gedeckelt wird – sonst überlappt sich der Rand in der Projektion
    auf die Bodenebene.

    Ein Einriss ist ein Randstück zwischen zwei Punkten, die im Raum nah (Sehne < max_width),
    entlang des Rands aber weit auseinander liegen (Bogen > min_ratio · Sehne). Er wird als
    eigenes Loch gefüllt (bearbeitbar unter „Füllungen“); der Rest des Rands läuft über die Sehne.
    Nur Einrisse, die in einer Wand liegen (Normale des Einrisses nicht parallel zur
    Bodennormale) – Schlitze in der Bodenkontur selbst (Nuten, Stege) bleiben unberührt.
    Rückgabe (Netz, Restschleife, Anzahl geschlossener Einrisse)."""
    from scipy.spatial import cKDTree

    from .holes import fill_hole
    nb = unit(np.asarray(normal, float))
    loop = np.asarray(loop, np.int64)
    count = 0
    for _ in range(max_iter):
        n = len(loop)
        if n < 12:
            break
        P = model.V[loop]
        seg = np.linalg.norm(np.roll(P, -1, 0) - P, axis=1)
        cum = np.r_[0.0, np.cumsum(seg)]
        L = cum[-1]
        w = max_width if max_width is not None else max(6.0 * float(np.median(seg)),
                                                         0.05 * float(np.ptp(P, axis=0).max()))
        pairs = cKDTree(P).query_pairs(w, output_type="ndarray")
        if len(pairs) == 0:
            break
        i, j = pairs[:, 0], pairs[:, 1]
        chord = np.linalg.norm(P[i] - P[j], axis=1)
        fwd = cum[j] - cum[i]                      # Bogen i→j in Schleifenrichtung (i < j)
        arc = np.minimum(fwd, L - fwd)
        ok = (arc > min_ratio * np.maximum(chord, 1e-9)) & (arc > 3.0 * np.median(seg))
        if not ok.any():
            break
        cand = np.nonzero(ok)[0][np.argsort(-arc[ok])]
        done = False
        for k in cand[:200]:
            a, b = int(i[k]), int(j[k])
            if fwd[k] <= L - fwd[k]:
                idx = np.arange(a, b + 1)          # Einriss a→b, Rest b→…→a
            else:
                idx = np.r_[np.arange(b, n), np.arange(0, a + 1)]
            Q = P[idx]
            X = Q - Q.mean(0)
            ev, evec = np.linalg.eigh(X.T @ X)
            if abs(float(evec[:, 0] @ nb)) > 0.7:  # Einriss liegt in der Bodenebene → Kontur, nicht Lücke
                continue
            try:
                m2, _ = fill_hole(model, loop[idx], "smooth", kind="hole")
            except Exception:
                continue
            m2.fills[-1]["label"] = tr("Randspalt", "rim slit")
            model = m2
            keep = np.ones(n, bool)
            keep[idx[1:-1]] = False
            loop = loop[keep]
            count += 1
            done = True
            break
        if not done:
            break
    return model, loop, count


def _merge_close_points(Q, eps):
    """Aufeinanderfolgende Punkte einer geschlossenen Folge zusammenfassen, solange sie näher als
    eps am ersten Punkt der Gruppe liegen. Rückgabe Gruppennummer je Punkt (0..g-1, in Reihenfolge)."""
    m = len(Q)
    grp = np.zeros(m, np.int64)
    g = 0
    anchor = Q[0]
    for i in range(1, m):
        if np.linalg.norm(Q[i] - anchor) > eps:
            g += 1
            anchor = Q[i]
        grp[i] = g
    # letzte Gruppe mit der ersten verschmelzen, wenn sie am Anfang klebt
    if g > 0 and np.linalg.norm(Q[-1] - Q[0]) <= eps:
        grp[grp == g] = 0
        g -= 1
    return grp, g + 1


def _close_base_try(model, loops, o, n, smooth, merge_eps=0.0):
    """Ein Versuch: smooth=None → senkrechte Projektion, sonst Wandverlängerung mit so vielen
    Glättungsschritten der Wandrichtungen. Projizierte Punkte, die näher als merge_eps
    beieinander liegen (Rand läuft senkrecht an einer Wand hoch), werden zu einem Bodenpunkt
    zusammengefasst – der Wandstreifen wird dort zum Fächer."""
    V = model.V
    new_pts = []
    new_faces = []
    proj_loops = []      # globale Indizes der Schleifen in der Ebene
    nxt = model.n_verts
    rim_he = []
    n_back = 0
    for loop in loops:
        loop = np.asarray(loop, np.int64)
        P = V[loop]
        dist = (P - o) @ n          # >0: Rand liegt außerhalb (jenseits der Ebene)
        if np.max(np.abs(dist)) < 1e-6:
            # Rand liegt schon in der Ebene
            rim_he += [(loop[i], loop[(i + 1) % len(loop)]) for i in range(len(loop))]
            proj_loops.append(loop)
            continue
        if smooth is not None:
            D = wall_directions(model, loop, smooth)
            dn = D @ n
            s = -dist / np.where(np.abs(dn) > 0.2, dn, np.nan)
            ok = np.isfinite(s) & (s >= -1e-9)
            Q = np.where(ok[:, None], P + s[:, None] * D, P - dist[:, None] * n)
            n_back += int((~ok).sum())
        else:
            Q = P - dist[:, None] * n
        m = len(loop)
        if merge_eps > 0:
            grp, g = _merge_close_points(Q, merge_eps)
            Qg = np.zeros((g, 3))
            np.add.at(Qg, grp, Q)
            Qg /= np.bincount(grp, minlength=g)[:, None]
        else:
            grp, g, Qg = np.arange(m), m, Q
        if g < 3:
            raise RuntimeError(tr("Rand zu klein", "rim too small"))
        ids = np.arange(nxt, nxt + g)
        nxt += g
        new_pts.append(Qg)
        for i in range(m):
            a, b = loop[i], loop[(i + 1) % m]
            ga, gb = ids[grp[i]], ids[grp[(i + 1) % m]]
            if ga == gb:
                new_faces.append((b, a, ga))
            else:
                new_faces.append((b, a, ga))
                new_faces.append((b, ga, gb))
        rim_he += [(ids[k], ids[(k + 1) % g]) for k in range(g)]
        proj_loops.append(ids)
    allV = np.vstack([V] + new_pts) if new_pts else V
    x, y, _ = orthonormal_frame(n)
    loops2d = [np.c_[(allV[l] - o) @ x, (allV[l] - o) @ y] for l in proj_loops]
    areas = [abs(polygon_area_2d(l)) for l in loops2d]
    order = np.argsort(areas)[::-1]
    outer_i = order[0]
    inners_i = [i for i in order[1:]]
    if polygon_self_intersections(loops2d):
        raise RuntimeError(tr("Bodenpolygon schneidet sich selbst", "base polygon self-intersects"))
    tris = triangulate_polygon(loops2d[outer_i], [loops2d[i] for i in inners_i])
    return allV, new_pts, new_faces, proj_loops, rim_he, tris, outer_i, inners_i, n_back


def close_base(model, loops, plane, extend=True, label=None, feature_id=None):
    """Offenen Boden schließen: Wände (optional) tangential bis zur Ebene verlängern,
    dann eben deckeln. loops: Randschleifen (erste = außen, weitere = innere Ränder wie
    Bohrungen). Rückgabe: (Netz, Streifen-Dreiecke, Deckel-Dreiecke, Bericht)."""
    o, n = np.asarray(plane[0], float), unit(plane[1])
    report = []
    orig_model = model
    orig_loops = [np.asarray(l, np.int64) for l in loops]
    # Schlitze/Einrisse im Rand zuerst per Reißverschluss schließen (sonst überlappt sich der
    # Rand in der Projektion auf die Bodenebene)
    zloops = []
    n_notch = 0
    for loop in orig_loops:
        model, rest, c = close_rim_notches(model, loop, n)
        zloops.append(rest)
        n_notch += c
    if n_notch:
        report.append(tr(f"{n_notch} Einriss(e) im Rand vorab gefüllt (Scan › Füllungen: „Randspalt“)",
                         f"{n_notch} slit(s) in the rim filled first (Scan › Fills: 'rim slit')"))
    loops = zloops
    # Strategien: Wand tangential verlängern (normal bzw. stark geglättet – verrauschte
    # Scan-Ränder können sonst ein sich selbst schneidendes Bodenpolygon ergeben), zuletzt
    # senkrecht projizieren. Die erste Variante mit gültiger Triangulierung gewinnt.
    e_med = float(np.median(np.concatenate([np.linalg.norm(np.diff(model.V[l], axis=0), axis=1) for l in loops])))
    strategies = ([(3, 0.25), (40, 0.25)] if extend else []) + [(None, 0.25), (None, 1.0), (None, 3.0)]
    last_err = None
    for smooth, mf in strategies:
        try:
            res = _close_base_try(model, loops, o, n, smooth, mf * e_med)
            break
        except RuntimeError as e:
            last_err = e
    else:
        raise RuntimeError(tr(f"Boden lässt sich nicht deckeln ({last_err}). Rand vorher glätten/reparieren "
                              "oder andere Ebene wählen.",
                              f"Base cannot be capped ({last_err}). Smooth/repair the rim first or choose "
                              "another plane."))
    allV, new_pts, new_faces, proj_loops, rim_he, tris, outer_i, inners_i, n_back = res
    if extend and smooth is None:
        report.append(tr("Wandverlängerung hätte ein sich selbst schneidendes Bodenpolygon ergeben – "
                         "Rand senkrecht auf die Ebene projiziert",
                         "wall extension would self-intersect – rim projected perpendicular onto the plane"))
    elif extend and smooth != strategies[0][0]:
        report.append(tr("Wandrichtungen stark geglättet (verrauschter Rand)",
                         "wall directions smoothed strongly (noisy rim)"))
    gidx = np.concatenate([proj_loops[outer_i]] + [proj_loops[i] for i in inners_i])
    cap = gidx[tris]
    cap = orient_to_rim(cap, rim_he)
    strip = np.asarray(new_faces, np.int64).reshape(-1, 3)
    V_new = allV[model.n_verts:]
    fill = {"id": new_fill_id(model.fills), "kind": "base", "mode": "plane",
            "rim": [orig_model.V[l].copy() for l in orig_loops], "plane": (o, n), "feature": feature_id,
            "extend": bool(extend), "label": label or tr("Boden", "Base")}
    m2, ids = append_geometry(model, V_new, np.vstack([strip, cap]) if len(strip) else cap, True, fill)
    strip_ids = ids[:len(strip)]
    cap_ids = ids[len(strip):]
    if n_back:
        report.append(tr(f"{n_back} Randpunkte liegen schon jenseits der Ebene (senkrecht projiziert)",
                         f"{n_back} rim points already beyond the plane (projected)"))
    report.append(tr(f"Boden geschlossen: {len(loops)} Ränder, {len(strip_ids)} Wand- + {len(cap_ids)} Deckeldreiecke",
                     f"base closed: {len(loops)} rims, {len(strip_ids)} wall + {len(cap_ids)} cap triangles"))
    return m2, strip_ids, cap_ids, report


def rims_near_plane(model, holes, plane, max_dist):
    """Randschleifen, deren Punkte alle nahe an der Ebene liegen (für Innenringe am Boden)."""
    o, n = np.asarray(plane[0]), unit(plane[1])
    out = []
    for h in holes:
        d = np.abs((model.V[h["loop"]] - o) @ n)
        if d.max() <= max_dist:
            out.append(h["loop"])
    return out


# ======================================================================== Füllungen bearbeiten
def find_fill(model, fill_id):
    for i, fl in enumerate(model.fills):
        if fl.get("id") == fill_id:
            return i, fl
    raise KeyError(fill_id)


def remove_fill(model, fill_id):
    """Füllung entfernen -> der Rand erscheint wieder als Loch."""
    i, fl = find_fill(model, fill_id)
    keep = np.ones(model.n_faces, bool)
    keep[np.asarray(fl["faces"], np.int64)] = False
    m2, fmap = compact(model, keep)
    m2.fills = [f for f in m2.fills if f.get("id") != fill_id]
    return m2, fmap, fl


def loops_for_rim(model, rims):
    """Aktuelle Randschleifen wiederfinden, die zu gespeicherten Randpunkten passen."""
    loops = model.boundary_loops()
    out = []
    for R in rims:
        R = np.asarray(R)
        c = R.mean(0)
        best = None
        for l in loops:
            P = model.V[l]
            if abs(len(l) - len(R)) > 0.5 * len(R) + 3:
                continue
            d = np.linalg.norm(P.mean(0) - c)
            if best is None or d < best[0]:
                best = (d, l)
        if best is not None:
            out.append(best[1])
    return out


def refill(model, fill_id, mode, plane=None):
    """Füllung entfernen und mit anderem Modus neu füllen. Rückgabe (Netz, face_map, neue ids)."""
    from .holes import fill_hole
    m2, fmap, fl = remove_fill(model, fill_id)
    loops = loops_for_rim(m2, fl["rim"])
    new_ids = []
    if fl.get("kind") == "base":
        pl = plane if plane is not None else fl.get("plane")
        m3, s_ids, c_ids, _ = close_base(m2, loops, pl, fl.get("extend", True), fl.get("label"), fl.get("feature"))
        m3.fills[-1]["id"] = fl["id"]
        return m3, fmap, np.r_[s_ids, c_ids], c_ids
    m3 = m2
    for loop in loops:
        m3, ids = fill_hole(m3, loop, mode, plane=plane, kind=fl.get("kind", "hole"))
        new_ids.append(ids)
    if m3.fills:
        m3.fills[-1]["id"] = fl["id"]
    ids = np.concatenate(new_ids) if new_ids else np.zeros(0, np.int64)
    return m3, fmap, ids, ids


def fair_fill(model, fill_id):
    _, fl = find_fill(model, fill_id)
    return smooth_region(model, fl["faces"], mode="fair")


def delete_and_fill(model, face_ids, mode="smooth"):
    """Bereich (z. B. Artefakt, Stütze, Reflexionsmüll) ausschneiden und das Loch füllen."""
    from .holes import fill_hole
    face_ids = np.asarray(face_ids, np.int64)
    touched = np.unique(model.F[face_ids])
    keep = np.ones(model.n_faces, bool)
    keep[face_ids] = False
    m2, fmap = compact(model, keep)
    # Punkte, die übrig bleiben und am gelöschten Bereich lagen
    vm = -np.ones(model.n_verts, np.int64)
    used = np.unique(model.F[keep])
    vm[used] = np.arange(len(used))
    touched_new = vm[touched]
    touched_new = set(touched_new[touched_new >= 0].tolist())
    loops = [l for l in m2.boundary_loops() if touched_new.intersection(l.tolist())]
    m3 = m2
    done = set()
    if mode == "flat" and len(loops) > 1:
        # ineinanderliegende, ebene Ränder (z. B. Ringnut um eine stehen bleibende Insel):
        # gemeinsam als Polygon mit Löchern füllen, sonst überlappen sich die Deckel
        for outer, inners in nested_flat_groups(m2.V, loops):
            try:
                m3 = fill_flat_multi(m3, loops[outer], [loops[i] for i in inners])
                done.add(outer)
                done.update(inners)
            except Exception:
                pass
    for k, loop in enumerate(loops):
        if k in done:
            continue
        try:
            m3, _ = fill_hole(m3, loop, mode, kind="replace")
        except Exception:
            continue
    return m3, fmap, len(loops)


def _newell(P):
    Q = np.roll(P, -1, axis=0)
    n = np.array([np.sum((P[:, 1] - Q[:, 1]) * (P[:, 2] + Q[:, 2])),
                  np.sum((P[:, 2] - Q[:, 2]) * (P[:, 0] + Q[:, 0])),
                  np.sum((P[:, 0] - Q[:, 0]) * (P[:, 1] + Q[:, 1]))])
    L = np.linalg.norm(n)
    return n / L if L > 1e-15 else np.array([0.0, 0.0, 1.0])


def nested_flat_groups(V, loops, rel_tol=0.04, abs_tol=0.1):
    """Gruppen (außen, [innen…]) ebener Randschleifen, deren innere in der Ebene der äußeren und
    in ihr liegen."""
    from .geom_utils import point_in_polygon
    info = []
    for l in loops:
        P = V[np.asarray(l, np.int64)]
        info.append((P, P.mean(0), _newell(P), float(np.ptp(P, axis=0).max())))
    order = sorted(range(len(loops)), key=lambda i: -info[i][3])
    parent = {}
    for jj, j in enumerate(order):
        Pj, cj, nj, sj = info[j]
        for i in reversed(order[:jj]):
            if i in parent:
                continue
            Pi, ci, ni, si = info[i]
            if abs(float(ni @ nj)) < 0.95:
                continue
            if np.abs((Pj - ci) @ ni).max() > max(rel_tol * si, abs_tol):
                continue
            x, y, _ = orthonormal_frame(ni)
            poly = np.c_[(Pi - ci) @ x, (Pi - ci) @ y]
            if point_in_polygon(np.array([(cj - ci) @ x, (cj - ci) @ y]), poly) and \
                    point_in_polygon(np.array([(Pj[0] - ci) @ x, (Pj[0] - ci) @ y]), poly):
                parent[j] = i
                break
    groups = {}
    for j, i in parent.items():
        groups.setdefault(i, []).append(j)
    return [(i, js) for i, js in groups.items()]


def fill_flat_multi(model, outer, inners, kind="replace"):
    """Ebener Deckel für einen Rand mit inneren Rändern (Polygon mit Löchern)."""
    V = model.V
    loops = [np.asarray(outer, np.int64)] + [np.asarray(l, np.int64) for l in inners]
    o = V[loops[0]].mean(0)
    n = _newell(V[loops[0]])
    x, y, _ = orthonormal_frame(n)
    loops2d = [np.c_[(V[l] - o) @ x, (V[l] - o) @ y] for l in loops]
    tris = np.asarray(triangulate_polygon(loops2d[0], loops2d[1:]), np.int64)
    gidx = np.concatenate(loops)
    T = gidx[tris]
    rim = set()
    for l in loops:
        rim.update(zip(l.tolist(), np.roll(l, -1).tolist()))
    same = sum(1 for t in T for k in range(3) if (int(t[k]), int(t[(k + 1) % 3])) in rim)
    opp = sum(1 for t in T for k in range(3) if (int(t[(k + 1) % 3]), int(t[k])) in rim)
    if same > opp:
        T = T[:, ::-1]
    F2 = np.vstack([model.F, T])
    new_faces = np.arange(model.n_faces, len(F2))
    syn = np.r_[model.synthetic, np.ones(len(T), bool)]
    fills = list(model.fills)
    fills.append({"id": new_fill_id(model.fills), "kind": kind, "mode": "flat", "faces": new_faces,
                  "rim": [V[l].copy() for l in loops], "plane": None, "feature": None, "extend": False})
    return MeshModel(V.copy(), F2, syn, fills)
