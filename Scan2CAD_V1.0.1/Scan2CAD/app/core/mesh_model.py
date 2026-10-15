"""Dreiecksnetz mit vorberechneten Größen für Auswahl und Fitting."""
from __future__ import annotations

import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree


SYNTH_WEIGHT = 0.1   # Gewicht künstlicher Dreiecke (Lochfüllungen, Boden) beim Fitten


class MeshModel:
    """Dreiecksnetz.

    synthetic : bool je Dreieck – künstlich erzeugt (Lochfüllung, Bodendeckel, Wandverlängerung).
                Solche Dreiecke tragen keine Messinformation: Fits gewichten sie mit SYNTH_WEIGHT,
                RANSAC nutzt sie nie als Startpunkt.
    fills     : Protokoll der Füllungen (Liste von dicts, s. mesh_ops), damit sie später
                neu gefüllt / gefairt / als Ebene erzwungen / entfernt werden können.
    """

    def __init__(self, V, F, synthetic=None, fills=None):
        self.V = np.ascontiguousarray(V, dtype=np.float64)
        self.F = np.ascontiguousarray(F, dtype=np.int64)
        if synthetic is None or len(synthetic) != len(self.F):
            synthetic = np.zeros(len(self.F), bool)
        self.synthetic = np.asarray(synthetic, bool)
        self.fills = list(fills) if fills else []
        self._update_geometry()
        self._pairs = None
        self._angles = None
        self._adj = None
        self._ctree = None
        self._angle_graph_cache = {}
        self._boundary_edge_count = None

    # ------------------------------------------------------------ Grundgrößen
    def _update_geometry(self):
        tri = self.V[self.F]
        cr = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        nrm = np.linalg.norm(cr, axis=1)
        self.areas = 0.5 * nrm
        self.normals = cr / np.maximum(nrm, 1e-300)[:, None]
        self.centroids = tri.mean(axis=1)
        nV = len(self.V)
        vn = np.zeros((nV, 3))
        va = np.zeros(nV)
        for k in range(3):
            for c in range(3):
                vn[:, c] += np.bincount(self.F[:, k], weights=cr[:, c], minlength=nV)
            va += np.bincount(self.F[:, k], weights=self.areas / 3.0, minlength=nV)
        self.vertex_normals = vn / np.maximum(np.linalg.norm(vn, axis=1), 1e-300)[:, None]
        self.vertex_areas = va

    def smoothed_normals(self, iterations=3):
        """Flächengewichtete Normalenglättung über die Nachbarschaft (Dual-Graph).

        Scan-Netze enthalten oft Splitter-Dreiecke mit stark verrauschten Normalen;
        für Fits/Erkennung werden daher geglättete Normalen verwendet.
        """
        key = int(iterations)
        cache = getattr(self, "_sn_cache", None)
        if cache is None:
            cache = self._sn_cache = {}
        if key in cache:
            return cache[key]
        if key <= 0:
            return self.normals
        A = self.adjacency.astype(np.float64)
        A = A + sparse.eye(self.n_faces, format="csr")
        N = self.normals * self.areas[:, None]
        for _ in range(key):
            N = A @ N
            N = N / np.maximum(np.linalg.norm(N, axis=1), 1e-300)[:, None]
            N = N * self.areas[:, None] + 1e-300
        N = N / np.maximum(np.linalg.norm(N, axis=1), 1e-300)[:, None]
        cache[key] = N
        return N

    @property
    def exact(self):
        """CAD-Export? Dann liegen die Eckpunkte exakt auf den wahren Flächen (Kennzeichen: viele
        exakt koplanare Nachbardreiecke außerhalb von Füllungen) – Fits und Erkennung nutzen dann
        die Eckpunkte statt der Schwerpunkte (diese liegen bei gekrümmten Facetten daneben)."""
        if getattr(self, "_exact", None) is None:
            try:
                pairs, ang = self.adjacency_pairs
                real = ~self.synthetic[pairs[:, 0]] & ~self.synthetic[pairs[:, 1]]
                self._exact = bool(real.sum() > 20 and np.mean(ang[real] < 1e-3) > 0.05)
            except Exception:
                self._exact = False
        return self._exact

    @property
    def fit_normals(self):
        if self.exact:
            return self.normals          # Facettennormalen nicht über Kanten hinweg verschmieren
        return self.smoothed_normals(2)

    def fit_samples(self, face_ids, max_pts=80000, seed=0):
        """Stützpunkte für Fits: (Punkte, Normalen, Gewichte). Scan: Schwerpunkte der Dreiecke.
        CAD-Export: die Eckpunkte der Dreiecke (exakt auf der Fläche), Normalen/Gewichte nur aus
        den gewählten Dreiecken."""
        ids = np.asarray(face_ids, np.int64)
        if not self.exact:
            if len(ids) > max_pts:
                ids = np.random.default_rng(seed).choice(ids, max_pts, replace=False)
            return self.centroids[ids], self.fit_normals[ids], self.weights[ids]
        F = self.F[ids]
        w_f = self.weights[ids]
        vid, inv = np.unique(F.ravel(), return_inverse=True)
        inv = inv.reshape(-1, 3)
        N = np.zeros((len(vid), 3))
        W = np.zeros(len(vid))
        nw = self.normals[ids] * w_f[:, None]
        for k in range(3):
            for c in range(3):
                N[:, c] += np.bincount(inv[:, k], weights=nw[:, c], minlength=len(vid))
            W += np.bincount(inv[:, k], weights=w_f / 3.0, minlength=len(vid))
        N = N / np.maximum(np.linalg.norm(N, axis=1), 1e-300)[:, None]
        P = self.V[vid]
        if len(P) > max_pts:
            sel = np.random.default_rng(seed).choice(len(P), max_pts, replace=False)
            P, N, W = P[sel], N[sel], W[sel]
        return P, N, W

    @property
    def weights(self):
        """Flächengewichte für Fits (künstliche Dreiecke stark abgewertet)."""
        return self.areas * np.where(self.synthetic, SYNTH_WEIGHT, 1.0)

    @property
    def quality(self):
        """Dreiecksqualität 4·sqrt(3)·A / Σ Kante² (1 = gleichseitig, ~0 = Splitter)."""
        if getattr(self, "_quality", None) is None:
            V, F = self.V, self.F
            e2 = ((V[F[:, 1]] - V[F[:, 0]]) ** 2).sum(1) + ((V[F[:, 2]] - V[F[:, 1]]) ** 2).sum(1) \
                + ((V[F[:, 0]] - V[F[:, 2]]) ** 2).sum(1)
            self._quality = 4 * np.sqrt(3) * self.areas / np.maximum(e2, 1e-300)
        return self._quality

    @property
    def edge_normals(self):
        """Normalen für Knickerkennung: Rohnormale bei gut geformten Dreiecken,
        geglättete Normale bei Splittern (deren Rohnormale ist durch Rauschen unbrauchbar)."""
        if getattr(self, "_edge_normals", None) is None:
            bad = self.quality < 0.3
            N = self.normals.copy()
            N[bad] = self.smoothed_normals(2)[bad]
            self._edge_normals = N
        return self._edge_normals

    @property
    def n_faces(self):
        return len(self.F)

    @property
    def n_verts(self):
        return len(self.V)

    @property
    def bbox(self):
        return self.V.min(0), self.V.max(0)

    @property
    def diag(self):
        mn, mx = self.bbox
        return float(np.linalg.norm(mx - mn))

    @property
    def total_area(self):
        return float(self.areas.sum())

    def mean_edge_length(self):
        e = self.V[self.F[:, 1]] - self.V[self.F[:, 0]]
        return float(np.linalg.norm(e, axis=1).mean())

    # ------------------------------------------------------------ Topologie
    def _build_adjacency(self):
        F = self.F
        nF = len(F)
        nV = len(self.V)
        e = F[:, [0, 1, 1, 2, 2, 0]].reshape(-1, 2)
        fidx = np.repeat(np.arange(nF), 3)
        es = np.sort(e, axis=1)
        key = es[:, 0] * nV + es[:, 1]
        order = np.argsort(key, kind="stable")
        ks = key[order]
        same = ks[1:] == ks[:-1]
        i0 = order[:-1][same]
        i1 = order[1:][same]
        pairs = np.stack([fidx[i0], fidx[i1]], axis=1)
        pairs = pairs[pairs[:, 0] != pairs[:, 1]]
        cosang = np.einsum("ij,ij->i", self.normals[pairs[:, 0]], self.normals[pairs[:, 1]])
        self._pairs = pairs
        self._angles = np.degrees(np.arccos(np.clip(cosang, -1, 1)))
        data = np.ones(2 * len(pairs), dtype=np.int8)
        rows = np.r_[pairs[:, 0], pairs[:, 1]]
        cols = np.r_[pairs[:, 1], pairs[:, 0]]
        self._adj = sparse.csr_matrix((data, (rows, cols)), shape=(nF, nF))
        uniq, cnt = np.unique(ks, return_counts=True)
        self._boundary_edge_count = int((cnt == 1).sum())

    @property
    def adjacency(self):
        if self._adj is None:
            self._build_adjacency()
        return self._adj

    @property
    def adjacency_pairs(self):
        if self._pairs is None:
            self._build_adjacency()
        return self._pairs, self._angles

    @property
    def is_closed(self):
        if self._boundary_edge_count is None:
            self._build_adjacency()
        return self._boundary_edge_count == 0

    @property
    def boundary_edge_count(self):
        if self._boundary_edge_count is None:
            self._build_adjacency()
        return self._boundary_edge_count

    @property
    def centroid_tree(self):
        if self._ctree is None:
            self._ctree = cKDTree(self.centroids)
        return self._ctree

    def angle_graph_labels(self, max_angle_deg):
        """Zusammenhangskomponenten, wenn nur Kanten mit Knickwinkel < max_angle gelten."""
        key = round(float(max_angle_deg), 3)
        if key in self._angle_graph_cache:
            return self._angle_graph_cache[key]
        pairs, _ = self.adjacency_pairs
        Ne = self.edge_normals
        ang = np.degrees(np.arccos(np.clip(np.einsum("ij,ij->i", Ne[pairs[:, 0]], Ne[pairs[:, 1]]), -1, 1)))
        m = ang < max_angle_deg
        p = pairs[m]
        nF = self.n_faces
        G = sparse.csr_matrix((np.ones(len(p), np.int8), (p[:, 0], p[:, 1])), shape=(nF, nF))
        _, labels = connected_components(G, directed=False)
        if len(self._angle_graph_cache) > 6:
            self._angle_graph_cache.clear()
        self._angle_graph_cache[key] = labels
        return labels

    def components_of(self, face_ids):
        """Zusammenhangskomponenten einer Teilmenge. Rückgabe: Liste von Index-Arrays (groß -> klein)."""
        face_ids = np.asarray(face_ids, dtype=np.int64)
        if len(face_ids) == 0:
            return []
        A = self.adjacency[face_ids][:, face_ids]
        n, lab = connected_components(A, directed=False)
        comps = [face_ids[lab == i] for i in range(n)]
        comps.sort(key=lambda c: -len(c))
        return comps

    def component_containing(self, face_ids, seed_ids):
        face_ids = np.asarray(face_ids, dtype=np.int64)
        if len(face_ids) == 0:
            return face_ids
        A = self.adjacency[face_ids][:, face_ids]
        _, lab = connected_components(A, directed=False)
        sorter = np.argsort(face_ids)
        fs = face_ids[sorter]
        pos = np.clip(np.searchsorted(fs, seed_ids), 0, len(fs) - 1)
        ok = fs[pos] == seed_ids
        if not ok.any():
            return face_ids[:0]
        seed_local = sorter[pos[ok]]
        wanted = np.unique(lab[seed_local])
        return face_ids[np.isin(lab, wanted)]

    def grow(self, mask, rings=1):
        A = self.adjacency
        m = mask.astype(np.int8)
        for _ in range(rings):
            m = ((A @ m) > 0).astype(np.int8) | m
        return m.astype(bool)

    def shrink(self, mask, rings=1):
        return ~self.grow(~mask, rings)

    # ------------------------------------------------------------ Teilnetze
    def submesh(self, face_ids):
        F = self.F[face_ids]
        used = np.unique(F)
        remap = -np.ones(self.n_verts, dtype=np.int64)
        remap[used] = np.arange(len(used))
        return self.V[used], remap[F], used

    def boundary_loops(self, face_ids=None):
        """Randschleifen (Vertex-Indizes, geordnet) einer Flächenmenge."""
        F = self.F if face_ids is None else self.F[face_ids]
        return boundary_loops_of(F, self.n_verts)

    def to_pyvista(self):
        import pyvista as pv
        faces = np.hstack([np.full((len(self.F), 1), 3, dtype=np.int64), self.F]).ravel()
        return pv.PolyData(self.V.copy(), faces)

    def transformed(self, T):
        V = self.V @ T[:3, :3].T + T[:3, 3]
        fills = []
        for fl in self.fills:
            g = dict(fl)
            if g.get("rim") is not None:
                g["rim"] = [np.asarray(r, float) @ T[:3, :3].T + T[:3, 3] for r in g["rim"]]
            if g.get("plane") is not None:
                o, n = g["plane"]
                g["plane"] = (T[:3, :3] @ np.asarray(o) + T[:3, 3], T[:3, :3] @ np.asarray(n))
            fills.append(g)
        return MeshModel(V, self.F, self.synthetic, fills)

    def with_vertices(self, V):
        """Gleiche Topologie, neue Punktlagen."""
        return MeshModel(V, self.F, self.synthetic, self.fills)


def boundary_loops_of(F, n_verts):
    e = F[:, [0, 1, 1, 2, 2, 0]].reshape(-1, 2)
    es = np.sort(e, axis=1)
    key = es[:, 0] * n_verts + es[:, 1]
    uniq, inv, cnt = np.unique(key, return_inverse=True, return_counts=True)
    bmask = cnt[inv] == 1
    be = e[bmask]
    if len(be) == 0:
        return []
    nxt = {}
    for a, b in be:
        nxt.setdefault(int(a), []).append(int(b))
    loops = []
    used = set()
    for a0, b0 in be:
        a0 = int(a0)
        b0 = int(b0)
        if (a0, b0) in used:
            continue
        loop = [a0]
        used.add((a0, b0))
        cur = b0
        guard = 0
        while cur != a0 and guard < len(be) + 5:
            loop.append(cur)
            cands = [c for c in nxt.get(cur, []) if (cur, c) not in used]
            if not cands:
                break
            c = cands[0]
            used.add((cur, c))
            cur = c
            guard += 1
        if len(loop) >= 3:
            loops.append(np.array(loop, dtype=np.int64))
    loops.sort(key=lambda l: -len(l))
    return loops


def slice_mesh(V, F, origin, normal, gap_tol=None):
    """Schnitt Netz/Ebene -> Liste von (Punkte, geschlossen)-Polylinien."""
    origin = np.asarray(origin, float)
    normal = np.asarray(normal, float)
    s = (V - origin) @ normal
    scale = np.abs(s).max() + 1e-12
    s = np.where(np.abs(s) < 1e-9 * scale, 1e-9 * scale, s)
    sf = s[F]
    pos = sf > 0
    npos = pos.sum(1)
    cut = (npos == 1) | (npos == 2)
    Fc = F[cut]
    sc = sf[cut]
    pc = pos[cut]
    if len(Fc) == 0:
        return []
    nV = len(V)
    seg_keys = []
    pts = {}
    for k in range(3):
        a = Fc[:, k]
        b = Fc[:, (k + 1) % 3]
        sa = sc[:, k]
        sb = sc[:, (k + 1) % 3]
        crosses = pc[:, k] != pc[:, (k + 1) % 3]
        t = sa / (sa - sb + 1e-300)
        P = V[a] + t[:, None] * (V[b] - V[a])
        key = np.minimum(a, b) * nV + np.maximum(a, b)
        seg_keys.append(np.where(crosses, key, -1))
        for kk, pp, cc in zip(key[crosses], P[crosses], np.ones(crosses.sum())):
            pts[int(kk)] = pp
    seg_keys = np.stack(seg_keys, axis=1)
    segs = []
    for row in seg_keys:
        r = row[row >= 0]
        if len(r) == 2:
            segs.append((int(r[0]), int(r[1])))
    adj = {}
    for a, b in segs:
        adj.setdefault(a, []).append(b)
        adj.setdefault(b, []).append(a)
    visited = set()
    polylines = []
    # offene Ketten zuerst (Endpunkte mit Grad 1)
    starts = [k for k, v in adj.items() if len(v) == 1] + list(adj.keys())
    for st in starts:
        if st in visited:
            continue
        chain = [st]
        visited.add(st)
        cur = st
        while True:
            nb = [n for n in adj[cur] if n not in visited]
            if not nb:
                break
            cur = nb[0]
            visited.add(cur)
            chain.append(cur)
        closed = len(chain) > 2 and st in adj.get(chain[-1], [])
        if len(chain) >= 2:
            polylines.append((np.array([pts[c] for c in chain]), closed))
    if gap_tol is None:
        e = np.linalg.norm(V[F[:, 1]] - V[F[:, 0]], axis=1)
        gap_tol = 4.0 * float(np.median(e))
    polylines = join_polylines(polylines, gap_tol)
    polylines.sort(key=lambda pc: -len(pc[0]))
    return polylines


def join_polylines(polylines, gap_tol):
    """Verbindet offene Ketten mit nahe beieinander liegenden Enden (Risse im Scan)."""
    closed = [pc for pc in polylines if pc[1]]
    open_ = [pc[0] for pc in polylines if not pc[1]]
    changed = True
    while changed and open_:
        changed = False
        for i in range(len(open_)):
            a = open_[i]
            # eigene Enden schließen?
            best = None
            for j in range(len(open_)):
                if j == i:
                    continue
                b = open_[j]
                for ra in (False, True):
                    for rb in (False, True):
                        ea = a[0] if ra else a[-1]
                        sb = b[-1] if rb else b[0]
                        d = np.linalg.norm(ea - sb)
                        if d < gap_tol and (best is None or d < best[0]):
                            best = (d, j, ra, rb)
            if best is not None:
                _, j, ra, rb = best
                aa = a[::-1] if ra else a
                bb = open_[j][::-1] if rb else open_[j]
                merged = np.vstack([aa, bb])
                open_ = [o for k, o in enumerate(open_) if k not in (i, j)] + [merged]
                changed = True
                break
    out = list(closed)
    for o in open_:
        L = np.linalg.norm(np.diff(o, axis=0), axis=1).sum()
        if len(o) > 3 and np.linalg.norm(o[0] - o[-1]) < gap_tol and L > 4 * gap_tol:
            out.append((o, True))
        else:
            out.append((o, False))
    return out
