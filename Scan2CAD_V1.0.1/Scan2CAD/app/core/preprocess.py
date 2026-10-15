"""Laden, Reparieren, Glätten und toleranzbasiertes Vereinfachen von Netzen."""
from __future__ import annotations

import numpy as np

from .i18n import tr
from .mesh_model import MeshModel


def _pd_to_vf(pd):
    pd = pd.triangulate()
    if hasattr(pd, "regular_faces"):
        F = np.asarray(pd.regular_faces, dtype=np.int64)
    else:
        F = pd.faces.reshape(-1, 4)[:, 1:]
    return np.asarray(pd.points, dtype=np.float64), F


def vf_to_pd(V, F):
    import pyvista as pv
    faces = np.hstack([np.full((len(F), 1), 3, dtype=np.int64), F]).ravel()
    return pv.PolyData(np.asarray(V, float), faces)


def load_mesh(path):
    import pyvista as pv
    pd = pv.read(str(path))
    if not isinstance(pd, pv.PolyData):
        pd = pd.extract_surface()
    pd = pd.triangulate().clean(tolerance=0.0)
    V, F = _pd_to_vf(pd)
    # degenerierte Dreiecke entfernen
    m = MeshModel(V, F)
    keep = m.areas > 1e-14
    keep &= (F[:, 0] != F[:, 1]) & (F[:, 1] != F[:, 2]) & (F[:, 0] != F[:, 2])
    if not keep.all():
        V, F = _compact(V, F[keep])
    return MeshModel(V, F)


def _compact(V, F):
    used = np.unique(F)
    remap = -np.ones(len(V), np.int64)
    remap[used] = np.arange(len(used))
    return V[used], remap[F]


def save_mesh(model, path):
    pd = vf_to_pd(model.V, model.F)
    pd.save(str(path))


def repair(model, merge_tol=0.0, min_component_frac=0.01, keep_largest=False,
           fill_holes=0.0, fix_normals=True):
    pd = vf_to_pd(model.V, model.F)
    report = []
    n0 = pd.n_cells
    if merge_tol > 0:
        pd = pd.clean(tolerance=float(merge_tol), absolute=True)
        report.append(tr(f"Punkte verschmolzen (Toleranz {merge_tol} mm)", f"points merged (tolerance {merge_tol} mm)"))
    conn = pd.connectivity("all")
    rid = np.asarray(conn.cell_data["RegionId"])
    counts = np.bincount(rid)
    if keep_largest:
        keep_ids = [int(np.argmax(counts))]
    else:
        keep_ids = np.nonzero(counts >= min_component_frac * counts.max())[0].tolist()
    if len(keep_ids) < len(counts):
        report.append(tr(f"{len(counts) - len(keep_ids)} kleine Teilstücke entfernt (von {len(counts)})", f"{len(counts) - len(keep_ids)} small pieces removed (of {len(counts)})"))
        mask = np.isin(rid, keep_ids)
        pd = conn.extract_cells(np.nonzero(mask)[0]).extract_surface()
    if fill_holes > 0:
        before = pd.n_cells
        pd = pd.fill_holes(float(fill_holes))
        report.append(tr(f"Löcher bis {fill_holes} mm gefüllt (+{pd.n_cells - before} Dreiecke)", f"holes up to {fill_holes} mm filled"))
    pd = pd.triangulate().clean(tolerance=0.0)
    if fix_normals:
        try:
            closed = pd.n_open_edges == 0
            pd = pd.compute_normals(cell_normals=True, point_normals=False, consistent_normals=True,
                                    auto_orient_normals=closed, split_vertices=False)
            report.append(tr("Normalen konsistent orientiert", "normals oriented consistently") + (tr(" (nach außen)", " (outward)") if closed else ""))
        except Exception as e:  # noqa: BLE001
            report.append(tr(f"Normalen-Korrektur übersprungen: {e}", f"normal fix skipped: {e}"))
    V, F = _pd_to_vf(pd)
    m = MeshModel(V, F)
    if model.synthetic.any() or model.fills:
        from .mesh_ops import transfer_model_state
        m = transfer_model_state(model, m)
    report.append(tr(f"Dreiecke: {n0} → {m.n_faces}; offene Kanten: {m.boundary_edge_count}",
                     f"triangles: {n0} → {m.n_faces}; open edges: {m.boundary_edge_count}"))
    return m, report


def smooth(model, iterations=10, pass_band=0.1, feature_angle=None):
    pd = vf_to_pd(model.V, model.F)
    kw = dict(n_iter=int(iterations), pass_band=float(pass_band), boundary_smoothing=False)
    if feature_angle:
        kw.update(feature_smoothing=False, edge_angle=float(feature_angle), feature_angle=float(feature_angle))
    try:
        pd = pd.smooth_taubin(**kw)
    except TypeError:
        pd = pd.smooth_taubin(n_iter=int(iterations), pass_band=float(pass_band))
    V, F = _pd_to_vf(pd)
    if len(F) == model.n_faces:
        return model.with_vertices(V)
    from .mesh_ops import transfer_model_state
    return transfer_model_state(model, MeshModel(V, F))


def deviation(src_points, target_model, max_points=200000, seed=0):
    """Abstand (vorzeichenbehaftet) von Punkten zur Zieloberfläche (VTK, C++)."""
    import pyvista as pv
    P = np.asarray(src_points, float)
    idx = None
    if len(P) > max_points:
        idx = np.random.default_rng(seed).choice(len(P), max_points, replace=False)
        P = P[idx]
    tgt = vf_to_pd(target_model.V, target_model.F)
    src = pv.PolyData(P)
    out = src.compute_implicit_distance(tgt)
    return np.asarray(out.point_data["implicit_distance"]), idx


def _decimate(V, F, reduction):
    try:
        import fast_simplification
        V2, F2 = fast_simplification.simplify(V.astype(np.float64), F.astype(np.int64),
                                              target_reduction=float(reduction), agg=7)
        V2, F2 = np.asarray(V2, float), np.asarray(F2, np.int64)
    except ImportError:
        pd = vf_to_pd(V, F).decimate(float(reduction))
        V2, F2 = _pd_to_vf(pd)
    return _clean_topology(V2, F2)


def _clean_topology(V, F):
    """Dezimierung kann vereinzelt gefaltete Doppeldreiecke ('Flossen') und gekippte Dreiecke
    erzeugen -> nur wenn nötig mit repair_manifold bereinigen."""
    from .mesh_ops import manifold_report, repair_manifold
    m = MeshModel(V, F)
    rep = manifold_report(m)
    if rep["nonmanifold_edges"] or rep["flipped_edges"]:
        m, _, _ = repair_manifold(m)
    # winzige Löcher, die die Dezimierung hinterlassen kann (≤ 8 Randkanten), direkt schließen
    if m.boundary_edge_count:
        add = []
        for loop in m.boundary_loops():
            if 3 <= len(loop) <= 8:
                for i in range(1, len(loop) - 1):
                    add.append((loop[0], loop[i + 1], loop[i]))
        if add:
            m = MeshModel(m.V, np.vstack([m.F, np.asarray(add, np.int64)]))
    return m.V, m.F


def simplify_to_tolerance(model, tol, max_faces=None, min_faces=500, iters=8, progress=None,
                          eval_points=20000):
    """Größtmögliche Reduktion, bei der die Abweichung (beidseitig) <= tol bleibt.

    Kriterium (auf je bis zu 60k Stichprobenpunkten):
        99.9-Perzentil |Abstand vereinfachte Ecken -> Original|  <= tol  und
        99.0-Perzentil |Abstand Originalecken -> vereinfacht|    <= tol.
    Perzentile statt Maximum, weil das Scanrauschen selbst in die Abstände eingeht
    und einzelne Ausreißer beim Vereinfachen verschwinden. Faustregel: tol >= 3x Rauschen.
    """
    V, F = model.V, model.F
    n0 = len(F)
    lo, hi = 0.0, 1.0 - min_faces / max(n0, 1)
    if hi <= 0:
        return model, {"faces": n0, "dev_max": 0.0, "dev_p999": 0.0, "reduction": 0.0}
    best = (model, 0.0, 0.0, 0.0)

    def trial(r):
        V2, F2 = _decimate(V, F, r)
        m2 = MeshModel(V2, F2)
        d_fwd, _ = deviation(m2.V, model, max_points=eval_points)
        d_bwd, _ = deviation(model.V, m2, max_points=eval_points)
        dmax = float(np.percentile(np.abs(d_fwd), 99.9)) if len(d_fwd) else 0.0
        p999 = float(np.percentile(np.abs(d_bwd), 99.0)) if len(d_bwd) else 0.0
        return m2, dmax, p999

    r_min = 0.0
    if max_faces and n0 > max_faces:
        # Mindestreduktion für die Höchstzahl zuerst: hält schon sie die Toleranz nicht ein,
        # gewinnt die Höchstzahl (keine nutzlosen Versuche mit zu vielen Dreiecken)
        r_min = min(hi, 1.0 - max_faces / n0 + 0.002)
        m2, dmax, p999 = trial(r_min)
        best = (m2, r_min, dmax, p999)
        lo = r_min
        if max(dmax, p999) > tol:
            iters = 0
    for i in range(iters):
        r = 0.5 * (lo + hi) if (i or r_min > 0) else min(hi, 0.9)
        m2, dmax, p999 = trial(r)
        ok = max(dmax, p999) <= tol
        if ok:
            lo = r
            if r >= best[1]:
                best = (m2, r, dmax, p999)
        else:
            hi = r
        if progress:
            progress((i + 1) / max(iters, 1))
        if hi - lo < 0.002:
            break
    m, r, dmax, p999 = best
    if m is not model and (model.synthetic.any() or model.fills):
        from .mesh_ops import transfer_model_state
        m = transfer_model_state(model, m)
    return m, {"faces": m.n_faces, "faces_before": n0, "dev_max": dmax, "dev_p999": p999,
               "reduction": r}


def decimate_to_count(model, target_faces):
    r = 1.0 - target_faces / max(model.n_faces, 1)
    if r <= 0:
        return model
    V2, F2 = _decimate(model.V, model.F, r)
    m = MeshModel(V2, F2)
    if model.synthetic.any() or model.fills:
        from .mesh_ops import transfer_model_state
        m = transfer_model_state(model, m)
    return m


# ============================================================ CAD-Export erkennen, gleichmäßig unterteilen
def _edge_dihedral(model):
    """Innere Kanten: (Flächenpaar, Knickwinkel [rad], Länge, mittlere Dreieckshöhe zur Kante)."""
    F, V = model.F, model.V
    nF = len(F)
    E = np.vstack([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
    opp = np.r_[F[:, 2], F[:, 0], F[:, 1]]
    fid = np.r_[np.arange(nF), np.arange(nF), np.arange(nF)]
    Es = np.sort(E, axis=1)
    key = Es[:, 0] * len(V) + Es[:, 1]
    order = np.argsort(key, kind="stable")
    ks = key[order]
    same = ks[1:] == ks[:-1]
    i1, i2 = order[:-1][same], order[1:][same]
    f1, f2 = fid[i1], fid[i2]
    cosang = np.einsum("ij,ij->i", model.normals[f1], model.normals[f2])
    ang = np.arccos(np.clip(cosang, -1, 1))
    a, b = V[Es[i1, 0]], V[Es[i1, 1]]
    L = np.linalg.norm(b - a, axis=1)
    h1 = 2 * model.areas[f1] / np.maximum(L, 1e-300)
    h2 = 2 * model.areas[f2] / np.maximum(L, 1e-300)
    return ang, L, 0.5 * (h1 + h2)


def mesh_kind(model):
    """CAD-Export (exakt ebene Facetten, Punkte auf der wahren Fläche, sehr ungleich große
    Dreiecke) oder Scan (gleichmäßig, verrauscht)? Kennzeichen eines CAD-Exports: viele exakt
    koplanare Nachbardreiecke. Bei CAD-Netzen wird außerdem der Facettenfehler (Sehnenabstand
    gekrümmter Flächen, s ≈ Breite·Knickwinkel/8) abgeschätzt – er ersetzt das Rauschen."""
    ang, L, h = _edge_dihedral(model)
    if len(ang) == 0:
        return {"kind": "scan", "coplanar": 0.0, "sagitta": 0.0}
    coplanar = float(np.mean(ang < 1e-5))
    el = np.linalg.norm(model.V[model.F[:, 1]] - model.V[model.F[:, 0]], axis=1)
    ratio = float(np.percentile(el, 99) / max(np.median(el), 1e-12))
    kind = "cad" if coplanar > 0.05 else "scan"
    smooth = (ang > np.radians(0.3)) & (ang < np.radians(40))
    sag = float(np.percentile(h[smooth] * ang[smooth] / 8.0, 90)) if smooth.sum() > 10 else 0.0
    return {"kind": kind, "coplanar": coplanar, "edge_ratio": ratio, "sagitta": sag,
            "max_edge": float(el.max()), "median_edge": float(np.median(el))}


def planar_patch_faces(model, smooth_deg=40.0):
    """Dreiecke, die zu echten ebenen Flächen eines CAD-Exports gehören (nicht zu den ebenen
    Facetten einer gekrümmten Fläche): zusammenhängende, exakt koplanare Bereiche, deren Rand nicht
    überwiegend aus flachen Knicken (< smooth_deg, Nachbarfacetten derselben Rundung) besteht."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    F, V = model.F, model.V
    nF = len(F)
    E = np.vstack([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
    fid = np.r_[np.arange(nF), np.arange(nF), np.arange(nF)]
    Es = np.sort(E, axis=1)
    key = Es[:, 0] * len(V) + Es[:, 1]
    order = np.argsort(key, kind="stable")
    ks = key[order]
    same = ks[1:] == ks[:-1]
    i1, i2 = order[:-1][same], order[1:][same]
    f1, f2 = fid[i1], fid[i2]
    dots = np.einsum("ij,ij->i", model.normals[f1], model.normals[f2])
    cop = dots > 1.0 - 1e-10
    G = coo_matrix((np.ones(int(cop.sum())), (f1[cop], f2[cop])), shape=(nF, nF))
    _, lab = connected_components(G, directed=False)
    L = np.linalg.norm(V[Es[i1, 1]] - V[Es[i1, 0]], axis=1)
    smooth = ~cop & (dots > np.cos(np.radians(smooth_deg)))
    rim_len = np.zeros(lab.max() + 1)
    sm_len = np.zeros(lab.max() + 1)
    for a, b in ((f1, f2), (f2, f1)):
        bnd = ~cop
        np.add.at(rim_len, lab[a[bnd]], L[bnd])
        np.add.at(sm_len, lab[a[bnd & smooth]], L[bnd & smooth])
    frac = sm_len / np.maximum(rim_len, 1e-300)
    return frac[lab] < 0.5


def subdivide_long_edges(model, h, max_faces=400000, max_rounds=14, only_coplanar=False):
    """Alle Kanten länger als h halbieren (konform: eine Kante wird für beide Nachbarn geteilt),
    bis keine mehr zu lang ist. Für grob triangulierte CAD-Exporte: gleichmäßige Abtastung für
    Erkennung und Flächenzuordnung; ebene Flächen bleiben exakt eben.
    only_coplanar: nur Kanten zwischen exakt koplanaren Dreiecken (und offene Randkanten) teilen –
    dann liegen auch alle neuen Punkte exakt auf der wahren Fläche (bei gekrümmten Facetten läge
    ein Kantenmittelpunkt auf der Sehne, also neben der Fläche)."""
    V = np.asarray(model.V, float)
    F = np.asarray(model.F, np.int64)
    syn = np.asarray(model.synthetic, bool)
    flat = planar_patch_faces(model) if only_coplanar else np.ones(len(F), bool)
    for _ in range(max_rounds):
        nV, nF = len(V), len(F)
        E = np.stack([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]], axis=1)      # (nF, 3, 2)
        Es = np.sort(E.reshape(-1, 2), axis=1)
        key = Es[:, 0] * nV + Es[:, 1]
        uk, first, inv = np.unique(key, return_index=True, return_inverse=True)
        ue = Es[first]
        Lu = np.linalg.norm(V[ue[:, 1]] - V[ue[:, 0]], axis=1)
        split = Lu > h
        if only_coplanar:
            fidx = np.repeat(np.arange(nF), 3)
            fmin = np.full(len(uk), nF, np.int64)
            fmax = np.full(len(uk), -1, np.int64)
            np.minimum.at(fmin, inv, fidx)
            np.maximum.at(fmax, inv, fidx)
            tri = V[F]
            cr = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
            nn = cr / np.maximum(np.linalg.norm(cr, axis=1), 1e-300)[:, None]
            # Kanten zwischen zwei echten Ebenen (koplanar oder Knick – die Schnittgerade ist exakt)
            split &= flat[fmin] & flat[np.maximum(fmax, 0)]
        if not split.any():
            break
        if nF + 3 * int(split.sum()) > max_faces:
            break
        mid = -np.ones(len(uk), np.int64)
        mid[split] = nV + np.arange(int(split.sum()))
        V = np.vstack([V, 0.5 * (V[ue[split, 0]] + V[ue[split, 1]])])
        M = mid[inv].reshape(nF, 3)                                            # Mittelpunkt je Kante k
        s = M >= 0
        cnt = s.sum(1)
        out_F, out_s, out_f = [F[cnt == 0]], [syn[cnt == 0]], [flat[cnt == 0]]
        # genau eine Kante: so drehen, dass sie Kante 0 (v0→v1) ist
        for r in range(3):
            sel = (cnt == 1) & s[:, r]
            if sel.any():
                f = np.roll(F[sel], -r, axis=1)
                m0 = M[sel, r]
                out_F += [np.c_[f[:, 0], m0, f[:, 2]], np.c_[m0, f[:, 1], f[:, 2]]]
                out_s += [syn[sel]] * 2
                out_f += [flat[sel]] * 2
        # zwei Kanten: so drehen, dass Kante 0 und 1 geteilt sind
        for r in range(3):
            sel = (cnt == 2) & s[:, r] & s[:, (r + 1) % 3]
            if sel.any():
                f = np.roll(F[sel], -r, axis=1)
                m01, m12 = M[sel, r], M[sel, (r + 1) % 3]
                out_F += [np.c_[m01, f[:, 1], m12], np.c_[f[:, 0], m01, m12], np.c_[f[:, 0], m12, f[:, 2]]]
                out_s += [syn[sel]] * 3
                out_f += [flat[sel]] * 3
        sel = cnt == 3
        if sel.any():
            f = F[sel]
            m01, m12, m20 = M[sel, 0], M[sel, 1], M[sel, 2]
            out_F += [np.c_[f[:, 0], m01, m20], np.c_[m01, f[:, 1], m12], np.c_[m20, m12, f[:, 2]],
                      np.c_[m01, m12, m20]]
            out_s += [syn[sel]] * 4
            out_f += [flat[sel]] * 4
        F = np.vstack(out_F)
        syn = np.concatenate(out_s)
        flat = np.concatenate(out_f)
    m = MeshModel(V, F, syn, [])
    if model.fills:
        from .mesh_ops import transfer_model_state
        m = transfer_model_state(model, m)
    return m
