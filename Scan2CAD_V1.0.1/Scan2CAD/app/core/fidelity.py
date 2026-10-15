"""Abgleich Features ↔ STEP-Körper.

Der Export baut den Körper aus der Scan-Topologie (brep_seg) und zieht Körper-Features per
Boolean ab bzw. an. Dabei kann ein Feature verloren gehen (Boolean ausgelassen, als Delle
eingeebnet, vom Nachbarträger geschluckt) oder durch eine Ersatzfläche dargestellt werden
(B-Spline durch die Scan-Grenze, Dreiecksfacetten). Hier wird für jedes Feature geprüft:

* liegt der Körper dort, wo die Scanpunkte des Features liegen (Abstand ≤ Schwelle)?
* hat die nächstgelegene B-Rep-Fläche den Typ des Features (Ebene bleibt Ebene …)?

Ergebnis: je Feature ein Status (ok / Freiform / Facetten / anderer Typ / teilweise / fehlt),
Kennzahlen, eine Zusammenfassung für den Exportbericht und je Dreieck der Körper-Tesselierung
die Herkunft (exakt, Freiform, Facette) zum Einfärben der Vorschau.
"""
from __future__ import annotations

import numpy as np

from .i18n import tr

EXPECTED = {
    "plane": {"plane"},
    "cylinder": {"cylinder", "cone"},
    "cone": {"cone", "cylinder"},
    "sphere": {"sphere"},
    "torus": {"torus"},
    "ellipsoid": {"bspline", "revolution", "other", "sphere"},
    "freeform": {"bspline", "plane", "cylinder", "cone", "sphere", "torus"},
    "hole": {"cylinder", "cone", "plane", "sphere"},
    "revolve": {"cylinder", "cone", "plane", "torus", "sphere", "revolution"},
    "extrusion": {"plane", "cylinder", "bspline", "extrusion"},
}
ANALYTIC = {"plane", "cylinder", "cone", "sphere", "torus"}

# Herkunftsklassen je B-Rep-Fläche
SRC_EXACT, SRC_FREEFORM, SRC_FACET = 0, 1, 2

STATUS_TEXT = {
    "ok": ("exakt", "exact"),
    "freeform": ("als Freiform", "as freeform"),
    "facets": ("als Dreiecke", "as triangles"),
    "type": ("anderer Flächentyp", "other surface type"),
    "partial": ("nur teilweise", "only partly"),
    "missing": ("fehlt", "missing"),
}


def face_sources(info, mean_tri_area):
    """Herkunft je B-Rep-Fläche aus (Typ, Kantenzahl, Fläche)."""
    out = np.zeros(len(info), np.int64)
    lim = max(25.0 * mean_tri_area, 1e-3)
    for i, (typ, ne, ar) in enumerate(info):
        if typ == "plane" and ne <= 4 and ar < lim:
            out[i] = SRC_FACET
        elif typ in ANALYTIC:
            out[i] = SRC_EXACT
        else:
            out[i] = SRC_FREEFORM
    return out


def check(model, features, body, tol=0.05, max_samples=300, defl=None, tess=None):
    """Abgleich. Rückgabe dict(rows, lines, tess=(V, F, fidx, info), src (je Dreieck),
    off_area_frac)."""
    from . import occ_build as ob
    from .preprocess import vf_to_pd
    diag = float(np.linalg.norm(model.bbox[1] - model.bbox[0]))
    if tess is None:
        defl = defl or max(0.02, 0.0007 * diag)
        tess = ob.tessellate_faces(body, defl, 0.3)
    V, F, fidx, info = tess
    if len(F) == 0:
        return {"rows": [], "lines": [tr("Abgleich: Körper leer", "check: body empty")], "tess": tess,
                "src": np.zeros(0, np.int64), "off_area_frac": 1.0, "spots": []}
    real = ~model.synthetic if hasattr(model, "synthetic") else np.ones(model.n_faces, bool)
    mean_a = float(model.areas[real].mean()) if real.any() else float(model.areas.mean())
    fsrc = face_sources(info, mean_a)
    src = fsrc[fidx]
    ftyp = np.array([t for t, _, _ in info], dtype=object)
    pd = vf_to_pd(V, F)

    def closest(P):
        P = np.asarray(P, float)
        C, Q = pd.find_closest_cell(P, return_closest_point=True)
        return np.linalg.norm(P - Q, axis=1), np.asarray(C, np.int64)

    thr0 = max(0.15, 3.0 * tol)
    rows = []
    rng = np.random.default_rng(0)
    for f in features:
        if f.role not in ("surface", "sub", "add") or f.params.get("aux"):
            continue
        ids = np.asarray(f.face_ids, np.int64)
        ids = ids[(ids >= 0) & (ids < model.n_faces)]
        ids = ids[real[ids]]
        if len(ids) < 3:
            continue
        if len(ids) > max_samples:
            ids = rng.choice(ids, max_samples, replace=False)
        P = model.centroids[ids]
        D, C = closest(P)
        st = f.stats or {}
        thr = max(thr0, 1.5 * float(st.get("p95", 0.0) or 0.0))
        thr = min(thr, max(thr0, 0.5))
        near = D <= thr
        in_frac = float(near.mean())
        fi = fidx[C]
        typ = ftyp[fi]
        exp_ = EXPECTED.get(f.kind, ANALYTIC | {"bspline"})
        sel = D <= 3 * thr
        if sel.sum() >= 3:
            type_frac = float(np.mean([t in exp_ for t in typ[sel]]))
            facet_frac = float(np.mean(fsrc[fi[sel]] == SRC_FACET))
            bs_frac = float(np.mean(fsrc[fi[sel]] == SRC_FREEFORM))
        else:
            type_frac = facet_frac = bs_frac = 0.0
        if in_frac < 0.3:
            status = "missing"
        elif in_frac < 0.75:
            status = "partial"
        elif facet_frac > 0.5:
            status = "facets"
        elif f.kind in ANALYTIC and bs_frac > 0.5:
            status = "freeform"
        elif type_frac < 0.5:
            status = "type"
        else:
            status = "ok"
        rows.append({"id": f.id, "name": f.name, "kind": f.kind, "role": f.role, "status": status,
                     "in_frac": in_frac, "median": float(np.median(D)), "p90": float(np.percentile(D, 90)),
                     "type_frac": type_frac, "facet_frac": facet_frac, "freeform_frac": bs_frac,
                     "area": float(model.areas[np.asarray(f.face_ids, np.int64)[
                         (np.asarray(f.face_ids) >= 0) & (np.asarray(f.face_ids) < model.n_faces)]].sum()),
                     "thr": thr})
    # Scanfläche, die der Körper nicht trifft (unabhängig von Features): je Scan-Dreieck der
    # Abstand zum Körper, zusammenhängende Bereiche mit Lage und Fläche
    import pyvista as pv
    allf = np.nonzero(real)[0]
    dist = np.abs(np.asarray(pv.PolyData(model.centroids[allf]).compute_implicit_distance(pd)
                             .point_data["implicit_distance"]))
    w = model.areas[allf]
    off_thr = 3 * thr0
    offm = dist > off_thr
    off = float(w[offm].sum() / max(w.sum(), 1e-12))
    spots = []
    if offm.any():
        for comp in model.components_of(allf[offm]):
            a = float(model.areas[comp].sum())
            if a < max(1.0, 20 * mean_a):
                continue
            dd = dist[np.searchsorted(allf, comp)]
            spots.append({"center": model.centroids[comp].mean(0), "area": a, "max": float(dd.max()),
                          "faces": comp})
        spots.sort(key=lambda x: -x["area"] * x["max"])
    ta = 0.5 * np.linalg.norm(np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]]), axis=1)
    src_frac = [float(ta[src == k].sum() / max(ta.sum(), 1e-12)) for k in (SRC_EXACT, SRC_FREEFORM, SRC_FACET)]
    lines = summary(rows, off, off_thr, spots)
    if src_frac[0] < 0.999:
        lines.insert(1 if lines else 0, tr(
            f"Körperoberfläche: {100 * src_frac[0]:.0f} % exakt, {100 * src_frac[1]:.0f} % Ersatz-Freiform, "
            f"{100 * src_frac[2]:.0f} % Scan-Dreiecke",
            f"solid surface: {100 * src_frac[0]:.0f} % exact, {100 * src_frac[1]:.0f} % substitute freeform, "
            f"{100 * src_frac[2]:.0f} % scan triangles"))
    return {"rows": rows, "lines": lines, "tess": tess, "src": src, "off_area_frac": off, "spots": spots,
            "src_frac": src_frac}


def summary(rows, off_frac, off_thr, spots=()):
    if not rows and not spots:
        return []
    cnt = {}
    order = ["ok", "freeform", "facets", "type", "partial", "missing"]
    for r in rows:
        cnt[r["status"]] = cnt.get(r["status"], 0) + 1
    head = tr("Abgleich Features ↔ STEP: ", "check features ↔ STEP: ") + ", ".join(
        f"{cnt[k]} {tr(*STATUS_TEXT[k])}" for k in order if cnt.get(k))
    out = [head] if rows else []
    bad = [r for r in rows if r["status"] != "ok"]
    bad.sort(key=lambda r: (order.index(r["status"]) * -1, -r["area"]))
    for r in bad[:25]:
        out.append(f"  {'⚠ ' if r['status'] in ('missing', 'partial') else ''}{r['name']}: {tr(*STATUS_TEXT[r['status']])}"
                   + tr(f" ({100 * r['in_frac']:.0f} % der Scanpunkte am Körper, Median {r['median']:.2f} mm)",
                        f" ({100 * r['in_frac']:.0f} % of scan points on the body, median {r['median']:.2f} mm)"))
    if len(bad) > 25:
        out.append(tr(f"  … und {len(bad) - 25} weitere", f"  … and {len(bad) - 25} more"))
    out.append(tr(f"Scanfläche weiter als {off_thr:.2f} mm vom Körper: {100 * off_frac:.1f} %",
                  f"scan area farther than {off_thr:.2f} mm from the body: {100 * off_frac:.1f} %"))
    for sp in list(spots)[:10]:
        c = sp["center"]
        out.append(tr(f"  ⚠ Scanbereich ohne Entsprechung bei ({c[0]:.1f}, {c[1]:.1f}, {c[2]:.1f}): "
                      f"{sp['area']:.1f} mm², bis {sp['max']:.2f} mm Abstand",
                      f"  ⚠ scan area without counterpart at ({c[0]:.1f}, {c[1]:.1f}, {c[2]:.1f}): "
                      f"{sp['area']:.1f} mm², up to {sp['max']:.2f} mm off"))
    return out
