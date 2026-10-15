"""Kantenrundungen und Splitter.

Scans runden scharfe Kanten ab (Messfleck, Glättung): entlang jeder Kante entsteht ein schmaler
Zylinderstreifen mit 0,2–0,5 mm Radius, an Innenecken ebenso. Als eigene Flächen machen diese
Streifen die Topologie brüchig (winzige Flächen, Ecken mit vier und mehr Flächen). Hier werden
sie als Kantenrundung zwischen zwei Ebenen erkannt und aus den Flächen genommen – die Kante wird
scharf; Radius und Nachbarflächen bleiben vermerkt (für eine spätere echte Verrundung).

Splitter: sehr kleine Flächen-Features (Bruchteile eines mm²) ohne Bezug zu Nachbarn.
"""
from __future__ import annotations

import math

import numpy as np

from .geom_utils import orthonormal_frame, unit
from .i18n import tr


def _faces(f, nF):
    ids = np.asarray(f.face_ids, np.int64)
    return ids[(ids >= 0) & (ids < nF)]


def detect_edge_blends(model, features, tol, r_max=1.0, near=1.5):
    """Zylinder-Features, die zwei Ebenen tangential verbinden. Rückgabe: Liste dict(id, r,
    planes=(id1, id2), convex)."""
    from scipy.spatial import cKDTree
    nF = model.n_faces
    planes = [f for f in features if f.kind == "plane" and f.role == "surface" and len(f.face_ids) >= 20]
    if len(planes) < 2:
        return []
    ptrees = {f.id: cKDTree(model.centroids[_faces(f, nF)]) for f in planes}
    out = []
    for f in features:
        if f.kind != "cylinder" or f.role != "surface":
            continue
        r = float(f.params["radius"])
        if r > r_max:
            continue
        ids = _faces(f, nF)
        if len(ids) < 3:
            continue
        a = unit(f.params["axis"])
        q = np.asarray(f.params["point"], float)
        C = model.centroids[ids]
        sub = C if len(C) <= 200 else C[np.linspace(0, len(C) - 1, 200).astype(int)]
        cand = []
        for p in planes:
            n = unit(p.params["normal"])
            if abs(float(n @ a)) > math.sin(math.radians(8)):
                continue                       # Achse muss in der Ebene liegen
            d, _ = ptrees[p.id].query(sub, distance_upper_bound=near)
            if np.isfinite(d).mean() < 0.3:
                continue
            o = np.asarray(p.params["origin"], float)
            dist = abs(float((q - o) @ n))     # Abstand Achse–Ebene = r bei Tangentialität
            if abs(dist - r) > max(0.5 * r, 3 * tol, 0.08):
                continue
            cand.append((p, n))
        if len(cand) < 2:
            # schmaler Streifen mit kleinem Radius entlang einer Kante (z. B. an einer Bohrung):
            # deckt nur einen kleinen Winkel ab, ist aber lang
            if r <= 0.5 * r_max:
                x, y, _ = orthonormal_frame(a)
                rel = C - q
                ph = np.arctan2(rel @ y, rel @ x)
                bins = len(np.unique(np.floor((ph + math.pi) / (2 * math.pi) * 36).astype(int) % 36))
                L = float(np.ptp(rel @ a))
                if bins <= 15 and L >= 4 * r:
                    rad = rel - np.outer(rel @ a, a)
                    conv = float(np.mean(np.einsum("ij,ij->i", model.normals[ids], unit(rad)))) > 0
                    out.append({"id": f.id, "r": r, "planes": tuple(p.id for p, _ in cand) + (-1,) * (2 - len(cand)),
                                "convex": conv, "name": f.name, "area": float(model.areas[ids].sum())})
            continue
        best = None
        for i in range(len(cand)):
            for j in range(i + 1, len(cand)):
                n1, n2 = cand[i][1], cand[j][1]
                ang = math.degrees(math.acos(min(1.0, abs(float(n1 @ n2)))))
                if ang < 20:
                    continue
                if best is None or ang > best[0]:
                    best = (ang, cand[i][0], cand[j][0])
        if best is None:
            continue
        rel = C - q
        rad = rel - np.outer(rel @ a, a)
        conv = float(np.mean(np.einsum("ij,ij->i", model.normals[ids], unit(rad)))) > 0
        out.append({"id": f.id, "r": r, "planes": (best[1].id, best[2].id), "convex": conv,
                    "name": f.name, "area": float(model.areas[ids].sum())})
    return out


def small_fragments(model, features, total_area, min_area=3.0, rel=5e-4):
    """Sehr kleine Flächen-Features (keine Körper-Features)."""
    nF = model.n_faces
    thr = max(min_area, rel * total_area)
    out = []
    for f in features:
        if f.role != "surface" or f.kind == "freeform" or f.params.get("fixed") or f.params.get("aux"):
            continue
        A = float(model.areas[_faces(f, nF)].sum())
        if A < thr:
            out.append((f, A))
    return out


def cleanup(doc, r_max=1.0):
    """Kantenrundungen und Splitter aus den Flächen nehmen. Rückgabe: Berichtszeilen."""
    m = doc.model
    out = []
    bl = detect_edge_blends(m, doc.features, doc.fit_tol, r_max=r_max)
    rm = {b["id"] for b in bl}
    names = {f.id: f.name for f in doc.features}
    if bl:
        conv = [b for b in bl if b["convex"]]
        conc = [b for b in bl if not b["convex"]]
        rs = sorted(b["r"] for b in bl)
        out.append(tr(f"{len(bl)} Kantenrundungen als scharfe Kanten übernommen ({len(conv)} außen, {len(conc)} innen; "
                      f"R {rs[0]:.2f}…{rs[-1]:.2f} mm)",
                      f"{len(bl)} edge blends turned into sharp edges ({len(conv)} outer, {len(conc)} inner; "
                      f"R {rs[0]:.2f}…{rs[-1]:.2f} mm)"))
        doc.tol["blends"] = [{"r": float(b["r"]), "planes": [int(b["planes"][0]), int(b["planes"][1])],
                              "convex": bool(b["convex"])} for b in bl]
        for b in bl:
            out.append(f"  {b['name']}: R {b['r']:.2f} " + tr("zwischen", "between")
                       + f" {names.get(b['planes'][0])} / {names.get(b['planes'][1])}")
    total = float(m.areas[~m.synthetic].sum())
    frag = [(f, A) for f, A in small_fragments(m, doc.features, total) if f.id not in rm]
    if frag:
        thr = max(3.0, 5e-4 * total)
        out.append(tr(f"{len(frag)} Splitterflächen (< {thr:.1f} mm²) entfernt: ",
                      f"{len(frag)} fragment surfaces (< {thr:.1f} mm²) removed: ")
                   + ", ".join(f.name for f, _ in frag))
        rm |= {f.id for f, _ in frag}
    doc.features = [f for f in doc.features if f.id not in rm]
    return out or [tr("Keine Kantenrundungen oder Splitter.", "No edge blends or fragments.")]
