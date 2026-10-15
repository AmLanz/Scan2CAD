"""Messen, Ausrichtung optimieren, ähnliche Features zusammenführen.

Messobjekte ("Measurable") sind kleine dicts:
  {"type": "plane", "o", "n"}          Ebene
  {"type": "axis",  "p", "d", "r"}      Achse (Zylinder/Kegel/Torus/Bohrung), r optional
  {"type": "point", "p"}               Punkt (Kugelmitte, Ecke, Klickpunkt)
  {"type": "sphere","p", "r"}
Sie entstehen aus Features oder aus einem lokalen Fit um einen Klickpunkt.
"""
from __future__ import annotations

import math

import numpy as np

from .geom_utils import unit
from .i18n import tr


# ===================================================================== Messobjekte
def feature_measurable(f):
    p, k = f.params, f.kind
    name = f.name
    if k == "plane":
        return {"type": "plane", "o": np.asarray(p["origin"], float), "n": unit(p["normal"]), "name": name}
    if k in ("cylinder",):
        return {"type": "axis", "p": np.asarray(p["point"], float), "d": unit(p["axis"]), "r": float(p["radius"]),
                "name": name}
    if k == "cone":
        return {"type": "axis", "p": np.asarray(p["apex"], float), "d": unit(p["axis"]), "name": name,
                "half_angle": float(p["half_angle"])}
    if k == "torus":
        return {"type": "axis", "p": np.asarray(p["center"], float), "d": unit(p["axis"]), "name": name,
                "R": float(p["R"]), "rt": float(p["r"])}
    if k == "hole":
        return {"type": "axis", "p": np.asarray(p["point"], float), "d": unit(p["axis"]), "r": 0.5 * float(p["d"]),
                "name": name}
    if k == "sphere":
        return {"type": "sphere", "p": np.asarray(p["center"], float), "r": float(p["radius"]), "name": name}
    if k in ("extrusion",):
        return {"type": "axis", "p": np.asarray(p["origin"], float), "d": unit(p["dir"]), "name": name}
    if k in ("revolve",):
        return {"type": "axis", "p": np.asarray(p["point"], float), "d": unit(p["axis"]), "name": name}
    if k == "ellipsoid":
        return {"type": "point", "p": np.asarray(p["center"], float), "name": name}
    raise ValueError(k)


def local_measurable(model, face_id, radius=None, tol=0.05):
    """Lokaler Fit um ein angeklicktes Dreieck: Ebene, sonst Zylinder, sonst Punkt."""
    from .fitting import fit_cylinder, fit_plane
    c0 = model.centroids[face_id]
    radius = radius or max(3 * model.mean_edge_length(), 1.0)
    ids = np.asarray(model.centroid_tree.query_ball_point(c0, radius), np.int64)
    ids = ids[~model.synthetic[ids]] if (~model.synthetic[ids]).sum() >= 8 else ids
    P, N, w = model.centroids[ids], model.fit_normals[ids], model.areas[ids]
    if len(ids) >= 4:
        fp = fit_plane(P, N, w)
        if fp.rms < 2.5 * tol:
            return {"type": "plane", "o": c0, "n": unit(fp.params["normal"]), "name": tr("lokale Ebene", "local plane"),
                    "rms": fp.rms}
        # größerer Ausschnitt für den Zylinder
        ids2 = np.asarray(model.centroid_tree.query_ball_point(c0, 2.5 * radius), np.int64)
        try:
            fc = fit_cylinder(model.centroids[ids2], model.fit_normals[ids2], model.areas[ids2])
            if fc.rms < 3 * tol and fc.params["radius"] < 50 * radius:
                return {"type": "axis", "p": np.asarray(fc.params["point"], float), "d": unit(fc.params["axis"]),
                        "r": float(fc.params["radius"]), "name": tr("lokaler Zylinder", "local cylinder"),
                        "rms": fc.rms}
        except Exception:
            pass
    return {"type": "point", "p": c0, "name": tr("Punkt", "point")}


# ===================================================================== Messen
def _ang(a, b):
    c = abs(float(np.clip(unit(a) @ unit(b), -1, 1)))
    return math.degrees(math.acos(c))


def _line_line(p1, d1, p2, d2):
    n = np.cross(d1, d2)
    if np.linalg.norm(n) < 1e-9:
        w = p2 - p1
        return float(np.linalg.norm(w - (w @ d1) * d1)), True
    return float(abs((p2 - p1) @ unit(n))), False


def measure(a, b):
    """Rückgabe: Liste (Bezeichnung, Wert, Einheit)."""
    out = []
    ta, tb = a["type"], b["type"]
    if ta == "sphere":
        a = dict(a, type="point")
        out.append((tr("Ø Kugel A", "Ø sphere A"), 2 * a["r"], "mm"))
    if tb == "sphere":
        b = dict(b, type="point")
        out.append((tr("Ø Kugel B", "Ø sphere B"), 2 * b["r"], "mm"))
    ta, tb = a["type"], b["type"]
    for nm, m in (("A", a), ("B", b)):
        if m["type"] == "axis" and m.get("r"):
            out.append((f"Ø {nm}", 2 * m["r"], "mm"))
    pair = {ta, tb}
    if ta == "plane" and tb == "plane":
        ang = _ang(a["n"], b["n"])
        out.append((tr("Winkel", "Angle"), ang, "°"))
        out.append((tr("Ergänzungswinkel", "Supplementary angle"), 180.0 - ang, "°"))
        if ang < 1.0:
            d1 = abs(float((b["o"] - a["o"]) @ a["n"]))
            d2 = abs(float((a["o"] - b["o"]) @ b["n"]))
            out.append((tr("Abstand (parallel)", "Distance (parallel)"), 0.5 * (d1 + d2), "mm"))
    elif pair == {"plane", "axis"}:
        pl, ax = (a, b) if ta == "plane" else (b, a)
        ang_n = _ang(pl["n"], ax["d"])
        out.append((tr("Winkel Achse–Ebene", "Angle axis–plane"), 90.0 - ang_n, "°"))
        if ang_n > 89.0:
            out.append((tr("Abstand Achse–Ebene", "Distance axis–plane"), abs(float((ax["p"] - pl["o"]) @ pl["n"])), "mm"))
        else:
            t = float((pl["o"] - ax["p"]) @ pl["n"] / (ax["d"] @ pl["n"]))
            hit = ax["p"] + t * ax["d"]
            out.append((tr("Durchstoßpunkt", "Intersection point"), hit, "mm"))
    elif ta == "axis" and tb == "axis":
        ang = _ang(a["d"], b["d"])
        out.append((tr("Winkel zwischen Achsen", "Angle between axes"), ang, "°"))
        d, par = _line_line(a["p"], unit(a["d"]), b["p"], unit(b["d"]))
        out.append((tr("Achsabstand" if par else "Abstand windschief", "Axis distance" if par else "Skew distance"), d, "mm"))
    elif pair == {"point", "plane"}:
        pt, pl = (a, b) if ta == "point" else (b, a)
        out.append((tr("Abstand Punkt–Ebene", "Distance point–plane"), abs(float((pt["p"] - pl["o"]) @ pl["n"])), "mm"))
    elif pair == {"point", "axis"}:
        pt, ax = (a, b) if ta == "point" else (b, a)
        w = pt["p"] - ax["p"]
        d = unit(ax["d"])
        out.append((tr("Abstand Punkt–Achse", "Distance point–axis"), float(np.linalg.norm(w - (w @ d) * d)), "mm"))
    elif ta == "point" and tb == "point":
        v = b["p"] - a["p"]
        out.append((tr("Abstand", "Distance"), float(np.linalg.norm(v)), "mm"))
        out += [("ΔX", float(v[0]), "mm"), ("ΔY", float(v[1]), "mm"), ("ΔZ", float(v[2]), "mm")]
    return out


def format_measure(rows):
    lines = []
    for lab, val, unit_ in rows:
        if isinstance(val, np.ndarray):
            lines.append(f"{lab}: ({val[0]:.3f}, {val[1]:.3f}, {val[2]:.3f}) {unit_}")
        else:
            lines.append(f"{lab}: {val:.4f} {unit_}" if unit_ == "mm" else f"{lab}: {val:.3f}{unit_}")
    return "\n".join(lines)


# ===================================================================== Ausrichtung optimieren
def feature_directions(features):
    """Richtungen (Normalen/Achsen) mit Gewicht (Flächenanzahl)."""
    D, W, names = [], [], []
    for f in features:
        k, p = f.kind, f.params
        key = {"plane": "normal", "cylinder": "axis", "cone": "axis", "torus": "axis", "hole": "axis",
               "extrusion": "dir", "revolve": "axis"}.get(k)
        if key is None or p.get("aux") or p.get("fixed"):
            continue
        D.append(unit(p[key]))
        W.append(float(max(len(f.face_ids), 1)) * (3.0 if k in ("cylinder", "hole", "cone") else 1.0))
        names.append(f.name)
    return np.asarray(D).reshape(-1, 3), np.asarray(W), names


def _kabsch(A, B, w):
    """Rotation R mit R @ a_i ≈ b_i (gewichtet)."""
    H = (A * w[:, None]).T @ B
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    Dm = np.diag([1, 1, d])
    return Vt.T @ Dm @ U.T


def optimize_alignment(features, capture_deg=6.0, iters=8, center=None):
    """'Manhattan'-Ausrichtung: Normalen/Achsen, die nahe an einer Hauptachse liegen, werden
    gemeinsam (gewichtete Kabsch-Rotation) exakt auf ±X/±Y/±Z gedreht.
    Rückgabe (T 4×4, Bericht dict)."""
    D, W, names = feature_directions(features)
    if len(D) < 2:
        raise ValueError(tr("Zu wenige gerichtete Features (Ebenen/Achsen).", "Too few directed features (planes/axes)."))
    E = np.vstack([np.eye(3), -np.eye(3)])
    R = np.eye(3)
    cos_c = math.cos(math.radians(capture_deg))
    used = np.zeros(len(D), bool)
    for _ in range(iters):
        Dr = D @ R.T
        dots = Dr @ E.T
        j = np.argmax(dots, axis=1)
        best = dots[np.arange(len(D)), j]
        used = best > cos_c
        if used.sum() < 2:
            break
        T = E[j[used]]
        # nur zwei unabhängige Richtungen nötig; Kabsch auf allen
        if np.linalg.matrix_rank(T, tol=1e-6) < 2:
            # nur eine Hauptrichtung: minimale Drehung um sie herum genügt
            # Richtungen vorher gleichsinnig zur Zielachse drehen (±X heben sich sonst auf)
            sg = np.sign(Dr[used] @ T[0])
            sg[sg == 0] = 1.0
            a = unit((Dr[used] * (W[used] * sg)[:, None]).sum(0))
            a = a if a @ T[0] >= 0 else -a
            from .geom_utils import rotation_between
            R = rotation_between(a, T[0]) @ R
            break
        dR = _kabsch(Dr[used], T, W[used])
        R = dR @ R
        if np.linalg.norm(dR - np.eye(3)) < 1e-12:
            break
    before = [math.degrees(math.acos(min(1.0, float(np.max(np.abs(d @ np.eye(3))))))) for d in D[used]]
    after = [math.degrees(math.acos(min(1.0, float(np.max(np.abs((R @ d) @ np.eye(3))))))) for d in D[used]]
    c = np.zeros(3) if center is None else np.asarray(center, float)
    T4 = np.eye(4)
    T4[:3, :3] = R
    T4[:3, 3] = c - R @ c
    rot_ang = math.degrees(math.acos(np.clip((np.trace(R) - 1) / 2, -1, 1)))
    rep = {"n_used": int(used.sum()), "n_total": len(D), "rotation_deg": rot_ang,
           "max_before": max(before) if before else 0.0, "max_after": max(after) if after else 0.0,
           "names": [n for n, u in zip(names, used) if u]}
    return T4, rep


def manhattan_frame(features, capture_deg=6.0, floor_normal=None):
    """Grobe Drehung, die möglichst viele Ebenennormalen/Achsen (flächengewichtet) auf X/Y/Z legt –
    für Teile, deren Hauptachsen (PCA) schräg zu den Konstruktionsflächen liegen (Winkel, Konsolen).
    Mit festgelegtem Boden bleibt dessen Normale −Z. Rückgabe (R 3×3 oder None, Anteil vorher, nachher)."""
    D, W, _ = feature_directions(features)
    if len(D) < 2:
        return None, 0.0, 0.0
    W = np.asarray(W, float)
    cos_c = math.cos(math.radians(capture_deg))

    def score(R):
        Dr = D @ R.T
        return float(W[np.abs(Dr).max(1) > cos_c].sum() / W.sum())

    order = np.argsort(-W)
    cand = []
    for i in order[:12]:
        d = unit(D[i])
        if all(abs(float(d @ c)) < 0.999 for c in cand):
            cand.append(d)
    zs = [-unit(floor_normal)] if floor_normal is not None else cand
    best = (score(np.eye(3)), None)
    s0 = best[0]
    for z in zs:
        for xd in cand:
            x = xd - (xd @ z) * z
            if np.linalg.norm(x) < 1e-6 or abs(float(unit(xd) @ z)) > math.sin(math.radians(capture_deg)):
                continue
            x = unit(x)
            y = np.cross(z, x)
            R = np.vstack([x, y, z])
            sc = score(R)
            if sc > best[0] + 1e-9:
                best = (sc, R)
    return best[1], s0, best[0]


def floor_translation(features, up=(0, 0, 1)):
    """Verschiebung, die die tiefste nach unten zeigende Ebene auf z=0 legt (nach dem Drehen)."""
    up = unit(up)
    best = None
    for f in features:
        if f.kind != "plane":
            continue
        n = unit(f.params["normal"])
        if n @ up < -0.999:
            z = float(np.asarray(f.params["origin"]) @ up)
            if best is None or z < best:
                best = z
    if best is None:
        return None
    T = np.eye(4)
    T[:3, 3] = -best * up
    return T


# ===================================================================== Zusammenführen
def _equiv(f, g, tol):
    if f.kind != g.kind or f.role != "surface" or g.role != "surface":
        return False
    k, p, q = f.kind, f.params, g.params
    lim = 30 * tol + 0.2
    if k == "plane":
        n1, n2 = unit(p["normal"]), unit(q["normal"])
        if n1 @ n2 < math.cos(math.radians(3)):
            return False
        return abs(float((np.asarray(q["origin"]) - p["origin"]) @ n1)) < max(3 * tol, 0.05)
    if k in ("cylinder", "cone", "torus"):
        a1, a2 = unit(p["axis"]), unit(q["axis"])
        if abs(float(a1 @ a2)) < math.cos(math.radians(3)):
            return False
        base1 = np.asarray(p.get("point", p.get("apex", p.get("center"))), float)
        base2 = np.asarray(q.get("point", q.get("apex", q.get("center"))), float)
        d = base2 - base1
        if k == "cylinder":
            return np.linalg.norm(d - (d @ a1) * a1) < max(3 * tol, 0.1) and abs(p["radius"] - q["radius"]) < max(3 * tol, 0.05)
        if k == "cone":
            return np.linalg.norm(d) < lim and abs(p["half_angle"] - q["half_angle"]) < math.radians(1.5)
        return np.linalg.norm(d) < lim and abs(p["R"] - q["R"]) < lim and abs(p["r"] - q["r"]) < max(3 * tol, 0.05)
    if k == "sphere":
        return (np.linalg.norm(np.asarray(p["center"]) - q["center"]) < max(3 * tol, 0.1)
                and abs(p["radius"] - q["radius"]) < max(3 * tol, 0.05))
    return False


def merge_similar(model, features, tol=0.05):
    """Gleichartige, geometrisch gleiche Flächen (z. B. Riffel-Ebenen einer Membranfüllung, zwei
    Hälften einer Bohrung) gemeinsam neu fitten. Rückgabe: Liste (neues Feature, [ersetzte IDs])."""
    from .features import Feature, _selection_data, _stats, trim_primitive
    from .fitting import FIT_FUNCS
    surf = [f for f in features if f.role == "surface" and f.kind in ("plane", "cylinder", "cone", "torus", "sphere")
            and not f.params.get("fixed")]
    parent = {f.id: f.id for f in surf}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i, f in enumerate(surf):
        for g in surf[i + 1:]:
            if find(f.id) != find(g.id) and (_equiv(f, g, tol) or _equiv_data(model, f, g, tol)):
                parent[find(g.id)] = find(f.id)
    groups = {}
    for f in surf:
        groups.setdefault(find(f.id), []).append(f)
    from .fitting import PRIMITIVE_ORDER
    out = []
    for grp in groups.values():
        if len(grp) < 2:
            continue
        ids = np.unique(np.concatenate([np.asarray(f.face_ids, np.int64) for f in grp]))
        C, N, A = _selection_data(model, ids)
        worst = max(float(f.stats.get("rms", tol)) for f in grp)
        # gemischte Gruppen (z. B. Zylinder + fast zylindrischer Kegel): einfachstes passendes Modell
        kinds = sorted({f.kind for f in grp}, key=PRIMITIVE_ORDER.index)
        fr = None
        for kind in kinds:
            try:
                cand = FIT_FUNCS[kind](C, N, A, max(2 * tol, 0.02))
            except Exception:
                continue
            if cand.rms <= 1.5 * max(worst, tol):
                fr = cand
                break
        if fr is None:
            continue
        kind = fr.kind
        try:
            pr = trim_primitive(kind, fr.params, model, ids, {"outline_tol": max(0.05, 2 * tol)})
        except Exception:
            pr = fr.params
        big = max(grp, key=lambda f: len(f.face_ids))
        nf = Feature(kind, pr, "surface", ids, name=big.name if big.kind == kind else None,
                     stats=_stats(fr.residuals, fr.weights), auto=big.auto, color=big.color)
        out.append((nf, [f.id for f in grp]))
    return out


def _equiv_data(model, f, g, tol):
    """Verschiedene Typen, gleiche Fläche? (z. B. RANSAC hat einen Zylinder zusätzlich als fast
    zylindrischen Kegel gefunden). Datentest: die Dreiecke des komplexeren Features (Kegel)
    liegen größtenteils auf der Fläche des einfacheren (Zylinder) – dann ist die Verjüngung im
    Scan nicht messbar und der Zylinder genügt."""
    rot = ("cylinder", "cone")
    if f.kind == g.kind or f.kind not in rot or g.kind not in rot:
        return False
    if abs(float(unit(f.params["axis"]) @ unit(g.params["axis"]))) < math.cos(math.radians(5)):
        return False
    cone, cyl = (f, g) if f.kind == "cone" else (g, f)
    if math.degrees(float(cone.params["half_angle"])) > 5.0:
        return False
    from .carriers import carrier_of
    lim = max(3 * tol, 0.05)
    ids = np.asarray(cone.face_ids, np.int64)
    ids = ids[(ids >= 0) & (ids < model.n_faces)]
    if len(ids) == 0:
        return False
    if len(ids) > 2000:
        ids = np.random.default_rng(0).choice(ids, 2000, replace=False)
    try:
        d = np.abs(carrier_of(cyl).distance(model.centroids[ids]))
    except Exception:
        return False
    return bool(np.mean(d < lim) >= 0.85)
