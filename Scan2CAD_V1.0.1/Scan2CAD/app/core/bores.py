"""Bohrungen: Feature 'hole' (Achse, Ø, Tiefe, Boden, Senkung, Gewinde), Erkennung im Scan,
Normwerte.

Ein Scanner sieht in Bohrungen meist nur den oberen Teil sauber (Sichtschatten); weiter unten
fehlen Daten oder es entstehen Artefakte ("Murks"). Deshalb wird eine Bohrung nicht aus allen
Dreiecken gefittet, sondern
  * Achse und Durchmesser aus dem sauberen Zylinderteil,
  * Eintrittspunkt aus der umgebenden Fläche (Ebene),
  * Tiefe/Boden aus erkannten Kegel-/Ebenenstücken – oder, wenn der Grund nicht erfasst ist,
    'sichtbare Tiefe' + Markierung 'Boden nicht erfasst' (vom Anwender festzulegen).
Alle Dreiecke im Bohrungsvolumen (inkl. Murks und Füllmembranen) gehören zur Bohrung und werden
beim Export durch die exakte Bohrungsgeometrie ersetzt.

Parameter (params):
  point        Eintrittspunkt auf der Achse (auf der Eintrittsfläche)
  axis         Bohrrichtung (ins Material)
  d            Bohrungs-Ø (bei Gewinde: Kernloch-Ø)
  depth        zylindrische Tiefe ab Eintritt (bei 'through': Länge bis zum Austritt)
  bottom       'flat' | 'drill' (Bohrerspitze, point_angle) | 'ball' | 'through'
  point_angle  Spitzenwinkel in Grad (118 Standard)
  head         'none' | 'csk' (Senkung: head_d, head_angle) | 'cbore' (Flachsenkung: head_d, head_depth)
  thread       None | {size, pitch, length, display: 'cosmetic'|'nominal'|'helix'}
  bottom_seen  ob der Boden im Scan erfasst war; seen_depth = tiefste echte Messung
"""
from __future__ import annotations

import math

import numpy as np

from .geom_utils import orthonormal_frame, unit
from .i18n import tr

# ---------------------------------------------------------------- Normtabellen (ISO 261/262, ISO 273, DIN 974)
COARSE = {"M1.6": (1.6, 0.35), "M2": (2.0, 0.4), "M2.5": (2.5, 0.45), "M3": (3.0, 0.5), "M4": (4.0, 0.7),
          "M5": (5.0, 0.8), "M6": (6.0, 1.0), "M8": (8.0, 1.25), "M10": (10.0, 1.5), "M12": (12.0, 1.75),
          "M14": (14.0, 2.0), "M16": (16.0, 2.0), "M18": (18.0, 2.5), "M20": (20.0, 2.5), "M22": (22.0, 2.5),
          "M24": (24.0, 3.0), "M27": (27.0, 3.0), "M30": (30.0, 3.5), "M36": (36.0, 4.0)}
FINE = {"M8": (1.0,), "M10": (1.25, 1.0), "M12": (1.5, 1.25), "M14": (1.5,), "M16": (1.5,), "M18": (2.0, 1.5),
        "M20": (2.0, 1.5), "M22": (2.0, 1.5), "M24": (2.0,), "M27": (2.0,), "M30": (2.0,), "M36": (3.0,)}
# Durchgangslöcher ISO 273: fein / mittel / grob
CLEARANCE = {"M2": (2.2, 2.4, 2.6), "M2.5": (2.7, 2.9, 3.1), "M3": (3.2, 3.4, 3.6), "M4": (4.3, 4.5, 4.8),
             "M5": (5.3, 5.5, 5.8), "M6": (6.4, 6.6, 7.0), "M8": (8.4, 9.0, 10.0), "M10": (10.5, 11.0, 12.0),
             "M12": (13.0, 13.5, 14.5), "M14": (15.0, 15.5, 16.5), "M16": (17.0, 17.5, 18.5),
             "M20": (21.0, 22.0, 24.0), "M24": (25.0, 26.0, 28.0), "M30": (31.0, 33.0, 35.0)}
# Flachsenkungen für Zylinderschrauben ISO 4762 (DIN 974-1, Reihe 1): (Ø, Tiefe)
CBORE = {"M3": (6.5, 3.4), "M4": (8.0, 4.4), "M5": (10.0, 5.4), "M6": (11.0, 6.4), "M8": (15.0, 8.6),
         "M10": (18.0, 10.6), "M12": (20.0, 12.6), "M16": (26.0, 16.6), "M20": (33.0, 20.6)}
# Kegelsenkungen 90° (ISO 10642 / DIN 74): Senk-Ø
CSK = {"M3": 6.5, "M4": 8.6, "M5": 10.4, "M6": 12.4, "M8": 16.4, "M10": 20.4, "M12": 24.4, "M16": 32.4}
POINT_ANGLES = (90.0, 118.0, 120.0, 130.0, 135.0, 140.0, 180.0)
HEAD_ANGLES = (60.0, 82.0, 90.0, 100.0, 120.0)
BOTTOMS = ("flat", "drill", "ball", "through")
HEADS = ("none", "csk", "cbore")


def tap_drill(size, pitch=None):
    D, P = COARSE[size]
    return round(D - (pitch or P), 2)


def thread_pitches(size):
    if size not in COARSE:
        return []
    return [COARSE[size][1]] + list(FINE.get(size, ()))


def standard_drills():
    out = [round(x, 2) for x in np.arange(0.5, 10.0001, 0.1)]
    out += [round(x, 2) for x in np.arange(10.5, 20.0001, 0.5)]
    out += [float(x) for x in range(21, 51)]
    return out


def diameter_candidates():
    """(Ø, Kennung DE, Kennung EN, Gewinde-Vorschlag)"""
    c = []
    for s, (D, P) in COARSE.items():
        for p in thread_pitches(s):
            k = f"{s}" if p == P else f"{s}×{p:g}"
            c.append((round(D - p, 2), f"Kernloch {k}", f"tap drill {k}", (s, p)))
        c.append((D, f"Nenn-Ø {s}", f"nominal {s}", None))
    for s, (f_, m_, g_) in CLEARANCE.items():
        c.append((f_, f"Durchgang {s} fein", f"clearance {s} fine", None))
        c.append((m_, f"Durchgang {s} mittel", f"clearance {s} medium", None))
        c.append((g_, f"Durchgang {s} grob", f"clearance {s} coarse", None))
    for d in standard_drills():
        c.append((d, f"Bohrer {d:g}", f"drill {d:g}", None))
    return c


def snap_diameter(d, tol=None, meas=None):
    """Nächster Normwert, sonst glatter Wert (0,1-mm-Raster). Rückgabe (Ø, Beschreibung,
    Gewinde-Vorschlag|None) oder None.

    Toleranz für Normwerte: aus der Messunsicherheit meas (Fit-Toleranz) und 0,4 % des Ø –
    ein exakt modellierter Ø 22,4 wird also nicht auf 22 gezogen. Ohne meas: 0,05 mm + 1 %."""
    if tol is None:
        tol = max(0.05, 0.004 * d + 2.0 * meas) if meas is not None else max(0.06, 0.01 * d + 0.05)
    best = None
    for val, de, en, th in diameter_candidates():
        err = abs(val - d)
        if err > tol:
            continue
        kind = de.split()[0]
        if abs(val * 2 - round(val * 2)) < 1e-9:      # ganze/halbe mm: neutral als Bohrer-Ø
            prio = {"Bohrer": 0, "Durchgang": 1, "Kernloch": 2, "Nenn-Ø": 3}[kind]
        else:
            prio = {"Durchgang": 0, "Kernloch": 1, "Bohrer": 2, "Nenn-Ø": 3}[kind]
        key = (round(err / max(tol, 1e-9) * 4), prio, err)
        if best is None or key < best[0]:
            best = (key, val, tr(de, en), th)
    if best is None:
        g = round(d * 10) / 10
        if abs(g - d) <= max(0.05, meas if meas is not None else 0.05) and abs(g - d) > 1e-9:
            return g, tr("glatter Wert", "round value"), None
        return None
    return best[1], best[2], best[3]


def suggest_thread(d, tol=None):
    """Gewinde, dessen Kernloch zu d passt: (Größe, Steigung) oder None."""
    tol = max(0.06, 0.015 * d) if tol is None else tol
    best = None
    for s in COARSE:
        for p in thread_pitches(s):
            err = abs(tap_drill(s, p) - d)
            if err <= tol and (best is None or err < best[0]):
                best = (err, s, p)
    return None if best is None else (best[1], best[2])


def default_params(point, axis, d, depth):
    return {"point": np.asarray(point, float), "axis": unit(axis), "d": float(d), "depth": float(depth),
            "bottom": "drill", "point_angle": 118.0, "head": "none", "head_d": float(d) * 1.8,
            "head_angle": 90.0, "head_depth": float(d) * 0.6, "thread": None, "bottom_seen": True,
            "seen_depth": float(depth)}


# ---------------------------------------------------------------- Geometrie
def hole_profile(p, margin=1.0):
    """Halbprofil (r, z) der Bohrung, z entlang der Bohrrichtung ab Eintritt (z<0 = über der Fläche).
    Geschlossenes Polygon über die Achse (r=0)."""
    r = 0.5 * float(p["d"])
    th = p.get("thread") or None
    if th and th.get("display") == "nominal" and th.get("size") in COARSE:
        r_thr = 0.5 * COARSE[th["size"]][0]
        L_thr = float(th.get("length") or p["depth"])
    else:
        r_thr, L_thr = r, 0.0
    depth = float(p["depth"])
    if p.get("bottom") == "through":
        depth += max(margin, float(p.get("ext_exit", 0.0)))
    head = p.get("head", "none")
    pts = []
    # ext_entry/ext_exit: Verlängerung an schräg angeschnittenen Enden (nur beim Export gesetzt)
    top = -max(margin, 0.5, float(p.get("ext_entry", 0.0)))
    if head == "csk":
        R = max(0.5 * float(p.get("head_d", 2 * r)), r)
        ang = math.radians(float(p.get("head_angle", 90.0)) / 2)
        hz = (R - max(r, r_thr)) / max(math.tan(ang), 1e-6)
        R_top = R + (-top) * math.tan(ang)
        pts += [(0.0, top), (R_top, top), (R, 0.0), (max(r, r_thr), hz)]
        z_cyl0 = hz
    elif head == "cbore":
        R = max(0.5 * float(p.get("head_d", 2 * r)), r)
        hd = float(p.get("head_depth", 0.0))
        pts += [(0.0, top), (R, top), (R, hd), (max(r, r_thr), hd)]
        z_cyl0 = hd
    else:
        pts += [(0.0, top), (max(r, r_thr), top)]
        z_cyl0 = 0.0
    if r_thr > r and L_thr > z_cyl0:
        zt = min(L_thr, depth)
        pts += [(r_thr, zt), (r, zt)]
    pts.append((r, depth))
    bottom = p.get("bottom", "flat")
    if bottom == "drill":
        a = math.radians(float(p.get("point_angle", 118.0)) / 2)
        pts.append((0.0, depth + r / max(math.tan(a), 1e-6)))
    elif bottom == "ball":
        for t in np.linspace(0, math.pi / 2, 10)[1:]:
            pts.append((r * math.cos(t), depth + r * math.sin(t)))
        pts[-1] = (0.0, depth + r)
    else:
        pts.append((0.0, depth))
    # doppelte Punkte entfernen
    out = [pts[0]]
    for q in pts[1:]:
        if abs(q[0] - out[-1][0]) > 1e-9 or abs(q[1] - out[-1][1]) > 1e-9:
            out.append(q)
    return out


def build_hole(p, role="sub"):
    """Werkzeugkörper der Bohrung (Rotationskörper), zum Abziehen vom Bauteil."""
    from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeFace, BRepBuilderAPI_MakePolygon
    from OCP.BRepPrimAPI import BRepPrimAPI_MakeRevol
    from OCP.gp import gp_Ax1
    from . import occ_build as ob
    o = np.asarray(p["point"], float)
    a = unit(p["axis"])
    x, _, _ = orthonormal_frame(a)
    prof = hole_profile(p)
    poly = BRepBuilderAPI_MakePolygon()
    for r, z in prof:
        poly.Add(ob.P(o + r * x + z * a))
    poly.Close()
    face = BRepBuilderAPI_MakeFace(poly.Wire(), True).Face()
    solid = BRepPrimAPI_MakeRevol(face, gp_Ax1(ob.P(o), ob.D(a)), 2 * math.pi).Shape()
    th = p.get("thread") or None
    if th and th.get("display") == "helix":
        try:
            groove = build_thread_groove(p)
            if groove is not None:
                solid = ob.compound([ob.fix_shape(solid), groove])   # occ_build.cut schneidet nacheinander
        except Exception:
            pass
    return solid if solid.ShapeType().name == 'TopAbs_COMPOUND' else ob.fix_shape(solid)


def thread_groove_mesh(p, seg_per_turn=24):
    """Gewinderille (Innengewinde) als geschlossenes Dreiecksnetz: Trapezprofil (ISO-ähnlich,
    an der Kernbohrung 0,75·P breit, am Außen-Ø 0,125·P) entlang einer Schraubenlinie."""
    th = p["thread"]
    size = th.get("size")
    if size not in COARSE:
        return None
    D = COARSE[size][0]
    P = float(th.get("pitch") or COARSE[size][1])
    L = float(th.get("length") or p["depth"])
    r_in = 0.5 * float(p["d"]) - 0.05
    r_out = 0.5 * D
    if r_out <= r_in + 0.05 or L <= P:
        return None
    o = np.asarray(p["point"], float)
    a = unit(p["axis"])
    x, y, _ = orthonormal_frame(a)
    turns = (L - P) / P
    n = max(int(turns * seg_per_turn), 4)
    th_ = np.linspace(0, 2 * math.pi * turns, n + 1)
    prof = [(r_in, -0.375 * P), (r_out, -0.0625 * P), (r_out, 0.0625 * P), (r_in, 0.375 * P)]
    V = []
    for t in th_:
        zc = 0.5 * P + P * t / (2 * math.pi)
        e = math.cos(t) * x + math.sin(t) * y
        for r, dz in prof:
            V.append(o + r * e + (zc + dz) * a)
    V = np.asarray(V)
    F = []
    for i in range(n):
        b0, b1 = 4 * i, 4 * (i + 1)
        for k in range(4):
            k2 = (k + 1) % 4
            F.append((b0 + k, b1 + k, b1 + k2))
            F.append((b0 + k, b1 + k2, b0 + k2))
    F += [(0, 2, 1), (0, 3, 2)]
    e0 = 4 * n
    F += [(e0, e0 + 1, e0 + 2), (e0, e0 + 2, e0 + 3)]
    F = np.asarray(F, np.int64)
    # Orientierung nach außen (Volumen positiv)
    tri = V[F]
    if np.einsum("ij,ij->i", tri[:, 0], np.cross(tri[:, 1], tri[:, 2])).sum() < 0:
        F = F[:, ::-1]
    return V, F


def build_thread_groove(p):
    """Experimentell: Gewinderille als facettierter Schraubkörper (robust für Boolesche Op.)."""
    from . import occ_build as ob
    mesh = thread_groove_mesh(p)
    if mesh is None:
        return None
    return ob.mesh_to_faceted(mesh[0], mesh[1], make_solid=True)


def hole_summary(p):
    d = float(p["d"])
    parts = [f"Ø{d:.3f}"]
    th = p.get("thread") or None
    if th and th.get("size"):
        pitch = th.get("pitch")
        parts.insert(0, f"{th['size']}" + (f"×{pitch:g}" if pitch and pitch != COARSE.get(th['size'], (0, 0))[1] else ""))
    b = p.get("bottom", "flat")
    if b == "through":
        parts.append(tr("durchgehend", "through"))
    else:
        parts.append(tr(f"Tiefe {p['depth']:.2f}", f"depth {p['depth']:.2f}"))
        parts.append({"flat": tr("flach", "flat"), "drill": tr(f"Spitze {p.get('point_angle', 118):g}°",
                                                                  f"point {p.get('point_angle', 118):g}°"),
                      "ball": tr("Kugel", "ball")}.get(b, b))
    h = p.get("head", "none")
    if h == "csk":
        parts.append(tr(f"Senkung Ø{p['head_d']:.2f}×{p['head_angle']:g}°", f"countersink Ø{p['head_d']:.2f}×{p['head_angle']:g}°"))
    elif h == "cbore":
        parts.append(tr(f"Flachsenkung Ø{p['head_d']:.2f}×{p['head_depth']:.2f}", f"counterbore Ø{p['head_d']:.2f}×{p['head_depth']:.2f}"))
    if not p.get("bottom_seen", True) and b != "through":
        parts.append(tr("⚠ Boden nicht erfasst", "⚠ bottom not scanned"))
    return ", ".join(parts)


def snap_hole(p, diameter=True, angles=True, depth_step=0.0, axis_dirs=None, axis_tol_deg=2.0, meas=None):
    """'Glatte' Werte. Rückgabe (neue params, Liste von Hinweisen). meas = Messunsicherheit."""
    q = dict(p)
    notes = []
    if diameter:
        s = snap_diameter(q["d"], meas=meas)
        if s is not None and abs(s[0] - q["d"]) > 1e-9:
            notes.append(f"Ø {q['d']:.3f} → {s[0]:g} ({s[1]})")
            q["d"] = s[0]
        if q.get("head") in ("csk", "cbore"):
            hd = round(float(q["head_d"]) * 10) / 10
            if abs(hd - q["head_d"]) > 1e-9:
                notes.append(tr(f"Senk-Ø → {hd:g}", f"head Ø → {hd:g}"))
                q["head_d"] = hd
    if angles:
        for key, vals in (("point_angle", POINT_ANGLES), ("head_angle", HEAD_ANGLES)):
            v = float(q.get(key, vals[0]))
            best = min(vals, key=lambda t: abs(t - v))
            if abs(best - v) <= 6.0 and abs(best - v) > 1e-9:
                notes.append(f"{key} {v:.1f}° → {best:g}°")
                q[key] = best
    if depth_step and q.get("bottom") != "through":
        dd = round(float(q["depth"]) / depth_step) * depth_step
        if abs(dd - q["depth"]) > 1e-9:
            notes.append(tr(f"Tiefe → {dd:g}", f"depth → {dd:g}"))
            q["depth"] = dd
        if q.get("head") == "cbore":
            q["head_depth"] = round(float(q["head_depth"]) / depth_step) * depth_step
    if axis_dirs is not None:
        a = unit(q["axis"])
        best = None
        for d in axis_dirs:
            d = unit(d)
            c = abs(float(a @ d))
            if best is None or c > best[0]:
                best = (c, d if a @ d > 0 else -d)
        if best is not None and math.degrees(math.acos(min(1.0, best[0]))) <= axis_tol_deg:
            if np.linalg.norm(best[1] - a) > 1e-12:
                notes.append(tr("Achse ausgerichtet", "axis aligned") +
                             f" ({math.degrees(math.acos(min(1.0, best[0]))):.2f}°)")
            q["axis"] = best[1]
    return q, notes


# ---------------------------------------------------------------- Netz-Analyse
class _InsideTest:
    """Innen/außen-Test über den vorzeichenbehafteten Abstand zum (möglichst geschlossenen) Netz."""

    def __init__(self, model):
        from .preprocess import vf_to_pd
        self.pd = vf_to_pd(model.V, model.F)
        self.model = model
        self._bnd = None

    def boundary(self):
        """Punkte an offenen Rändern (Sichtschatten im Scan)."""
        if self._bnd is None:
            from .mesh_ops import boundary_vertex_mask
            self._bnd = boundary_vertex_mask(self.model)
        return self._bnd

    def signed(self, P):
        import pyvista as pv
        out = pv.PolyData(np.asarray(P, float)).compute_implicit_distance(self.pd)
        return np.asarray(out.point_data["implicit_distance"])


def _feature_faces(f, nF):
    ids = np.asarray(f.face_ids, np.int64)
    return ids[(ids >= 0) & (ids < nF)]


def _concavity(model, ids, point, axis):
    if len(ids) == 0:
        return 0.0
    P = model.centroids[ids] - point
    rad = P - np.outer(P @ axis, axis)
    rad = unit(rad)
    return float(np.mean(np.einsum("ij,ij->i", model.normals[ids], rad)))


def _coaxial(a1, p1, a2, p2, ang_deg=2.0, off=0.2):
    if abs(float(unit(a1) @ unit(a2))) < math.cos(math.radians(ang_deg)):
        return False
    d = np.asarray(p2, float) - np.asarray(p1, float)
    a = unit(a1)
    return float(np.linalg.norm(d - (d @ a) * a)) < off


def _axis_hits_carrier(car, o, a, t0, span):
    """Parameter t, bei dem die Gerade o + t·a die Trägerfläche trifft (nahe t0)."""
    if car.kind == "plane":
        n = unit(car.p["normal"])
        den = float(a @ n)
        if abs(den) < 1e-6:
            return None
        return float((np.asarray(car.p["origin"]) - o) @ n / den)
    ts = np.linspace(t0 - span, t0 + span, 61)
    d = car.distance(o + np.outer(ts, a))
    s = np.nonzero(np.sign(d[:-1]) != np.sign(d[1:]))[0]
    if len(s) == 0:
        return None
    k = s[np.argmin(np.abs(ts[s] - t0))]
    t1, t2, d1, d2 = ts[k], ts[k + 1], d[k], d[k + 1]
    return float(t1 - d1 * (t2 - t1) / (d2 - d1))


def hole_face_ids(model, p, margin=None, tol=0.05):
    """Alle Dreiecke im Bohrungsvolumen (Wand, Senkung, Boden, Murks, Füllmembranen)."""
    margin = max(0.25, 4 * tol) if margin is None else margin
    o = np.asarray(p["point"], float)
    a = unit(p["axis"])
    q = model.centroids - o
    z = q @ a
    rho = np.linalg.norm(q - np.outer(z, a), axis=1)
    prof = np.asarray(hole_profile(p, margin=0.0))
    # Außenkontur r(z): Profil ohne die Achspunkte
    zz, rr = prof[:, 1], prof[:, 0]
    z_end = float(zz.max())
    zs = np.linspace(-margin, z_end + margin, 400)
    rmax = np.zeros_like(zs)
    for i in range(len(prof) - 1):
        r1, z1 = prof[i]
        r2, z2 = prof[i + 1]
        lo, hi = min(z1, z2), max(z1, z2)
        m = (zs >= lo - 1e-9) & (zs <= hi + 1e-9)
        if hi - lo < 1e-9:
            rmax[m] = np.maximum(rmax[m], max(r1, r2))
        else:
            t = (zs[m] - z1) / (z2 - z1)
            rmax[m] = np.maximum(rmax[m], r1 + t * (r2 - r1))
    head_r = 0.5 * float(p.get("head_d", p["d"])) if p.get("head", "none") != "none" else 0.5 * float(p["d"])
    rmax[zs < 0] = max(head_r, 0.5 * float(p["d"]))
    rlim = np.interp(z, zs, rmax, left=-1.0, right=-1.0) + margin
    sel = (z > -0.6 * margin) & (z < z_end + margin) & (rho < rlim)
    return np.nonzero(sel)[0]


def _csk_from_free_faces(model, base, a, r, hend, sgn, taken, tol):
    """Kegelsenkung aus nicht zugeordneten Dreiecken am Bohrungsende (sgn: Richtung zur Öffnung).
    Gegen Scan-Verrundungen der Bohrungskante abgesichert: radiale Breite, Umfangsabdeckung,
    Öffnungswinkel 50–140°, koaxial, konkav."""
    from .features import Feature
    from .fitting import fit_cone
    C = model.centroids
    Q = C - base
    hq = Q @ a
    rq = np.linalg.norm(Q - np.outer(hq, a), axis=1)
    dz = (hq - hend) * sgn
    sel = ~taken & ~model.synthetic & (rq > 1.03 * r) & (rq < 2.6 * r) & (dz > -1.5 * r) & (dz < 1.5 * r)
    fids = np.nonzero(sel)[0]
    if len(fids) < 30:
        return None
    ca = np.abs(model.normals[fids] @ a)
    fids = fids[(ca > 0.3) & (ca < 0.95)]
    if len(fids) < 30:
        return None
    fids = np.asarray(model.components_of(fids)[0], np.int64)
    if len(fids) < 30 or np.ptp(rq[fids]) < max(0.6, 0.2 * r):
        return None
    x0, y0, _ = orthonormal_frame(a)
    phi = np.arctan2(Q[fids] @ y0, Q[fids] @ x0)
    if len(np.unique(np.floor((phi + np.pi) / (2 * np.pi) * 36).astype(int))) < 0.6 * 36:
        return None
    try:
        fr = fit_cone(C[fids], model.fit_normals[fids], model.weights[fids], fixed_axis=a, robust=max(2 * tol, 0.02))
    except Exception:
        return None
    al = math.degrees(float(fr.params["half_angle"]))
    if not (25.0 < al < 70.0) or fr.rms > max(3 * tol, 0.05):
        return None
    if not _coaxial(a, base, fr.params["axis"], fr.params["apex"], 3.0, max(0.3, 0.1 * r)):
        return None
    if _concavity(model, fids, base, a) > -0.2:
        return None
    return Feature("cone", dict(fr.params), "surface", fids, name=tr("Senkung", "Countersink"))


def _hole_from_cylinders(model, cyls, others, inside, tol, log):
    """Bohrung aus einer Gruppe koaxialer, gleich großer konkaver Zylinder."""
    from .carriers import carrier_of
    nF = model.n_faces
    car0 = carrier_of(cyls[0])
    a = car0.axis
    base = car0.base
    ids = np.concatenate([_feature_faces(f, nF) for f in cyls])
    w = model.areas[ids]
    r = float(np.average([f.params["radius"] for f in cyls], weights=[len(f.face_ids) + 1 for f in cyls]))
    h = (model.centroids[ids] - base) @ a
    real = ~model.synthetic[ids]
    h_lo, h_hi = float(h.min()), float(h.max())
    delta = max(1.0, 0.6 * r)
    probe = np.array([base + (h_lo - delta) * a, base + (h_hi + delta) * a])
    sd = inside.signed(probe)
    open_lo, open_hi = sd[0] > 0, sd[1] > 0
    # Endet die Wand an einem offenen Scan-Rand (Sichtschatten), ist das Ende nicht „offen“,
    # sondern nicht erfasst: Grundloch oder Durchgang entscheidet dann der Anwender.
    bnd = inside.boundary() if hasattr(inside, "boundary") else np.zeros(model.n_verts, bool)
    unseen = {"lo": False, "hi": False}
    if bnd.any():
        x0, y0, _ = orthonormal_frame(a)
        # Randpunkte auf dem Bohrungsmantel nahe dem jeweiligen Ende, Abdeckung des Umfangs
        B = model.V[bnd] - base
        hb = B @ a
        rb = np.linalg.norm(B - np.outer(hb, a), axis=1)
        phib = np.arctan2(B @ y0, B @ x0)
        band = max(1.0, 0.3 * r)
        on_wall = np.abs(rb - r) < max(0.3, 0.15 * r)

        def cover(phi):
            return len(np.unique(np.floor((phi + np.pi) / (2 * np.pi) * 36).astype(int))) / 36.0

        def end_unseen(hend):
            sel = on_wall & (np.abs(hb - hend) < band)
            return bool(sel.sum() >= 3 and cover(phib[sel]) > 0.4)

        unseen = {"lo": end_unseen(h_lo), "hi": end_unseen(h_hi)}
        if unseen["lo"] or unseen["hi"]:
            # Fortsetzung jenseits der Lücke? (Durchgangsbohrung, von der Gegenseite gescannt)
            Qc = model.centroids - base
            hq = Qc @ a
            radq = Qc - np.outer(hq, a)
            rq = np.linalg.norm(radq, axis=1)
            conc = -np.einsum("ij,ij->i", model.normals, radq / np.maximum(rq, 1e-12)[:, None])
            wall = ((np.abs(rq - r) < max(3 * tol, 0.03 * r, 0.1)) & (conc > 0.7) & ~model.synthetic)
            for key in ("lo", "hi"):
                if not unseen[key]:
                    continue
                if key == "lo":
                    cand = np.nonzero(wall & (hq < h_lo - 0.2) & (hq > h_lo - model.diag))[0]
                else:
                    cand = np.nonzero(wall & (hq > h_hi + 0.2) & (hq < h_hi + model.diag))[0]
                if len(cand) < 20:
                    continue
                phic = np.arctan2(Qc[cand] @ y0, Qc[cand] @ x0)
                if cover(phic) < 0.5:
                    continue
                ids = np.concatenate([ids, cand])
                h = np.concatenate([h, hq[cand]])
                real = np.concatenate([real, np.ones(len(cand), bool)])
                if key == "lo":
                    h_lo = float(hq[cand].min())
                    open_lo = bool(inside.signed([base + (h_lo - delta) * a])[0] > 0)
                    unseen["lo"] = end_unseen(h_lo)
                else:
                    h_hi = float(hq[cand].max())
                    open_hi = bool(inside.signed([base + (h_hi + delta) * a])[0] > 0)
                    unseen["hi"] = end_unseen(h_hi)
    if unseen["lo"]:
        open_lo = False
    if unseen["hi"]:
        open_hi = False
    if not open_lo and not open_hi:
        if unseen["lo"] and not unseen["hi"]:
            open_lo = True          # Gegenseite geschlossen (Boden erfasst) → Eintritt im Schatten
        elif unseen["hi"] and not unseen["lo"]:
            open_hi = True
        elif unseen["lo"] and unseen["hi"]:
            if a[2] >= 0:           # beide Enden ohne Daten → Standard: von oben gebohrt
                open_hi = True
            else:
                open_lo = True
        else:
            return None
    through = open_lo and open_hi
    used = {f.id for f in cyls}
    # Kopf-/Bodenteile unter den übrigen Features suchen
    heads, bottoms = [], []
    for f in others:
        if f.id in used or f.role != "surface":
            continue
        fids = _feature_faces(f, nF)
        if len(fids) == 0:
            continue
        hf = (model.centroids[fids] - base) @ a
        if f.kind == "cone" and _coaxial(a, base, f.params["axis"], f.params["apex"], 3.0, max(0.3, 0.1 * r)):
            if _concavity(model, fids, base, a) < -0.2:
                (heads if True else bottoms).append(f)
        elif f.kind == "cylinder" and _coaxial(a, base, f.params["axis"], f.params["point"], 3.0, max(0.3, 0.1 * r)):
            if f.params["radius"] > r * 1.15 and _concavity(model, fids, base, a) < -0.2:
                heads.append(f)
        elif f.kind == "plane" and abs(float(unit(f.params["normal"]) @ a)) > math.cos(math.radians(4)):
            # Ringfläche einer Flachsenkung oder flacher Boden: Punkte nahe der Achse
            rho = np.linalg.norm((model.centroids[fids] - base) - np.outer(hf, a), axis=1)
            near = rho < 2.5 * r
            if near.mean() > 0.8:
                bottoms.append(f)
        elif f.kind == "sphere":
            c = np.asarray(f.params["center"], float)
            if _coaxial(a, base, a, c, 1.0, max(0.3, 0.1 * r)) and abs(f.params["radius"] - r) < max(0.2, 0.08 * r):
                bottoms.append(f)
    # Kegelsenkung, die die Formerkennung nicht als eigenes Feature gefunden hat: freie Dreiecke
    # am offenen Ende mit fester Achse als Kegel fitten
    if not heads:
        taken = np.zeros(nF, bool)
        for f in others:
            if f.role == "surface":
                taken[_feature_faces(f, nF)] = True
        taken[ids] = True
        for key, hend, sgn in (("lo", h_lo, -1.0), ("hi", h_hi, 1.0)):
            if not (open_lo if key == "lo" else open_hi):
                continue
            hf = _csk_from_free_faces(model, base, a, r, hend, sgn, taken, tol)
            if hf is not None:
                heads.append(hf)
                taken[_feature_faces(hf, nF)] = True

    # Eintrittsseite und Bohrrichtung
    def head_side(f):
        fids = _feature_faces(f, nF)
        hm = float(np.mean((model.centroids[fids] - base) @ a))
        return "lo" if hm < 0.5 * (h_lo + h_hi) else "hi"

    if through:
        sides = [head_side(f) for f in heads if f.kind == "cone" or f.kind == "cylinder"]
        if sides:
            entry = max(set(sides), key=sides.count)
        else:
            entry = "hi" if a[2] >= 0 else "lo"          # Standard: von oben gebohrt
    else:
        entry = "hi" if open_hi else "lo"
    s = 1.0 if entry == "lo" else -1.0                  # Bohrrichtung = s·a
    adir = s * a
    h_entry = h_lo if entry == "lo" else h_hi
    owner = -np.ones(nF, np.int64)
    feat_by_id = {}
    for f in others:
        if f.role == "surface" and f.id not in used and f not in heads:
            fids = _feature_faces(f, nF)
            owner[fids] = f.id
            feat_by_id[f.id] = f

    def end_surface(side, h_end):
        """Fläche, in der die Bohrung an diesem Ende beginnt/endet: häufigstes Nachbar-Feature am
        Randband (Senkungen auf dieser Seite gehören zum Rand), Achsschnittpunkt auf dieser Seite."""
        band = min(max(1.0, 0.3 * r), max(0.35 * (h_hi - h_lo), 1e-3))
        mm = np.zeros(nF, bool)
        mm[ids[np.abs(h - h_end) < band]] = True
        for f in heads:
            if head_side(f) == side:
                mm[_feature_faces(f, nF)] = True
        ring = model.grow(mm, 4) & ~mm
        cand = owner[ring]
        cand = cand[cand >= 0]
        if not len(cand):
            return None, None
        vals, cnt = np.unique(cand, return_counts=True)
        for fid in vals[np.argsort(-cnt)]:
            fe = feat_by_id[int(fid)]
            if fe.kind not in ("plane", "cylinder", "cone", "sphere", "torus", "freeform"):
                continue
            try:
                t = _axis_hits_carrier(carrier_of(fe), base, a, h_end, max(5.0, 2 * r + 5))
            except Exception:
                t = None
            side_ok = t is not None and ((t - 0.5 * (h_lo + h_hi)) * (1.0 if side == "hi" else -1.0) > 0)
            if side_ok and abs(t - h_end) < max(6.0, 3 * r) + (20.0 if heads else 0.0):
                return t, fe.name
        return None, None

    t_entry, entry_name = end_surface(entry, h_entry)
    if t_entry is None:
        t_entry = h_entry
    if entry_name is None:
        # keine Eintrittsfläche als Feature (z. B. Klick auf rohen Scan): Eintritt am äußeren
        # Rand einer Senkung auf dieser Seite, sonst am Ende der Wand
        ext = [(model.centroids[_feature_faces(f, nF)] - base) @ a for f in heads if head_side(f) == entry]
        if ext:
            hh = np.concatenate(ext)
            t_entry = float(np.percentile(hh, 99.5)) if entry == "hi" else float(np.percentile(hh, 0.5))
    point = base + t_entry * a
    # Tiefe
    z_real = s * (h[real] - t_entry) if real.any() else s * (h - t_entry)
    seen = float(z_real.max()) if len(z_real) else 0.0
    p = {"point": point, "axis": adir, "d": 2 * r, "depth": seen, "bottom": "drill", "point_angle": 118.0,
         "head": "none", "head_d": 2 * r * 1.8, "head_angle": 90.0, "head_depth": 0.0, "thread": None,
         "bottom_seen": False, "seen_depth": seen}
    consumed = [f.id for f in cyls]
    notes = []
    if unseen["lo" if entry == "hi" else "hi"] and not through:
        notes.append(tr("Bohrungsende im Scan offen (Sichtschatten) – Grundloch angenommen, ggf. „durchgehend“ wählen",
                        "hole end open in scan (occlusion) – blind assumed, choose 'through' if needed"))
    if through:
        z_all = s * (h - t_entry)
        # Austritt: Fläche auf der Gegenseite, sonst Ende der Wand
        exit_side = "lo" if entry == "hi" else "hi"
        t_exit, _ = end_surface(exit_side, h_lo if exit_side == "lo" else h_hi)
        p["bottom"] = "through"
        p["depth"] = float(abs(t_exit - t_entry)) if t_exit is not None else float(z_all.max())
        p["bottom_seen"] = True
        notes.append(tr("durchgehend", "through"))
    # Senkungen
    for f in heads:
        fids = _feature_faces(f, nF)
        zf = s * ((model.centroids[fids] - base) @ a - t_entry)
        if np.median(zf) > 0.6 * max(seen, 1.0) and not through:
            # gehört eher zum Boden (Bohrerspitze)
            if f.kind == "cone":
                bottoms.append(f)
            continue
        if f.kind == "cone":
            al = float(f.params["half_angle"])
            apex = np.asarray(f.params["apex"], float)
            t_apex = float((apex - point) @ adir)
            if t_apex <= 0:        # Spitze muss in Bohrrichtung liegen
                continue
            R = t_apex * math.tan(al)
            if R <= r * 1.05:
                continue
            p.update(head="csk", head_d=2 * R, head_angle=math.degrees(2 * al))
            consumed.append(f.id)
        elif f.kind == "cylinder":
            R = float(f.params["radius"])
            zmax = float(np.percentile(zf, 98))
            p.update(head="cbore", head_d=2 * R, head_depth=max(zmax, 0.0))
            consumed.append(f.id)
    # Boden
    if not through:
        for f in bottoms:
            fids = _feature_faces(f, nF)
            zf = s * ((model.centroids[fids] - base) @ a - t_entry)
            if f.kind == "cone":
                al = float(f.params["half_angle"])
                apex = np.asarray(f.params["apex"], float)
                t_apex = float((apex - point) @ adir)
                if t_apex <= seen * 0.5:
                    continue
                p.update(bottom="drill", point_angle=math.degrees(2 * al),
                         depth=max(t_apex - r / max(math.tan(al), 1e-6), 0.0), bottom_seen=True)
                consumed.append(f.id)
            elif f.kind == "plane":
                if p.get("head") == "cbore" and abs(np.median(zf) - p["head_depth"]) < 0.5:
                    consumed.append(f.id)
                    continue
                if np.median(zf) < 0.5 * seen:
                    continue
                n = unit(f.params["normal"])
                t = float((np.asarray(f.params["origin"]) - point) @ n / (adir @ n))
                p.update(bottom="flat", depth=t, bottom_seen=True)
                consumed.append(f.id)
            elif f.kind == "sphere":
                c = np.asarray(f.params["center"], float)
                p.update(bottom="ball", depth=float((c - point) @ adir), bottom_seen=True)
                consumed.append(f.id)
        if not p["bottom_seen"]:
            notes.append(tr(f"Boden nicht erfasst – sichtbare Tiefe {seen:.2f} mm",
                            f"bottom not scanned – visible depth {seen:.2f} mm"))
    sn = snap_diameter(p["d"])
    if sn is not None and sn[2] is not None:
        notes.append(tr(f"passt zu Kernloch {sn[2][0]} – Gewinde?", f"matches tap drill {sn[2][0]} – thread?"))
    if entry_name:
        notes.append(tr(f"Eintritt auf {entry_name}", f"entry on {entry_name}"))
    return {"params": p, "consumed": consumed, "notes": notes}


def _cone_as_cylinder(model, f):
    from .features import Feature
    from .fitting import fit_cylinder
    ids = _feature_faces(f, model.n_faces)
    if len(ids) < 8:
        return None
    try:
        fr = fit_cylinder(model.centroids[ids], model.fit_normals[ids], model.weights[ids])
    except Exception:
        return None
    g = Feature("cylinder", dict(fr.params), "surface", ids, name=f.name)
    g.id = f.id
    return g


def detect_holes(model, features, tol=0.05, max_dia=None, log=None):
    """Konkave Zylinder -> Bohrungen. Rückgabe: Liste {params, consumed, face_ids, notes}."""
    log = log or (lambda s: None)
    nF = model.n_faces
    size = float(np.linalg.norm(model.bbox[1] - model.bbox[0]))
    max_dia = 0.35 * size if max_dia is None else max_dia
    cyl = []
    for f in features:
        if f.kind == "cone" and f.role == "surface" and f.params.get("half_angle", 1.0) < math.radians(3.0):
            # fast zylindrischer Kegel (z. B. schräg erfasste Bohrwand) → als Zylinder prüfen
            f = _cone_as_cylinder(model, f)
            if f is None:
                continue
        if f.kind != "cylinder" or f.role != "surface":
            continue
        ids = _feature_faces(f, nF)
        if len(ids) < 8 or 2 * f.params["radius"] > max_dia:
            continue
        a = unit(f.params["axis"])
        if _concavity(model, ids, np.asarray(f.params["point"], float), a) < -0.3:
            cyl.append(f)
    # koaxial + gleicher Radius gruppieren
    groups = []
    for f in sorted(cyl, key=lambda f: -len(f.face_ids)):
        for g in groups:
            g0 = g[0]
            if (_coaxial(g0.params["axis"], g0.params["point"], f.params["axis"], f.params["point"], 2.0,
                         max(0.15, 3 * tol))
                    and abs(g0.params["radius"] - f.params["radius"]) < max(0.06, 3 * tol)):
                g.append(f)
                break
        else:
            groups.append([f])
    inside = _InsideTest(model)
    out = []
    taken = set()
    for g in groups:
        # Bohrwand muss einen Großteil des Umfangs einnehmen; konkave Kantenrundungen in
        # Innenecken (~90°) sind keine Bohrungen
        ids = np.concatenate([_feature_faces(c, nF) for c in g])
        a = unit(g[0].params["axis"])
        x, y, _ = orthonormal_frame(a)
        rel = model.centroids[ids] - np.asarray(g[0].params["point"], float)
        ph = np.arctan2(rel @ y, rel @ x)
        nb = len(np.unique(np.floor((ph + math.pi) / (2 * math.pi) * 36).astype(int) % 36))
        if nb < 15:
            log(f"hole: coverage {10 * nb}° < 150° – skipped")
            continue
        try:
            res = _hole_from_cylinders(model, g, features, inside, tol, log)
        except Exception as e:  # noqa: BLE001
            log(f"hole: {e}")
            res = None
        if res is None:
            continue
        if taken & set(res["consumed"]):
            continue
        taken |= set(res["consumed"])
        res["face_ids"] = hole_face_ids(model, res["params"], tol=tol)
        rr = [float(c.stats.get("rms")) for c in g if getattr(c, "stats", None) and c.stats.get("rms") is not None]
        if rr:
            res["rms"] = min(rr)
        out.append(res)
    return out


def hole_from_faces(model, face_ids, features=(), tol=0.05):
    """Bohrung aus einer Auswahl (sauberer Wandteil) – Achse und Ø per Zylinderfit."""
    from .features import Feature
    from .fitting import fit_cylinder
    ids = np.asarray(face_ids, np.int64)
    real = ids[~model.synthetic[ids]] if (~model.synthetic[ids]).sum() >= 12 else ids
    fr = fit_cylinder(model.centroids[real], model.fit_normals[real], model.weights[real],
                      robust=max(2 * tol, 0.02))
    tmp = Feature("cylinder", dict(fr.params), "surface", real, name="tmp")
    a = unit(tmp.params["axis"])
    if _concavity(model, real, np.asarray(tmp.params["point"], float), a) > -0.2:
        raise ValueError(tr("Auswahl ist keine Bohrungswand (nicht konkav).",
                            "Selection is not a hole wall (not concave)."))
    inside = _InsideTest(model)
    others = [f for f in features if f.role == "surface"]
    # gleich große koaxiale Zylinder (andere Hälfte einer Durchgangsbohrung) mitnehmen
    group = [tmp] + [f for f in others if f.kind == "cylinder"
                     and _coaxial(a, tmp.params["point"], f.params["axis"], f.params["point"], 2.0, max(0.15, 3 * tol))
                     and abs(f.params["radius"] - tmp.params["radius"]) < max(0.06, 3 * tol)]
    res = _hole_from_cylinders(model, group, others, inside, tol, None)
    if res is None:
        raise ValueError(tr("Keine offene Seite gefunden – ist das Netz geschlossen?",
                            "No open end found – is the mesh closed?"))
    res["consumed"] = [c for c in res["consumed"] if c != tmp.id]
    res["face_ids"] = hole_face_ids(model, res["params"], tol=tol)
    res["rms"] = fr.rms
    return res


def hole_faces_near_click(model, face_id, tol=0.05, max_angle_deg=25.0, max_faces=20000):
    """Werkzeug K: vom angeklickten Dreieck aus die Bohrungswand wachsen lassen, danach
    Zylinderfit -> hole_from_faces.

    Reine Normalen-Kontinuität läuft bei Scans über verrundete Kanten auf die Nachbarflächen
    (jeder Schritt ist klein). Darum zweistufig: erst ein kleines Stück um den Klick, daraus
    Achse, Lage und Ø schätzen; dann nur Dreiecke aufnehmen, die zu diesem Zylinder passen
    (Abstand ≈ r, Normale ⟂ Achse und zur Achse zeigend) – zweimal nachgefittet."""
    from collections import deque

    from .fitting import fit_cylinder
    adj = model.adjacency
    N = model.normals
    C = model.centroids
    syn = model.synthetic
    cos_t = math.cos(math.radians(max_angle_deg))
    seed = int(face_id)

    def grow(ok, seeds):
        seen = set(int(s) for s in seeds)
        q = deque(seen)
        while q and len(seen) < max_faces:
            f = q.popleft()
            for g in adj.indices[adj.indptr[f]:adj.indptr[f + 1]]:
                g = int(g)
                if g in seen or syn[g] or not ok[g]:
                    continue
                if float(N[f] @ N[g]) >= cos_t:
                    seen.add(g)
                    q.append(g)
        return np.fromiter(seen, np.int64)

    def cylinder(ids):
        try:
            fr = fit_cylinder(C[ids], model.fit_normals[ids], model.weights[ids], robust=max(2 * tol, 0.02))
        except Exception:
            return None
        a = unit(fr.params["axis"])
        p = np.asarray(fr.params["point"], float)
        r = float(fr.params["radius"])
        if not np.isfinite(r) or r <= 0 or r > 0.5 * model.diag:
            return None
        D = C - p
        rad = D - np.outer(D @ a, a)
        rho = np.linalg.norm(rad, axis=1)
        concave = np.einsum("ij,ij->i", N, rad / np.maximum(rho, 1e-12)[:, None])
        if float(np.mean(concave[ids])) > -0.2:
            return None                                 # konvex/kein Zylinder
        return a, r, rho, concave

    el = model.mean_edge_length()
    dist = np.linalg.norm(C - C[seed], axis=1)
    ok_real = ~syn
    best = np.array([seed])
    # 1) Startstück um den Klick (nicht über Kontinuität, damit ein einzelnes schiefes
    #    Klick-Dreieck nicht stört). Erst großzügig, dann enger – nahe an verrundeten Kanten
    #    würden sonst Kantendreiecke die Achsschätzung verfälschen.
    for dev in (30.0, 18.0):
        R = max(4.0 * el, 1.0)
        for _ in range(6):
            disk = (dist < R) & ok_real
            nm = N[seed]
            for ang in (60.0, dev):
                sel = disk & ((N @ nm) > math.cos(math.radians(ang)))
                if sel.sum() == 0:
                    break
                nm = unit((N[sel] * model.areas[sel, None]).sum(0))
            patch = np.nonzero(disk & ((N @ nm) > math.cos(math.radians(dev))))[0]
            if len(patch) >= 30:
                ev = np.linalg.eigvalsh(np.cov(N[patch].T))
                if ev[1] > 0.01 * max(ev[2], 1e-12) + 4e-4:   # genug Krümmung für eine Achse
                    break
            R *= 1.6
        if len(patch) < 12:
            continue
        if len(patch) > len(best):
            best = patch
        cyl = cylinder(patch)
        if cyl is None:
            continue
        # 2) am geschätzten Zylinder entlang wachsen, zweimal nachfitten
        ids = patch
        for _ in range(3):
            a, r, rho, concave = cyl
            band = max(6.0 * tol, 0.04 * r, 0.15)
            ok = ok_real & (np.abs(rho - r) < band) & (np.abs(N @ a) < 0.5) & (concave < -0.6)
            new = grow(ok, ids[ok[ids]] if ok[ids].any() else ids)
            if len(new) <= len(ids) * 1.02:
                ids = new if len(new) >= len(ids) else ids
                break
            ids = new
            cyl = cylinder(ids)
            if cyl is None:
                break
        return ids
    return best
