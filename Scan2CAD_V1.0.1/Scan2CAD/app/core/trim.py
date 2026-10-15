"""Flächen gegeneinander verschneiden und anhand des Scans auswählen.

Ablauf (wie beim manuellen Flächenrückführen im CAD, hier automatisiert):
  1. Jede Fläche wird um 'ext' mm über ihren Auswahlrand hinaus verlängert, damit sich
     Nachbarflächen tatsächlich schneiden (die Auswahlränder lassen sonst Lücken).
  2. General Fuse (OCCT BOPAlgo_Builder): alle Flächen werden an allen gegenseitigen
     Schnittkurven aufgeteilt.
  3. Jedes Teilstück wird gegen den Scan geprüft (Abstand der Stützpunkte). Teilstücke ohne
     Scan-Bezug (Überstände, Kappen im Materialinneren) fliegen raus.
  4. Die übrigen Teilstücke werden vernäht; ist die Hülle geschlossen, entsteht ein Volumenkörper.
"""
from __future__ import annotations

import math

import numpy as np

from . import occ_build as ob
from .geom_utils import polygon_area_2d
from .occ_compat import to_face
from .profiles import offset_profile, profile_to_dense

TWO_PI = 2 * math.pi


def _ext_range(a, b, ext, period=TWO_PI):
    if b - a >= period - 1e-6:
        return 0.0, period
    a2, b2 = a - ext, b + ext
    if b2 - a2 >= period:
        return 0.0, period
    return a2, b2


def extended_params(kind, p, ext):
    """Parameter einer Fläche, um ext (mm) über den Rand hinaus verlängert."""
    q = dict(p)
    if kind == "plane":
        loops = p.get("loops") or []
        if loops and isinstance(loops[0], dict):
            new = [offset_profile(loops[0], +ext)]
            for l in loops[1:]:
                try:
                    a = abs(polygon_area_2d(profile_to_dense(l, 0.5)[0]))
                    lo = offset_profile(l, -ext)
                    a2 = abs(polygon_area_2d(profile_to_dense(lo, 0.5)[0]))
                    if a2 > 0.05 * a:
                        new.append(lo)
                except Exception:
                    pass
            q["loops"] = new
        elif loops:
            from .profiles import offset_profile as op
            outer = {"type": "polyline", "pts": loops[0], "closed": True}
            q["loops"] = [np.asarray(op({"type": "spline", "pts": loops[0], "closed": True}, ext)["pts"]).tolist()]
        return q
    if kind == "cylinder":
        R = max(float(p["radius"]), 1e-6)
        q["u0"], q["u1"] = _ext_range(float(p.get("u0", 0)), float(p.get("u1", TWO_PI)), ext / R)
        q["v0"], q["v1"] = float(p["v0"]) - ext, float(p["v1"]) + ext
        return q
    if kind == "cone":
        Rm = max(float(p["h1"]) * math.tan(float(p["half_angle"])), 1e-6)
        q["u0"], q["u1"] = _ext_range(float(p.get("u0", 0)), float(p.get("u1", TWO_PI)), ext / Rm)
        q["h0"] = max(0.0, float(p["h0"]) - ext)
        q["h1"] = float(p["h1"]) + ext
        return q
    if kind == "sphere":
        R = max(float(p["radius"]), 1e-6)
        q["u0"], q["u1"] = _ext_range(float(p.get("u0", 0)), float(p.get("u1", TWO_PI)), ext / R)
        q["v0"] = max(-math.pi / 2, float(p.get("v0", -math.pi / 2)) - ext / R)
        q["v1"] = min(math.pi / 2, float(p.get("v1", math.pi / 2)) + ext / R)
        return q
    if kind == "torus":
        q["u0"], q["u1"] = _ext_range(float(p.get("u0", 0)), float(p.get("u1", TWO_PI)), ext / max(p["R"], 1e-6))
        q["v0"], q["v1"] = _ext_range(float(p.get("v0", 0)), float(p.get("v1", TWO_PI)), ext / max(p["r"], 1e-6))
        return q
    if kind == "extrusion":
        q["z0"], q["z1"] = float(p["z0"]) - ext, float(p["z1"]) + ext
        return q
    if kind == "freeform" and p.get("trim_uv") is not None:
        T = np.asarray(p["trim_uv"], float)
        C = np.asarray(p["poles"], float)
        size = max(np.linalg.norm(C[-1, 0] - C[0, 0]), np.linalg.norm(C[0, -1] - C[0, 0]), 1e-6)
        c = T.mean(0)
        dirs = T - c
        nrm = np.linalg.norm(dirs, axis=1, keepdims=True)
        T2 = np.clip(T + dirs / np.maximum(nrm, 1e-12) * (ext / size), 0.0, 1.0)
        q["trim_uv"] = T2.tolist()
        return q
    return q


def _faces_of(shape):
    from OCP.TopAbs import TopAbs_FACE
    from OCP.TopExp import TopExp_Explorer
    out = []
    e = TopExp_Explorer(shape, TopAbs_FACE)
    while e.More():
        out.append(to_face(e.Current()))
        e.Next()
    return out


def _face_samples(face, n=300, seed=0):
    """Flächengleichverteilte Zufallspunkte auf der (tessellierten) Fläche."""
    from OCP.BRep import BRep_Tool
    from OCP.TopLoc import TopLoc_Location
    loc = TopLoc_Location()
    tri = BRep_Tool.Triangulation_s(face, loc)
    if tri is None:
        return np.zeros((0, 3)), 0.0
    tr = loc.Transformation()
    nn = tri.NbNodes()
    V = np.empty((nn, 3))
    for i in range(nn):
        p = tri.Node(i + 1).Transformed(tr)
        V[i] = (p.X(), p.Y(), p.Z())
    nt = tri.NbTriangles()
    Fi = np.empty((nt, 3), np.int64)
    for i in range(nt):
        Fi[i] = tri.Triangle(i + 1).Get()
    Fi -= 1
    A = V[Fi]
    ar = 0.5 * np.linalg.norm(np.cross(A[:, 1] - A[:, 0], A[:, 2] - A[:, 0]), axis=1)
    tot = float(ar.sum())
    if tot <= 0:
        return V, 0.0
    rng = np.random.default_rng(seed)
    k = rng.choice(nt, size=n, p=ar / tot)
    r1, r2 = rng.random(n), rng.random(n)
    s = np.sqrt(r1)
    P = (1 - s)[:, None] * A[k, 0] + (s * (1 - r2))[:, None] * A[k, 1] + (s * r2)[:, None] * A[k, 2]
    return P, tot


def scan_distance(model, P):
    """Betrag des Abstands von Punkten zur Scan-Oberfläche (VTK, ein Aufruf)."""
    import pyvista as pv
    from .preprocess import vf_to_pd
    if len(P) == 0:
        return np.zeros(0)
    tgt = getattr(model, "_pd_cache", None)
    if tgt is None:
        tgt = vf_to_pd(model.V, model.F)
        model._pd_cache = tgt
    out = pv.PolyData(np.asarray(P, float)).compute_implicit_distance(tgt)
    return np.abs(np.asarray(out.point_data["implicit_distance"]))


def split_and_select(args, model, keep_dist, fuzzy=1e-3, support=0.6, body_support=0.3,
                     lin_defl=None, log=None):
    """args: Liste (name, shape, rgb, is_body). Rückgabe: Liste (name, [faces], rgb, is_body)."""
    from OCP.BOPAlgo import BOPAlgo_Builder
    log = log or (lambda s: None)
    b = BOPAlgo_Builder()
    for _, shp, _, _ in args:
        b.AddArgument(shp)
    b.SetFuzzyValue(float(fuzzy))
    b.SetRunParallel(True)
    b.Perform()
    if b.HasErrors():
        raise RuntimeError("General Fuse fehlgeschlagen (Flächen evtl. tangential/degeneriert)")
    res = b.Shape()
    defl = lin_defl or max(0.05, 0.002 * (model.diag if model is not None else 100))
    from OCP.BRepMesh import BRepMesh_IncrementalMesh
    BRepMesh_IncrementalMesh(res, float(defl), False, 0.35, True)
    pieces = []  # (arg_index, face)
    for ai, (_, shp, _, _) in enumerate(args):
        for f in _faces_of(shp):
            imgs = list(b.Modified(f))
            if not imgs:
                if b.IsDeleted(f):
                    continue
                imgs = [f]
            for im in imgs:
                pieces.append((ai, to_face(im)))
    # Stützpunkte sammeln, ein Abstandsaufruf
    samples, slices, areas = [], [], []
    k = 0
    for ai, f in pieces:
        S, area = _face_samples(f)
        samples.append(S)
        slices.append((k, k + len(S)))
        areas.append(area)
        k += len(S)
    allS = np.vstack([s for s in samples if len(s)]) if k else np.zeros((0, 3))
    d = scan_distance(model, allS)
    out = {i: [] for i in range(len(args))}
    dropped = 0
    for (ai, f), (s0, s1), area in zip(pieces, slices, areas):
        if s1 <= s0:
            continue
        dd = d[s0:s1]
        frac = float(np.mean(dd < keep_dist))
        need = body_support if args[ai][3] else support
        if frac >= need:
            out[ai].append(f)
        else:
            dropped += 1
    log(f"Verschneiden: {len(pieces)} Teilflächen, {len(pieces) - dropped} behalten, {dropped} ohne Scan-Bezug verworfen")
    return [(args[i][0], out[i], args[i][2], args[i][3]) for i in range(len(args))]


def make_shell_or_solid(faces, tol, log=None):
    log = log or (lambda s: None)
    if not faces:
        return None
    from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeSolid, BRepBuilderAPI_Sewing
    from OCP.ShapeFix import ShapeFix_Solid
    from OCP.TopAbs import TopAbs_SHELL
    from OCP.TopExp import TopExp_Explorer
    from .occ_compat import to_shell
    best = None
    # schrittweise größere Nähtoleranz, bis die Hülle dicht ist (kleine Restlücken an Rundungen)
    for k in (1.0, 3.0, 6.0):
        sw = BRepBuilderAPI_Sewing(float(tol * k))
        for f in faces:
            sw.Add(f)
        sw.Perform()
        cand = (sw.NbFreeEdges(), sw.SewedShape(), tol * k)
        if best is None or cand[0] < best[0]:
            best = cand
        if cand[0] == 0:
            break
    free, sewn, used_tol = best
    if used_tol > tol:
        log(f"Nähtoleranz auf {used_tol:.3f} mm erhöht")
    shells = []
    e = TopExp_Explorer(sewn, TopAbs_SHELL)
    while e.More():
        shells.append(to_shell(e.Current()))
        e.Next()
    if free == 0 and len(shells) == 1:
        try:
            sol = BRepBuilderAPI_MakeSolid(shells[0]).Solid()
            fx = ShapeFix_Solid(sol)
            fx.Perform()
            sol = fx.Solid()
            if ob.is_valid(sol):
                log("Hülle geschlossen → Volumenkörper erzeugt")
                return sol
        except Exception as ex:  # noqa: BLE001
            log(f"Volumenkörper nicht möglich: {ex}")
    log(f"Hülle offen ({free} freie Kanten) → als Flächenverbund exportiert")
    return sewn


# ====================================================================== Regionen wachsen
PRIM = ("plane", "cylinder", "cone", "sphere", "torus", "ellipsoid")


def feature_distance(f, model, P):
    from .fitting import model_distance
    if f.kind in PRIM:
        return np.abs(model_distance(f.kind, f.params, P))
    import pyvista as pv
    from .preprocess import vf_to_pd
    V, F = ob.tessellate(f.shape(), max(0.02, 0.001 * model.diag))
    if len(F) == 0:
        return np.full(len(P), np.inf)
    d = pv.PolyData(np.asarray(P, float)).compute_implicit_distance(vf_to_pd(V, F))
    return np.abs(np.asarray(d.point_data["implicit_distance"]))


def grow_regions(model, feats, d_max, ang_deg=35.0):
    """Ordnet auch die nicht ausgewählten Übergangsdreiecke der jeweils nächsten Fläche zu
    (Voronoi-artig, nur zusammenhängend mit der Original-Auswahl). Schließt die Lücken,
    die durch die Auswahlränder entstehen."""
    from .fitting import model_normals
    nF = model.n_faces
    C, Nn = model.centroids, model.fit_normals
    owner = -np.ones(nF, np.int64)
    for i, f in enumerate(feats):
        ids = np.asarray(f.face_ids, np.int64)
        owner[ids[(ids >= 0) & (ids < nF)]] = i
    D = np.full((len(feats), nF), np.inf)
    cos_a = np.cos(np.radians(ang_deg))
    for i, f in enumerate(feats):
        try:
            d = feature_distance(f, model, C)
        except Exception:
            continue
        ok = d < d_max
        if f.kind in PRIM and ok.any():
            idx = np.nonzero(ok)[0]
            g = model_normals(f.kind, f.params, C[idx])
            ok[idx[np.abs(np.einsum("ij,ij->i", g, Nn[idx])) < cos_a]] = False
        D[i] = np.where(ok, d, np.inf)
    free = owner < 0
    best = np.argmin(D, axis=0)
    bestd = D[best, np.arange(nF)]
    assign = np.where(free & np.isfinite(bestd), best, owner)
    out = []
    for i, f in enumerate(feats):
        orig = np.asarray(f.face_ids, np.int64)
        orig = orig[(orig >= 0) & (orig < nF)]
        if len(orig) == 0:
            out.append(orig)
            continue
        cand = np.nonzero(assign == i)[0]
        out.append(model.component_containing(cand, orig))
    return out


# ====================================================================== (u,v)-Trimmung
def _uv_and_scale(kind, p, P):
    if kind == "cylinder":
        o, x, y, a = ob.frame3(p["point"], p["axis"], p.get("xdir"))
        r = P - o
        u = np.arctan2(r @ y, r @ x)
        v = r @ a
        return u, v, (float(p["radius"]), 1.0)
    if kind == "cone":
        apex, x, y, a = ob.frame3(p["apex"], p["axis"], p.get("xdir"))
        al = float(p["half_angle"])
        h0 = max(float(p["h0"]), 0.0)
        r = P - apex
        h = r @ a
        u = np.arctan2(r @ y, r @ x)
        v = (h - h0) / np.cos(al)
        Rm = max(float(np.median(h)) * np.tan(al), 1e-3)
        return u, v, (Rm, 1.0)
    if kind == "sphere":
        o, x, y, z = ob.frame3(p["center"], p.get("zdir", [0, 0, 1]), p.get("xdir"))
        r = P - o
        n = np.maximum(np.linalg.norm(r, axis=1), 1e-12)
        u = np.arctan2(r @ y, r @ x)
        v = np.arcsin(np.clip((r @ z) / n, -1, 1))
        R = float(p["radius"])
        return u, v, (R * max(np.cos(float(np.median(v))), 0.2), R)
    if kind == "torus":
        c, x, y, a = ob.frame3(p["center"], p["axis"], p.get("xdir"))
        r = P - c
        h = r @ a
        rad = r - np.outer(h, a)
        rho = np.linalg.norm(rad, axis=1)
        u = np.arctan2(r @ y, r @ x)
        v = np.arctan2(h, rho - float(p["R"]))
        vm = float(np.median(v))
        return u, v, (float(p["R"]) + float(p["r"]) * np.cos(vm), float(p["r"]))
    raise ValueError(kind)


def _geom_surface(kind, p):
    from OCP.Geom import (Geom_ConicalSurface, Geom_CylindricalSurface, Geom_SphericalSurface,
                          Geom_ToroidalSurface)
    from OCP.gp import gp_Ax3
    if kind == "cylinder":
        o, x, y, a = ob.frame3(p["point"], p["axis"], p.get("xdir"))
        return Geom_CylindricalSurface(gp_Ax3(ob.P(o), ob.D(a), ob.D(x)), float(p["radius"]))
    if kind == "cone":
        apex, x, y, a = ob.frame3(p["apex"], p["axis"], p.get("xdir"))
        al = float(p["half_angle"])
        h0 = max(float(p["h0"]), 0.0)
        return Geom_ConicalSurface(gp_Ax3(ob.P(apex + h0 * a), ob.D(a), ob.D(x)), al, h0 * np.tan(al))
    if kind == "sphere":
        return Geom_SphericalSurface(ob.ax3(p["center"], p.get("zdir", [0, 0, 1]), p.get("xdir")), float(p["radius"]))
    if kind == "torus":
        c, x, y, a = ob.frame3(p["center"], p["axis"], p.get("xdir"))
        return Geom_ToroidalSurface(gp_Ax3(ob.P(c), ob.D(a), ob.D(x)), float(p["R"]), float(p["r"]))
    raise ValueError(kind)


def uv_trimmed_face(kind, p, model, ids, smooth_tol):
    """Fläche eines Primitivs, getrimmt mit dem (geglätteten) Rand der Region im (u,v)-Raum.
    None, wenn die Region ringförmig ist oder die Konstruktion scheitert."""
    from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeEdge, BRepBuilderAPI_MakeFace, BRepBuilderAPI_MakeWire
    from OCP.BRepLib import BRepLib
    from OCP.Geom2dAPI import Geom2dAPI_Interpolate
    from OCP.gp import gp_Pnt2d
    from OCP.ShapeFix import ShapeFix_Face
    from .geom_utils import angular_range
    from .occ_compat import HArray1Pnt2d
    from .profiles import fit_spline_profile
    loops = model.boundary_loops(ids)
    if not loops:
        return None
    Pv = model.V[np.unique(model.F[ids])]
    u_all, _, _ = _uv_and_scale(kind, p, Pv)
    a0, a1 = angular_range(u_all, 20.0)
    if a1 - a0 >= 2 * np.pi - 1e-6:
        return None  # Ring -> rechteckige Bereiche genügen
    big = [l for l in loops if len(l) >= 0.15 * len(loops[0])]
    if len(big) > 1:
        return None
    loop = loops[0]
    u, v, (su, sv) = _uv_and_scale(kind, p, model.V[loop])
    um = 0.5 * (a0 + a1)
    u = um + (u - um + np.pi) % (2 * np.pi) - np.pi
    mm = np.c_[u * su, v * sv]
    spl = fit_spline_profile(mm, True, 0.8, max(smooth_tol, 1e-3), max(0.5, 15 * smooth_tol))
    q = np.asarray(spl["pts"], float)
    uv = np.c_[q[:, 0] / su, q[:, 1] / sv]
    if polygon_area_2d(uv) < 0:
        uv = uv[::-1]
    if len(uv) > 300:
        # periodische Interpolation löst ein volles N×N-System – bei langen Rändern Gigabytes
        k = np.linspace(0, len(uv), 300, endpoint=False).astype(int)
        uv = uv[k]
    srf = _geom_surface(kind, p)
    arr = HArray1Pnt2d(1, len(uv))
    for i, (a, b) in enumerate(uv):
        arr.SetValue(i + 1, gp_Pnt2d(float(a), float(b)))
    it = Geom2dAPI_Interpolate(arr, True, 1e-9)
    it.Perform()
    if not it.IsDone():
        return None
    edge = BRepBuilderAPI_MakeEdge(it.Curve(), srf).Edge()
    wire = BRepBuilderAPI_MakeWire(edge).Wire()
    face = BRepBuilderAPI_MakeFace(srf, wire, True).Face()
    BRepLib.BuildCurves3d_s(face)
    fx = ShapeFix_Face(face)
    fx.Perform()
    face = fx.Face()
    return face if ob.is_valid(face) else None


def surface_for_trimming(f, model, ids, rings, tol, ext):
    """Fläche zum Verschneiden: Region um 'rings' Dreiecksringe in die Nachbarn erweitert,
    Primitive mit (u,v)-Trimmung, Ebenen mit gefittetem Umriss."""
    from .features import trim_primitive
    if f.kind in ("plane", "cylinder", "cone", "sphere", "torus") and len(ids) >= 3:
        mask = np.zeros(model.n_faces, bool)
        mask[ids] = True
        idd = np.nonzero(model.grow(mask, rings))[0]
        base = {k: v for k, v in f.params.items() if k not in ("loops", "u0", "u1", "v0", "v1", "h0", "h1")}
        if f.kind == "plane":
            p = trim_primitive("plane", base, model, idd, {"outline": "fit", "outline_tol": 3 * tol, "tol": tol})
            return ob.build_shape("plane", p, "surface")
        p = trim_primitive(f.kind, base, model, idd, {})
        try:
            face = uv_trimmed_face(f.kind, p, model, idd, 2 * tol)
        except Exception:
            face = None
        if face is not None:
            return face
        return ob.build_shape(f.kind, extended_params(f.kind, p, 0.0), "surface")
    return ob.build_shape(f.kind, extended_params(f.kind, f.params, ext), "surface")
