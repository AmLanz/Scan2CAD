"""Aufbau von OpenCASCADE-Geometrie aus Feature-Parametern, Tessellierung und STEP-Export."""
from __future__ import annotations

import math

import numpy as np
from OCP.BRep import BRep_Builder, BRep_Tool
from OCP.BRepAlgoAPI import BRepAlgoAPI_Cut, BRepAlgoAPI_Fuse
from OCP.BRepBuilderAPI import (BRepBuilderAPI_GTransform, BRepBuilderAPI_MakeEdge,
                                BRepBuilderAPI_MakeFace, BRepBuilderAPI_MakePolygon,
                                BRepBuilderAPI_MakeVertex, BRepBuilderAPI_MakeWire,
                                BRepBuilderAPI_Sewing)
from OCP.BRepCheck import BRepCheck_Analyzer
from OCP.BRepFilletAPI import BRepFilletAPI_MakeFillet2d
from OCP.BRepLib import BRepLib
from OCP.BRepMesh import BRepMesh_IncrementalMesh
from OCP.BRepPrimAPI import (BRepPrimAPI_MakeCone, BRepPrimAPI_MakeCylinder,
                             BRepPrimAPI_MakeHalfSpace, BRepPrimAPI_MakePrism,
                             BRepPrimAPI_MakeRevol, BRepPrimAPI_MakeSphere,
                             BRepPrimAPI_MakeTorus)
from OCP.BRepTools import BRepTools
from OCP.Geom import (Geom_BSplineSurface, Geom_ConicalSurface, Geom_CylindricalSurface,
                      Geom_SphericalSurface, Geom_ToroidalSurface)
from OCP.Geom2dAPI import Geom2dAPI_Interpolate
from OCP.GeomAPI import GeomAPI_Interpolate
from OCP.gp import (gp_Ax1, gp_Ax2, gp_Ax3, gp_Circ, gp_Dir, gp_GTrsf, gp_Mat, gp_Pln,
                    gp_Pnt, gp_Pnt2d, gp_Vec, gp_XYZ)
from OCP.ShapeFix import ShapeFix_Face, ShapeFix_Shape
from OCP.STEPControl import STEPControl_AsIs, STEPControl_Writer
from OCP.TopAbs import TopAbs_COMPOUND, TopAbs_EDGE, TopAbs_FACE, TopAbs_REVERSED, TopAbs_VERTEX
from OCP.TopExp import TopExp_Explorer
from OCP.TopLoc import TopLoc_Location
from OCP.TopoDS import TopoDS_Compound, TopoDS_Shape
from OCP.Interface import Interface_Static

from .geom_utils import orthonormal_frame, polygon_area_2d, unit
from .occ_compat import (Array1Int, Array1Real, Array2Pnt, HArray1Pnt, HArray1Pnt2d,
                         to_edge, to_face, to_vertex, to_wire)

TWO_PI = 2 * math.pi


# ------------------------------------------------------------------ Basics
def P(v):
    return gp_Pnt(float(v[0]), float(v[1]), float(v[2]))


def D(v):
    v = unit(np.asarray(v, float))
    return gp_Dir(float(v[0]), float(v[1]), float(v[2]))


def frame3(origin, z, x=None):
    xx, yy, zz = orthonormal_frame(np.asarray(z, float), None if x is None else np.asarray(x, float))
    return np.asarray(origin, float), xx, yy, zz


def ax3(origin, z, x=None):
    o, xx, yy, zz = frame3(origin, z, x)
    return gp_Ax3(P(o), D(zz), D(xx))


def ax2(origin, z, x=None):
    o, xx, yy, zz = frame3(origin, z, x)
    return gp_Ax2(P(o), D(zz), D(xx))


def is_valid(shape):
    try:
        return (not shape.IsNull()) and BRepCheck_Analyzer(shape).IsValid()
    except Exception:
        return False


def fix_shape(shape):
    try:
        sf = ShapeFix_Shape(shape)
        sf.Perform()
        return sf.Shape()
    except Exception:
        return shape


def compound(shapes):
    comp = TopoDS_Compound()
    b = BRep_Builder()
    b.MakeCompound(comp)
    for s in shapes:
        if s is not None and not s.IsNull():
            b.Add(comp, s)
    return comp


# ------------------------------------------------------------------ 2D-Profile -> Wire
def _to3d(pts2, o, x, y):
    pts2 = np.asarray(pts2, float)
    return o + pts2[:, :1] * x + pts2[:, 1:2] * y


def _polygon_wire(pts3, closed=True):
    mp = BRepBuilderAPI_MakePolygon()
    for p in pts3:
        mp.Add(P(p))
    if closed:
        mp.Close()
    return mp.Wire()


def _interp_edge(pts3, closed):
    pts3 = np.asarray(pts3, float)
    if closed and np.linalg.norm(pts3[0] - pts3[-1]) < 1e-9:
        pts3 = pts3[:-1]
    arr = HArray1Pnt(1, len(pts3))
    for i, p in enumerate(pts3):
        arr.SetValue(i + 1, P(p))
    it = GeomAPI_Interpolate(arr, bool(closed), 1e-7)
    it.Perform()
    if not it.IsDone():
        raise RuntimeError("Spline-Interpolation fehlgeschlagen")
    return BRepBuilderAPI_MakeEdge(it.Curve()).Edge()


def _unique_vertices(shape):
    verts = []
    exp = TopExp_Explorer(shape, TopAbs_VERTEX)
    seen = []
    while exp.More():
        v = to_vertex(exp.Current())
        p = BRep_Tool.Pnt_s(v)
        xyz = np.array([p.X(), p.Y(), p.Z()])
        if not any(np.linalg.norm(xyz - s) < 1e-9 for s in seen):
            seen.append(xyz)
            verts.append((v, xyz))
        exp.Next()
    return verts


def profile_face(prof, o, x, y, z):
    """Ebene Fläche aus einem geschlossenen Profil (2D im Rahmen o,x,y)."""
    t = prof["type"]
    if t == "circle":
        c = _to3d([prof["c"]], o, x, y)[0]
        circ = gp_Circ(gp_Ax2(P(c), D(z), D(x)), float(prof["r"]))
        w = BRepBuilderAPI_MakeWire(BRepBuilderAPI_MakeEdge(circ).Edge()).Wire()
        return BRepBuilderAPI_MakeFace(w, True).Face()
    if t == "poly":
        pts2 = np.asarray(prof["pts"], float)
        fil = list(prof.get("fillets") or [0.0] * len(pts2))
        if polygon_area_2d(pts2) < 0:
            pts2 = pts2[::-1]
            fil = fil[::-1]
        pts3 = _to3d(pts2, o, x, y)
        face = BRepBuilderAPI_MakeFace(_polygon_wire(pts3, True), True).Face()
        if any(r > 1e-6 for r in fil):
            mf = BRepFilletAPI_MakeFillet2d(face)
            verts = _unique_vertices(face)
            for (v, xyz) in verts:
                j = int(np.argmin(np.linalg.norm(pts3 - xyz, axis=1)))
                if fil[j] > 1e-6:
                    mf.AddFillet(v, float(fil[j]))
            mf.Build()
            if mf.IsDone():
                face = to_face(mf.Shape())
        return face
    if t in ("spline", "polyline"):
        pts3 = _to3d(prof["pts"], o, x, y)
        if t == "spline":
            w = BRepBuilderAPI_MakeWire(_interp_edge(pts3, True)).Wire()
        else:
            w = _polygon_wire(pts3, True)
        return BRepBuilderAPI_MakeFace(w, True).Face()
    raise ValueError(t)


def profile_wire_open(prof, o, x, y):
    pts3 = _to3d(prof["pts"], o, x, y)
    closed = prof.get("closed", False)
    if prof["type"] == "spline":
        return BRepBuilderAPI_MakeWire(_interp_edge(pts3, closed)).Wire()
    return _polygon_wire(pts3, closed)


def face_with_holes(outer_face, inner_faces):
    if not inner_faces:
        return outer_face
    ow = BRepTools.OuterWire_s(outer_face)
    mf = BRepBuilderAPI_MakeFace(ow, True)
    for f in inner_faces:
        iw = BRepTools.OuterWire_s(f)
        mf.Add(to_wire(iw.Reversed()))
    face = mf.Face()
    fx = ShapeFix_Face(face)
    fx.Perform()
    return fx.Face()


# ------------------------------------------------------------------ Builder je Typ
def build_plane(p, role):
    o, x, y, z = frame3(p["origin"], p["normal"], p.get("xdir"))
    if role == "cut":
        face = BRepBuilderAPI_MakeFace(gp_Pln(gp_Ax3(P(o), D(z), D(x)))).Face()
        ref = o + z * 1e5
        return BRepPrimAPI_MakeHalfSpace(face, P(ref)).Solid()
    raw = p.get("loops", [])
    if not raw:
        raise RuntimeError("Ebene ohne Umriss")
    if isinstance(raw[0], dict):
        from .geom_utils import point_in_polygon
        from .profiles import profile_to_dense
        faces = []
        dens = []
        for prof in raw:
            try:
                faces.append(profile_face(prof, o, x, y, z))
                dens.append(profile_to_dense(prof, 0.5)[0])
            except Exception:
                continue
        if faces:
            inner = [f for f, d in zip(faces[1:], dens[1:]) if point_in_polygon(d.mean(0), dens[0])]
            face = face_with_holes(faces[0], inner)
            if is_valid(face):
                return face
            return faces[0]
        raise RuntimeError("Ebenen-Umriss ungültig")
    loops = [np.asarray(l, float) for l in raw]
    outer = loops[0]
    if polygon_area_2d(outer) < 0:
        outer = outer[::-1]
    pln = gp_Pln(gp_Ax3(P(o), D(z), D(x)))
    mf = BRepBuilderAPI_MakeFace(pln, _polygon_wire(_to3d(outer, o, x, y)), True)
    for l in loops[1:]:
        if polygon_area_2d(l) > 0:
            l = l[::-1]
        mf.Add(_polygon_wire(_to3d(l, o, x, y)))
    face = mf.Face()
    fx = ShapeFix_Face(face)
    fx.Perform()
    face = fx.Face()
    if not is_valid(face):
        from scipy.spatial import ConvexHull
        hull = outer[ConvexHull(outer).vertices]
        face = BRepBuilderAPI_MakeFace(pln, _polygon_wire(_to3d(hull, o, x, y)), True).Face()
    return face


def build_sphere(p, role):
    if role in ("add", "sub"):
        return BRepPrimAPI_MakeSphere(P(p["center"]), float(p["radius"])).Shape()
    s = Geom_SphericalSurface(ax3(p["center"], p.get("zdir", [0, 0, 1]), p.get("xdir")), float(p["radius"]))
    v0 = max(float(p.get("v0", -math.pi / 2)), -math.pi / 2)
    v1 = min(float(p.get("v1", math.pi / 2)), math.pi / 2)
    return BRepBuilderAPI_MakeFace(s, float(p.get("u0", 0)), float(p.get("u1", TWO_PI)), v0, v1, 1e-7).Face()


def build_cylinder(p, role):
    o, x, y, a = frame3(p["point"], p["axis"], p.get("xdir"))
    v0, v1 = float(p["v0"]), float(p["v1"])
    R = float(p["radius"])
    if role in ("add", "sub"):
        return BRepPrimAPI_MakeCylinder(gp_Ax2(P(o + v0 * a), D(a), D(x)), R, v1 - v0).Shape()
    s = Geom_CylindricalSurface(gp_Ax3(P(o), D(a), D(x)), R)
    return BRepBuilderAPI_MakeFace(s, float(p.get("u0", 0)), float(p.get("u1", TWO_PI)), v0, v1, 1e-7).Face()


def build_cone(p, role):
    apex, x, y, a = frame3(p["apex"], p["axis"], p.get("xdir"))
    al = float(p["half_angle"])
    h0, h1 = max(float(p["h0"]), 0.0), float(p["h1"])
    if role in ("add", "sub"):
        R1, R2 = h0 * math.tan(al), h1 * math.tan(al)
        return BRepPrimAPI_MakeCone(gp_Ax2(P(apex + h0 * a), D(a), D(x)), R1, R2, h1 - h0).Shape()
    s = Geom_ConicalSurface(gp_Ax3(P(apex + h0 * a), D(a), D(x)), al, h0 * math.tan(al))
    vmax = (h1 - h0) / math.cos(al)
    return BRepBuilderAPI_MakeFace(s, float(p.get("u0", 0)), float(p.get("u1", TWO_PI)), 0.0, vmax, 1e-7).Face()


def build_torus(p, role):
    c, x, y, a = frame3(p["center"], p["axis"], p.get("xdir"))
    if role in ("add", "sub"):
        return BRepPrimAPI_MakeTorus(gp_Ax2(P(c), D(a), D(x)), float(p["R"]), float(p["r"])).Shape()
    s = Geom_ToroidalSurface(gp_Ax3(P(c), D(a), D(x)), float(p["R"]), float(p["r"]))
    return BRepBuilderAPI_MakeFace(s, float(p.get("u0", 0)), float(p.get("u1", TWO_PI)),
                                   float(p.get("v0", 0)), float(p.get("v1", TWO_PI)), 1e-7).Face()


def build_ellipsoid(p, role):
    sph = BRepPrimAPI_MakeSphere(gp_Pnt(0, 0, 0), 1.0).Shape()
    Rm = np.asarray(p["axes"], float)
    if np.linalg.det(Rm) < 0:
        Rm = Rm.copy()
        Rm[2] = -Rm[2]
    M = Rm.T @ np.diag(np.asarray(p["radii"], float))
    g = gp_GTrsf()
    g.SetVectorialPart(gp_Mat(*[float(v) for v in M.ravel()]))
    c = p["center"]
    g.SetTranslationPart(gp_XYZ(float(c[0]), float(c[1]), float(c[2])))
    shp = BRepBuilderAPI_GTransform(sph, g, True).Shape()
    if role in ("add", "sub"):
        return shp
    exp = TopExp_Explorer(shp, TopAbs_FACE)
    faces = []
    while exp.More():
        faces.append(exp.Current())
        exp.Next()
    return faces[0] if len(faces) == 1 else compound(faces)


def bspline_surface(p):
    C = np.asarray(p["poles"], float)
    nu, nv = C.shape[:2]
    k = int(p.get("degree", 3))
    poles = Array2Pnt(1, nu, 1, nv)
    for i in range(nu):
        for j in range(nv):
            poles.SetValue(i + 1, j + 1, P(C[i, j]))

    def knots(t):
        t = np.asarray(t, float)
        u, m = np.unique(np.round(t, 12), return_counts=True)
        K = Array1Real(1, len(u))
        Mm = Array1Int(1, len(u))
        for i, (uu, mm) in enumerate(zip(u, m)):
            K.SetValue(i + 1, float(uu))
            Mm.SetValue(i + 1, int(mm))
        return K, Mm

    ku, mu = knots(p["knots_u"])
    kv, mv = knots(p["knots_v"])
    return Geom_BSplineSurface(poles, ku, kv, mu, mv, k, k)


def build_freeform(p, role):
    srf = bspline_surface(p)
    trim = p.get("trim_uv")
    if trim is not None and len(trim) >= 4:
        try:
            T = np.asarray(trim, float)
            arr = HArray1Pnt2d(1, len(T))
            for i, (u, v) in enumerate(T):
                arr.SetValue(i + 1, gp_Pnt2d(float(u), float(v)))
            it = Geom2dAPI_Interpolate(arr, True, 1e-9)
            it.Perform()
            edge = BRepBuilderAPI_MakeEdge(it.Curve(), srf).Edge()
            wire = BRepBuilderAPI_MakeWire(edge).Wire()
            face = BRepBuilderAPI_MakeFace(srf, wire, True).Face()
            BRepLib.BuildCurves3d_s(face)
            fx = ShapeFix_Face(face)
            fx.Perform()
            face = fx.Face()
            if is_valid(face):
                return face
        except Exception:
            pass
    return BRepBuilderAPI_MakeFace(srf, 1e-7).Face()


def build_extrusion(p, role):
    o, x, y, z = frame3(p["origin"], p["dir"], p.get("xdir"))
    z0, z1 = float(p["z0"]), float(p["z1"])
    loops = p["loops"]
    base = o + z0 * z
    vec = gp_Vec(*(z * (z1 - z0)))
    if p.get("open") or role == "surface":
        shells = []
        for prof in loops:
            if prof["type"] == "circle" or prof["type"] == "poly":
                f = profile_face(prof, base, x, y, z)
                w = BRepTools.OuterWire_s(f)
            else:
                w = profile_wire_open(prof, base, x, y)
            shells.append(BRepPrimAPI_MakePrism(w, vec).Shape())
        return shells[0] if len(shells) == 1 else compound(shells)
    faces = [profile_face(prof, base, x, y, z) for prof in loops]
    outer, inners = faces[0], faces[1:]
    face = face_with_holes(outer, inners)
    return BRepPrimAPI_MakePrism(face, vec).Shape()


def fillet_wire(pts3, fillets, closed):
    """Linienzug mit Eckenradien als Wire aus Geraden und Kreisbögen."""
    from OCP.GC import GC_MakeArcOfCircle
    pts3 = np.asarray(pts3, float)
    n = len(pts3)
    fillets = list(fillets) if fillets is not None else [0.0] * n
    corner_pts = []  # je Ecke: (Eintritt, Bogenmitte, Austritt) oder Punkt
    for i in range(n):
        r = fillets[i] if i < len(fillets) else 0.0
        if (not closed and (i == 0 or i == n - 1)) or r <= 1e-9:
            corner_pts.append((pts3[i], None, pts3[i]))
            continue
        a, b = pts3[i - 1] - pts3[i], pts3[(i + 1) % n] - pts3[i]
        la, lb = np.linalg.norm(a), np.linalg.norm(b)
        u1, u2 = a / la, b / lb
        th = math.acos(float(np.clip(u1 @ u2, -1, 1)))
        if th < 1e-3 or th > math.pi - 1e-3:
            corner_pts.append((pts3[i], None, pts3[i]))
            continue
        t = r / math.tan(th / 2)
        bis = unit(u1 + u2)
        c = pts3[i] + bis * (r / math.sin(th / 2))
        pa, pb = pts3[i] + u1 * t, pts3[i] + u2 * t
        mid = c + unit(pts3[i] - c) * r
        corner_pts.append((pa, mid, pb))
    mw = BRepBuilderAPI_MakeWire()
    m = n if closed else n - 1
    for i in range(n):
        pa, mid, pb = corner_pts[i]
        if mid is not None:
            arc = GC_MakeArcOfCircle(P(pa), P(mid), P(pb)).Value()
            mw.Add(BRepBuilderAPI_MakeEdge(arc).Edge())
        if i < m:
            nxt = corner_pts[(i + 1) % n][0]
            if np.linalg.norm(nxt - pb) > 1e-9:
                mw.Add(BRepBuilderAPI_MakeEdge(P(pb), P(nxt)).Edge())
    return mw.Wire()


def _closed_revolve_profile(prof):
    """Offenes (r,z)-Profil zur Achse hin schließen -> (pts, fillets)."""
    pts = np.asarray(prof["pts"], float)
    fil = list(prof.get("fillets") or [0.0] * len(pts))
    seq, fseq = [], []
    if pts[0, 0] > 1e-6:
        seq.append([0.0, pts[0, 1]])
        fseq.append(0.0)
    seq += pts.tolist()
    fseq += [0.0] + fil[1:-1] + [0.0] if len(pts) > 1 else [0.0]
    if pts[-1, 0] > 1e-6:
        seq.append([0.0, pts[-1, 1]])
        fseq.append(0.0)
    return np.asarray(seq), fseq


def build_revolve(p, role):
    q, x, y, a = frame3(p["point"], p["axis"], p.get("xdir"))
    prof = dict(p["profile"])  # Punkte als (r, z)
    ang = float(p.get("angle", TWO_PI))
    axis = gp_Ax1(P(q), D(a))
    t = prof["type"]
    closed = prof.get("closed", t in ("poly", "circle"))
    solid = role in ("add", "sub")
    if t == "spline" and not closed:
        pts = np.asarray(prof["pts"], float)
        e_spl = _interp_edge(_to3d(pts, q, x, a), False)
        if not solid:
            return BRepPrimAPI_MakeRevol(BRepBuilderAPI_MakeWire(e_spl).Wire(), axis, ang).Shape()
        mw = BRepBuilderAPI_MakeWire()
        p_first, p_last = _to3d(pts[[0, -1]], q, x, a)
        a_first, a_last = q + pts[0, 1] * a, q + pts[-1, 1] * a
        mw.Add(e_spl)
        if pts[-1, 0] > 1e-6:
            mw.Add(BRepBuilderAPI_MakeEdge(P(p_last), P(a_last)).Edge())
        mw.Add(BRepBuilderAPI_MakeEdge(P(a_last), P(a_first)).Edge())
        if pts[0, 0] > 1e-6:
            mw.Add(BRepBuilderAPI_MakeEdge(P(a_first), P(p_first)).Edge())
        face = BRepBuilderAPI_MakeFace(mw.Wire(), True).Face()
        return BRepPrimAPI_MakeRevol(face, axis, ang).Shape()
    if t in ("poly_open", "polyline") and not closed:
        if not solid:
            w = fillet_wire(_to3d(prof["pts"], q, x, a), prof.get("fillets"), False)
            return BRepPrimAPI_MakeRevol(w, axis, ang).Shape()
        seq, fseq = _closed_revolve_profile(prof)
        w = fillet_wire(_to3d(seq, q, x, a), fseq, True)
        face = BRepBuilderAPI_MakeFace(w, True).Face()
        return BRepPrimAPI_MakeRevol(face, axis, ang).Shape()
    # geschlossenes Profil (Ring o. ä.)
    pr = dict(prof)
    if t == "polyline":
        pr["type"] = "poly"
    face = profile_face(pr, q, x, a, np.cross(x, a))
    if solid:
        return BRepPrimAPI_MakeRevol(face, axis, ang).Shape()
    return BRepPrimAPI_MakeRevol(BRepTools.OuterWire_s(face), axis, ang).Shape()


BUILDERS = {
    "plane": build_plane, "sphere": build_sphere, "cylinder": build_cylinder,
    "cone": build_cone, "torus": build_torus, "ellipsoid": build_ellipsoid,
    "freeform": build_freeform, "extrusion": build_extrusion, "revolve": build_revolve,
}


def _build_hole(p, role):
    from .bores import build_hole
    return build_hole(p, role)


BUILDERS["hole"] = _build_hole


def build_shape(kind, params, role):
    shp = BUILDERS[kind](params, role)
    if shp is None or shp.IsNull():
        raise RuntimeError("Leere Geometrie")
    return shp


# ------------------------------------------------------------------ Tessellierung
def tessellate(shape, lin_defl=0.05, ang_defl=0.3):
    BRepMesh_IncrementalMesh(shape, float(lin_defl), False, float(ang_defl), True)
    Vs, Fs = [], []
    off = 0
    exp = TopExp_Explorer(shape, TopAbs_FACE)
    while exp.More():
        face = to_face(exp.Current())
        loc = TopLoc_Location()
        tri = BRep_Tool.Triangulation_s(face, loc)
        if tri is not None:
            tr = loc.Transformation()
            nn = tri.NbNodes()
            V = np.empty((nn, 3))
            for i in range(nn):
                pnt = tri.Node(i + 1).Transformed(tr)
                V[i] = (pnt.X(), pnt.Y(), pnt.Z())
            nt = tri.NbTriangles()
            F = np.empty((nt, 3), np.int64)
            for i in range(nt):
                F[i] = tri.Triangle(i + 1).Get()
            F -= 1
            if face.Orientation() == TopAbs_REVERSED:
                F = F[:, ::-1]
            Vs.append(V)
            Fs.append(F + off)
            off += nn
        exp.Next()
    if not Vs:
        return np.zeros((0, 3)), np.zeros((0, 3), np.int64)
    return np.vstack(Vs), np.vstack(Fs)



def tessellate_faces(shape, lin_defl=0.05, ang_defl=0.3):
    """Wie tessellate, zusätzlich je Dreieck der Index der B-Rep-Fläche und je Fläche
    (Flächentyp, Kantenzahl, Fläche). Typen: plane, cylinder, cone, sphere, torus, bspline,
    revolution, extrusion, other."""
    from OCP.BRepAdaptor import BRepAdaptor_Surface
    from OCP.BRepGProp import BRepGProp
    from OCP.GProp import GProp_GProps
    from OCP.GeomAbs import (GeomAbs_BezierSurface, GeomAbs_BSplineSurface, GeomAbs_Cone,
                             GeomAbs_Cylinder, GeomAbs_Plane, GeomAbs_Sphere,
                             GeomAbs_SurfaceOfExtrusion, GeomAbs_SurfaceOfRevolution, GeomAbs_Torus)
    names = {GeomAbs_Plane: "plane", GeomAbs_Cylinder: "cylinder", GeomAbs_Cone: "cone",
             GeomAbs_Sphere: "sphere", GeomAbs_Torus: "torus", GeomAbs_BSplineSurface: "bspline",
             GeomAbs_BezierSurface: "bspline", GeomAbs_SurfaceOfRevolution: "revolution",
             GeomAbs_SurfaceOfExtrusion: "extrusion"}
    BRepMesh_IncrementalMesh(shape, float(lin_defl), False, float(ang_defl), True)
    Vs, Fs, Is, info = [], [], [], []
    off = 0
    exp = TopExp_Explorer(shape, TopAbs_FACE)
    while exp.More():
        face = to_face(exp.Current())
        try:
            typ = names.get(BRepAdaptor_Surface(face).GetType(), "other")
        except Exception:  # noqa: BLE001
            typ = "other"
        ne = 0
        ee = TopExp_Explorer(face, TopAbs_EDGE)
        while ee.More():
            ne += 1
            ee.Next()
        g = GProp_GProps()
        try:
            BRepGProp.SurfaceProperties_s(face, g)
            ar = abs(g.Mass())
        except Exception:  # noqa: BLE001
            ar = 0.0
        k = len(info)
        info.append((typ, ne, ar))
        loc = TopLoc_Location()
        tri = BRep_Tool.Triangulation_s(face, loc)
        if tri is not None:
            tr = loc.Transformation()
            nn = tri.NbNodes()
            V = np.empty((nn, 3))
            for i in range(nn):
                pnt = tri.Node(i + 1).Transformed(tr)
                V[i] = (pnt.X(), pnt.Y(), pnt.Z())
            nt = tri.NbTriangles()
            F = np.empty((nt, 3), np.int64)
            for i in range(nt):
                F[i] = tri.Triangle(i + 1).Get()
            F -= 1
            if face.Orientation() == TopAbs_REVERSED:
                F = F[:, ::-1]
            Vs.append(V)
            Fs.append(F + off)
            Is.append(np.full(nt, k, np.int64))
            off += nn
        exp.Next()
    if not Vs:
        return np.zeros((0, 3)), np.zeros((0, 3), np.int64), np.zeros(0, np.int64), info
    return np.vstack(Vs), np.vstack(Fs), np.concatenate(Is), info

# ------------------------------------------------------------------ Netz -> facettiert
def mesh_to_faceted(V, F, make_solid=True, progress=None):
    """Dreiecksnetz als B-Rep mit gemeinsam genutzten Kanten (schneller als Sewing)."""
    from OCP.TopoDS import TopoDS_Shell
    b = BRep_Builder()
    verts = [BRepBuilderAPI_MakeVertex(P(v)).Vertex() for v in V]
    edges = {}
    shell = TopoDS_Shell()
    b.MakeShell(shell)
    nF = len(F)
    for k, (i, j, l) in enumerate(F):
        if i == j or j == l or l == i:
            continue
        try:
            mw = BRepBuilderAPI_MakeWire()
            for a, c in ((i, j), (j, l), (l, i)):
                key = (a, c) if a < c else (c, a)
                e = edges.get(key)
                if e is None:
                    e = BRepBuilderAPI_MakeEdge(verts[key[0]], verts[key[1]]).Edge()
                    edges[key] = e
                mw.Add(e)
            if not mw.IsDone():
                continue
            mf = BRepBuilderAPI_MakeFace(mw.Wire(), True)
            if mf.IsDone():
                b.Add(shell, mf.Face())
        except Exception:
            continue
        if progress and k % 5000 == 0:
            progress(k / nF)
    if make_solid:
        try:
            from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeSolid
            ms = BRepBuilderAPI_MakeSolid(shell)
            sol = ms.Solid()
            fx = ShapeFix_Shape(sol)
            fx.Perform()
            return fx.Shape()
        except Exception:
            pass
    return shell


# ------------------------------------------------------------------ Boolesch / Nähen
def _bool_op(cls, a, b, fuzzy):
    from .occ_compat import ListOfShape
    op = cls()
    la, lb = ListOfShape(), ListOfShape()
    la.Append(a)
    lb.Append(b)
    op.SetArguments(la)
    op.SetTools(lb)
    if fuzzy and fuzzy > 0:
        op.SetFuzzyValue(float(fuzzy))
    op.SetRunParallel(True)
    op.Build()
    if not op.IsDone():
        raise RuntimeError("Boolesche Operation fehlgeschlagen")
    return op.Shape()


def fuse_all(shapes, fuzzy=0.0):
    """Vereinigung; 'fuzzy' (mm) lässt Körper verschmelzen, die sich nur fast berühren
    (z. B. Zapfen 0,02 mm über der Grundfläche – typisch bei gefitteten Scans)."""
    res = None
    for s in shapes:
        if res is None:
            res = s
            continue
        try:
            res = _bool_op(BRepAlgoAPI_Fuse, res, s, fuzzy)
        except Exception:
            res = BRepAlgoAPI_Fuse(res, s).Shape()
    return res


def _solids_of(shape):
    from OCP.TopAbs import TopAbs_SOLID
    from .occ_compat import to_solid
    out = []
    e = TopExp_Explorer(shape, TopAbs_SOLID)
    while e.More():
        out.append(to_solid(e.Current()))
        e.Next()
    return out


def cut(a, b, fuzzy=0.0):
    # Werkzeug-Verbund aus sich überlappenden Körpern (z. B. Bohrung + Gewinderille) nacheinander
    if b.ShapeType() == TopAbs_COMPOUND:
        parts = _solids_of(b)
        if len(parts) > 1:
            for i, t in enumerate(parts):
                a = cut(a, t, fuzzy if i == 0 else 0.0)
            return a
    try:
        return _bool_op(BRepAlgoAPI_Cut, a, b, fuzzy)
    except Exception:
        return BRepAlgoAPI_Cut(a, b).Shape()


def sew(shapes, tol=0.05):
    sw = BRepBuilderAPI_Sewing(float(tol))
    for s in shapes:
        sw.Add(s)
    sw.Perform()
    return sw.SewedShape()


# ------------------------------------------------------------------ STEP
def write_step(items, path, schema="AP214IS"):
    """items: Liste (name, shape, rgb|None). Versucht XCAF (Namen+Farben), sonst einfacher Writer."""
    Interface_Static.SetCVal_s("write.step.schema", schema)
    Interface_Static.SetCVal_s("write.step.unit", "MM")
    try:
        from OCP.Quantity import Quantity_Color, Quantity_TypeOfColor
        from OCP.STEPCAFControl import STEPCAFControl_Writer
        from OCP.TCollection import TCollection_ExtendedString
        from OCP.TDataStd import TDataStd_Name
        from OCP.TDocStd import TDocStd_Document
        from OCP.XCAFApp import XCAFApp_Application
        from OCP.XCAFDoc import XCAFDoc_ColorType, XCAFDoc_DocumentTool

        app = XCAFApp_Application.GetApplication_s()
        doc = TDocStd_Document(TCollection_ExtendedString("XmlXCAF"))
        app.InitDocument(doc)
        st = XCAFDoc_DocumentTool.ShapeTool_s(doc.Main())
        ct = XCAFDoc_DocumentTool.ColorTool_s(doc.Main())
        for name, shape, rgb in items:
            lab = st.AddShape(shape, False)
            TDataStd_Name.Set_s(lab, TCollection_ExtendedString(str(name)))
            if rgb is not None:
                col = Quantity_Color(float(rgb[0]), float(rgb[1]), float(rgb[2]), Quantity_TypeOfColor.Quantity_TOC_RGB)
                ct.SetColor(lab, col, XCAFDoc_ColorType.XCAFDoc_ColorSurf)
        w = STEPCAFControl_Writer()
        w.SetNameMode(True)
        w.SetColorMode(True)
        if not w.Transfer(doc, STEPControl_AsIs):
            raise RuntimeError("XCAF-Transfer fehlgeschlagen")
        st_ = w.Write(str(path))
        if "RetDone" not in str(st_):
            raise RuntimeError(str(st_))
        return True
    except Exception:
        w = STEPControl_Writer()
        w.Transfer(compound([s for _, s, _ in items]), STEPControl_AsIs)
        st_ = w.Write(str(path))
        if "RetDone" not in str(st_):
            raise RuntimeError(f"STEP-Export fehlgeschlagen: {st_}")
        return False


# ------------------------------------------------------------------ Hilfen für Vorschau/Export
def transform_shape(shape, T):
    """Starre 4x4-Transformation auf eine Form anwenden."""
    from OCP.BRepBuilderAPI import BRepBuilderAPI_Transform
    from OCP.gp import gp_Trsf
    tr = gp_Trsf()
    M = np.asarray(T, float)
    tr.SetValues(*[float(M[i, j]) for i in range(3) for j in range(4)])
    return BRepBuilderAPI_Transform(shape, tr, True).Shape()


def edge_polylines(shape, lin_defl=0.05, ang_defl=0.25, max_edges=20000):
    """B-Rep-Kanten als Polylinien (jede Kante einmal)."""
    from OCP.BRepAdaptor import BRepAdaptor_Curve
    from OCP.GCPnts import GCPnts_TangentialDeflection
    from OCP.TopAbs import TopAbs_EDGE
    seen = set()
    out = []
    exp = TopExp_Explorer(shape, TopAbs_EDGE)
    while exp.More():
        e = to_edge(exp.Current())
        exp.Next()
        h = hash(e)
        if h in seen:
            continue
        seen.add(h)
        if len(seen) > max_edges:
            break
        try:
            if BRep_Tool.Degenerated_s(e):
                continue
            ad = BRepAdaptor_Curve(e)
            td = GCPnts_TangentialDeflection(ad, float(ang_defl), float(lin_defl))
            n = td.NbPoints()
            if n < 2:
                continue
            pts = np.array([[td.Value(i).X(), td.Value(i).Y(), td.Value(i).Z()] for i in range(1, n + 1)])
            out.append(pts)
        except Exception:
            continue
    return out


def count_faces(shape):
    n = 0
    exp = TopExp_Explorer(shape, TopAbs_FACE)
    while exp.More():
        n += 1
        exp.Next()
    return n
