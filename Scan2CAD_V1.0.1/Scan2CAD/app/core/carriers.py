"""Trägerflächen: einheitliche Sicht auf gefittete Flächen-Features.

Für jedes Flächen-Feature (Ebene, Zylinder, Kegel, Kugel, Torus, Freiform, Ellipsoid) liefert
`Carrier`:
  project(P)   nächster Punkt auf der (unbegrenzten bzw. verlängerten) Fläche
  distance(P)  vorzeichenbehafteter Abstand (positiv in Richtung der Flächennormalen)
  normal(P)    Flächennormale am Fußpunkt
  geom()       OCC Geom_Surface (analytisch unbegrenzt, Freiform um einen Rand verlängert)
Analytische Typen werden geschlossen in numpy gerechnet, Freiform über OCC.
"""
from __future__ import annotations

import math

import numpy as np

from .geom_utils import orthonormal_frame, unit

CARRIER_KINDS = ("plane", "cylinder", "cone", "sphere", "torus", "freeform", "ellipsoid")
PERIODIC_KINDS = ("cylinder", "cone", "sphere", "torus")


def _rad(q, a):
    h = q @ a
    radial = q - np.outer(h, a)
    rho = np.linalg.norm(radial, axis=1)
    return h, radial, rho


class Carrier:
    def __init__(self, kind, params, name="", ext=5.0):
        self.kind = kind
        self.p = params
        self.name = name
        self.ext = float(ext)
        self._geom = None
        self._proj = None
        self._tess = None

    # ------------------------------------------------------------------ Achse/Basis
    @property
    def axis(self):
        p = self.p
        if self.kind == "plane":
            return unit(p["normal"])
        if self.kind in ("cylinder", "cone", "torus"):
            return unit(p["axis"])
        if self.kind == "sphere":
            return unit(p.get("zdir", [0, 0, 1]))
        return None

    @property
    def base(self):
        p = self.p
        return np.asarray({"plane": p.get("origin"), "cylinder": p.get("point"), "cone": p.get("apex"),
                           "torus": p.get("center"), "sphere": p.get("center")}.get(self.kind,
                                                                                     np.zeros(3)), float)

    # ------------------------------------------------------------------ Projektion
    def project(self, P):
        P = np.atleast_2d(np.asarray(P, float))
        k, p = self.kind, self.p
        if k == "plane":
            n = unit(p["normal"])
            return P - np.outer((P - p["origin"]) @ n, n)
        if k == "sphere":
            c = np.asarray(p["center"])
            d = unit(P - c)
            return c + p["radius"] * d
        if k == "cylinder":
            a = unit(p["axis"])
            q = P - p["point"]
            h, radial, rho = _rad(q, a)
            radial = radial / np.maximum(rho, 1e-12)[:, None]
            return np.asarray(p["point"]) + np.outer(h, a) + p["radius"] * radial
        if k == "cone":
            a = unit(p["axis"])
            apex = np.asarray(p["apex"])
            al = float(p["half_angle"])
            q = P - apex
            h, radial, rho = _rad(q, a)
            e = radial / np.maximum(rho, 1e-12)[:, None]
            g = np.array([math.sin(al), math.cos(al)])        # Erzeugende in (rho, h)
            t = np.maximum(rho * g[0] + h * g[1], 0.0)
            return apex + np.outer(t * g[1], a) + (t * g[0])[:, None] * e
        if k == "torus":
            a = unit(p["axis"])
            c = np.asarray(p["center"])
            q = P - c
            h, radial, rho = _rad(q, a)
            e = radial / np.maximum(rho, 1e-12)[:, None]
            tc = c + p["R"] * e                               # Rohrmitte
            return tc + p["r"] * unit(P - tc)
        return self._project_occ(P)

    def distance(self, P):
        P = np.atleast_2d(np.asarray(P, float))
        Q = self.project(P)
        n = self.normal(Q)
        return np.einsum("ij,ij->i", P - Q, n)

    def normal(self, Q):
        Q = np.atleast_2d(np.asarray(Q, float))
        k, p = self.kind, self.p
        if k == "plane":
            return np.tile(unit(p["normal"]), (len(Q), 1))
        if k == "sphere":
            return unit(Q - p["center"])
        if k in ("cylinder", "cone", "torus"):
            from .fitting import model_normals
            return model_normals(k, p, Q)
        return self._normal_occ(Q)

    # ------------------------------------------------------------------ OCC
    def geom(self):
        if self._geom is None:
            self._geom = self._make_geom()
        return self._geom

    def _make_geom(self):
        from OCP.Geom import (Geom_ConicalSurface, Geom_CylindricalSurface, Geom_Plane,
                              Geom_SphericalSurface, Geom_ToroidalSurface)
        from OCP.gp import gp_Ax3
        from . import occ_build as ob
        k, p = self.kind, self.p
        if k == "plane":
            o, x, y, z = ob.frame3(p["origin"], p["normal"], p.get("xdir"))
            return Geom_Plane(gp_Ax3(ob.P(o), ob.D(z), ob.D(x)))
        if k == "cylinder":
            o, x, y, a = ob.frame3(p["point"], p["axis"], p.get("xdir"))
            return Geom_CylindricalSurface(gp_Ax3(ob.P(o), ob.D(a), ob.D(x)), float(p["radius"]))
        if k == "cone":
            apex, x, y, a = ob.frame3(p["apex"], p["axis"], p.get("xdir"))
            al = float(p["half_angle"])
            href = max(float(p.get("h1", 1.0)), 1e-3)
            return Geom_ConicalSurface(gp_Ax3(ob.P(apex + href * a), ob.D(a), ob.D(x)), al, href * math.tan(al))
        if k == "sphere":
            return Geom_SphericalSurface(ob.ax3(p["center"], p.get("zdir", [0, 0, 1]), p.get("xdir")),
                                         float(p["radius"]))
        if k == "torus":
            c, x, y, a = ob.frame3(p["center"], p["axis"], p.get("xdir"))
            return Geom_ToroidalSurface(gp_Ax3(ob.P(c), ob.D(a), ob.D(x)), float(p["R"]), float(p["r"]))
        if k == "freeform":
            from OCP.GeomLib import GeomLib
            srf = ob.bspline_surface(p)
            C = np.asarray(p["poles"], float)
            size = max(np.linalg.norm(C[-1, 0] - C[0, 0]), np.linalg.norm(C[0, -1] - C[0, 0]), 1.0)
            L = max(self.ext, 0.15 * size)
            for in_u in (True, False):
                for after in (True, False):
                    try:
                        GeomLib.ExtendSurfByLength_s(srf, float(L), 1, in_u, after)
                    except Exception:
                        pass
            return srf
        if k == "ellipsoid":
            from OCP.BRep import BRep_Tool
            from .trim import _faces_of
            shp = ob.build_ellipsoid(p, "surface")
            return BRep_Tool.Surface_s(_faces_of(shp)[0])
        raise ValueError(k)

    def _projector(self):
        if self._proj is None:
            from OCP.ShapeAnalysis import ShapeAnalysis_Surface
            self._proj = ShapeAnalysis_Surface(self.geom())
        return self._proj

    def _project_occ(self, P):
        from OCP.gp import gp_Pnt
        sas = self._projector()
        srf = self.geom()
        out = np.empty_like(P)
        prev = None
        for i, q in enumerate(P):
            g = gp_Pnt(float(q[0]), float(q[1]), float(q[2]))
            try:
                uv = sas.NextValueOfUV(prev, g, 1e-6, -1.0) if prev is not None else sas.ValueOfUV(g, 1e-6)
            except Exception:
                uv = sas.ValueOfUV(g, 1e-6)
            prev = uv
            r = srf.Value(uv.X(), uv.Y())
            out[i] = (r.X(), r.Y(), r.Z())
        return out

    def uv(self, P):
        from OCP.gp import gp_Pnt
        sas = self._projector()
        out = np.empty((len(P), 2))
        for i, q in enumerate(np.atleast_2d(P)):
            uv = sas.ValueOfUV(gp_Pnt(float(q[0]), float(q[1]), float(q[2])), 1e-6)
            out[i] = (uv.X(), uv.Y())
        return out

    def _normal_occ(self, Q):
        from OCP.GeomLProp import GeomLProp_SLProps
        srf = self.geom()
        uv = self.uv(Q)
        out = np.empty_like(Q)
        for i, (u, v) in enumerate(uv):
            pr = GeomLProp_SLProps(srf, float(u), float(v), 1, 1e-7)
            if pr.IsNormalDefined():
                n = pr.Normal()
                out[i] = (n.X(), n.Y(), n.Z())
            else:
                out[i] = (0, 0, 1)
        return out


def carrier_of(feature, ext=5.0):
    return Carrier(feature.kind, feature.params, feature.name, ext)


def is_carrier(feature):
    return feature.kind in CARRIER_KINDS and feature.role == "surface"


def project_points_to_feature(feature, P):
    if feature.kind in CARRIER_KINDS:
        return carrier_of(feature).project(P)
    # Körper-Features: über die Tessellierung
    import pyvista as pv
    from . import occ_build as ob
    from .preprocess import vf_to_pd
    V, F = ob.tessellate(feature.shape(), 0.01)
    pd = vf_to_pd(V, F)
    _, cp = pd.find_closest_cell(np.asarray(P, float), return_closest_point=True)
    return np.asarray(cp, float)


def split_plane(carrier, phi0=0.0):
    """Ebene durch die Achse eines periodischen Trägers (für die Naht-Teilung).
    Rückgabe: (Ursprung, Normale, Halbachse in der Ebene)."""
    a = carrier.axis if carrier.kind != "sphere" else unit(carrier.p.get("zdir", [0, 0, 1]))
    x, y, _ = orthonormal_frame(a, carrier.p.get("xdir"))
    d = math.cos(phi0) * x + math.sin(phi0) * y      # Richtung der Halbebene
    n = np.cross(a, d)
    return carrier.base, unit(n), unit(d)
