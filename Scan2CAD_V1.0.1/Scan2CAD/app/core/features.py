"""Feature-Datenmodell: gefittete Geometrie + zugeordnete Netzflächen + Rolle im Modell."""
from __future__ import annotations

import colorsys
import copy
from dataclasses import dataclass, field

import numpy as np

from . import fitting as fit
from .freeform import fit_freeform_patch
from .i18n import LangDict, tr
from .geom_utils import (angular_range, douglas_peucker, orthonormal_frame, polygon_area_2d,
                         unit, weighted_mean)
from .occ_build import build_shape
from .sweeps import fit_extrusion, fit_revolve

ROLES = {
    "plane": ["surface", "cut"],
    "sphere": ["surface", "add", "sub"],
    "cylinder": ["surface", "add", "sub"],
    "cone": ["surface", "add", "sub"],
    "torus": ["surface", "add", "sub"],
    "ellipsoid": ["surface", "add", "sub"],
    "freeform": ["surface"],
    "extrusion": ["add", "sub", "surface"],
    "revolve": ["add", "sub", "surface"],
    "hole": ["sub"],
}
ROLE_LABELS = LangDict({"surface": ("Fläche", "Surface"), "add": ("Körper +", "Solid +"),
                        "sub": ("Körper −", "Solid −"), "cut": ("Schnitt (Halbraum)", "Cut (half space)")})

POINT_KEYS = ("origin", "point", "apex", "center")
DIR_KEYS = ("normal", "axis", "xdir", "zdir", "dir")

_ID = {"next": 1}


def _new_id(want=0):
    if want:
        _ID["next"] = max(_ID["next"], int(want) + 1)
        return int(want)
    i = _ID["next"]
    _ID["next"] += 1
    return i


def _golden_color(i):
    h = (i * 0.61803398875) % 1.0
    r, g, b = colorsys.hsv_to_rgb(h, 0.55, 0.92)
    return (r, g, b)


def to_jsonable(o):
    if isinstance(o, dict):
        return {k: to_jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [to_jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def _deep_equal(a, b):
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_deep_equal(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(_deep_equal(x, y) for x, y in zip(a, b))
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        try:
            return np.array_equal(np.asarray(a), np.asarray(b))
        except Exception:
            return False
    return a == b


def _arr(params):
    """Listen mit Vektorbedeutung wieder als numpy."""
    p = dict(params)
    for k in POINT_KEYS + DIR_KEYS:
        if k in p and p[k] is not None:
            p[k] = np.asarray(p[k], float)
    for k in ("axes", "radii", "poles", "knots_u", "knots_v"):
        if k in p and p[k] is not None:
            p[k] = np.asarray(p[k], float)
    return p


@dataclass(eq=False)  # Identität statt Feldvergleich (numpy-Arrays!)
class Feature:
    kind: str
    params: dict
    role: str = "surface"
    face_ids: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int64))
    name: str = ""
    stats: dict = field(default_factory=dict)
    visible: bool = True
    auto: bool = False
    id: int = 0
    color: tuple = (0.6, 0.6, 0.6)

    def __post_init__(self):
        self.id = _new_id(self.id)
        if self.color == (0.6, 0.6, 0.6):
            self.color = _golden_color(self.id)
        if not self.name:
            self.name = f"{fit.KIND_LABELS.get(self.kind, self.kind)} {self.id}"
        self.params = _arr(self.params)
        self._shape = None
        self._shape_err = None
        self.version = 0

    # -------------------------------------------------------------- Geometrie
    def invalidate(self):
        self._shape = None
        self._shape_err = None
        self.version += 1

    # -------------------------------------------------------------- Rückgängig/Wiederholen
    def get_state(self):
        return {"kind": self.kind, "params": copy.deepcopy(self.params), "role": self.role,
                "name": self.name, "face_ids": self.face_ids, "visible": self.visible,
                "color": self.color, "stats": dict(self.stats)}

    def set_state(self, st):
        changed = (st["kind"] != self.kind or st["role"] != self.role
                   or not _deep_equal(st["params"], self.params))
        self.kind = st["kind"]
        self.params = copy.deepcopy(st["params"])
        self.role = st["role"]
        self.name = st["name"]
        self.face_ids = st["face_ids"]
        self.visible = st["visible"]
        self.color = st["color"]
        self.stats = dict(st["stats"])
        if changed:
            self.invalidate()
        return changed

    def shape(self):
        if self._shape is None and self._shape_err is None:
            try:
                self._shape = build_shape(self.kind, self.params, self.role)
            except Exception as e:  # noqa: BLE001
                self._shape_err = str(e) or e.__class__.__name__
        if self._shape is None:
            raise RuntimeError(f"{self.name}: {self._shape_err}")
        return self._shape

    def set_role(self, role):
        if role not in ROLES[self.kind]:
            raise ValueError(role)
        self.role = role
        self.invalidate()

    def transform(self, T):
        R = T[:3, :3]
        t = T[:3, 3]
        p = self.params
        for k in POINT_KEYS:
            if k in p and p[k] is not None:
                p[k] = R @ np.asarray(p[k]) + t
        for k in DIR_KEYS:
            if k in p and p[k] is not None:
                p[k] = R @ np.asarray(p[k])
        if "axes" in p and p["axes"] is not None:
            p["axes"] = (R @ np.asarray(p["axes"]).T).T
        if "poles" in p and p["poles"] is not None:
            C = np.asarray(p["poles"])
            p["poles"] = C @ R.T + t
        self.invalidate()

    def summary(self):
        p = self.params
        k = self.kind
        f = lambda v: np.array2string(np.asarray(v), precision=3, suppress_small=True, separator=", ")
        lines = [f"{self.name}  [{fit.KIND_LABELS.get(k, k)}, {ROLE_LABELS.get(self.role, self.role)}]"]
        if p.get("fixed"):
            lines.append(tr("festgelegt (keine Messdaten, nur über Bedingungen veränderbar)",
                            "fixed (no scan data, changed only through constraints)"))
        if p.get("aux"):
            lines.append(tr("Hilfsfläche (automatisch ergänzt)", "auxiliary surface (added automatically)"))
        if k == "plane":
            lines += [tr("Punkt ", "Point ") + f(p['origin']), tr("Normale ", "Normal ") + f(p['normal'])]
        elif k == "sphere":
            lines += [tr("Mitte ", "Center ") + f(p['center']), f"Radius {p['radius']:.4f} mm (Ø {2 * p['radius']:.4f})"]
        elif k == "cylinder":
            lines += [tr("Achse ", "Axis ") + f(p['axis']) + tr(" durch ", " through ") + f(p['point']),
                      f"Radius {p['radius']:.4f} mm (Ø {2 * p['radius']:.4f}), " + tr("Länge", "length")
                      + f" {p['v1'] - p['v0']:.3f}"]
        elif k == "cone":
            lines += [tr("Spitze ", "Apex ") + f(p['apex']) + tr(", Achse ", ", axis ") + f(p['axis']),
                      tr("Halbwinkel", "Half angle") + f" {np.degrees(p['half_angle']):.3f}°, h {p['h0']:.3f}…{p['h1']:.3f}"]
        elif k == "torus":
            lines += [tr("Mitte ", "Center ") + f(p['center']) + tr(", Achse ", ", axis ") + f(p['axis']),
                      f"R {p['R']:.4f} mm, r {p['r']:.4f} mm"]
        elif k == "ellipsoid":
            lines += [tr("Mitte ", "Center ") + f(p['center']), tr("Halbachsen ", "Semi-axes ") + f(p['radii'])]
        elif k == "freeform":
            C = np.asarray(p["poles"])
            lines += [tr("B-Spline ", "B-spline ") + f"{C.shape[0]}×{C.shape[1]} " + tr("Kontrollpunkte, Grad", "control points, degree")
                      + f" {p.get('degree', 3)}",
                      tr("Parametrisierung: ", "Parametrisation: ") + f"{p.get('mode', '?')}"
                      + (tr(" (getrimmt)", " (trimmed)") if p.get("trim_uv") is not None else "")]
        elif k == "extrusion":
            kinds = p.get("profile_kinds") or [l["type"] for l in p["loops"]]
            lines += [tr("Richtung ", "Direction ") + f(p['dir']), tr("Höhe", "Height") + f" {p['z1'] - p['z0']:.3f} mm",
                      tr("Profile: ", "Profiles: ") + ", ".join(kinds)]
            for l in p["loops"]:
                if l["type"] == "circle":
                    lines.append(tr("  Kreis", "  circle") + f" r={l['r']:.4f} (Ø {2 * l['r']:.4f})")
                elif l["type"] == "poly":
                    fil = [r for r in l.get("fillets", []) if r > 0]
                    lines.append(tr("  Polygon ", "  polygon ") + f"{len(l['pts'])} " + tr("Ecken", "corners")
                                 + ((tr(", Radien ", ", radii ") + ", ".join(f"{r:.3f}" for r in fil)) if fil else ""))
        elif k == "revolve":
            lines += [tr("Achse ", "Axis ") + f(p['axis']) + tr(" durch ", " through ") + f(p['point']),
                      tr("Profil: ", "Profile: ") + f"{p['profile']['type']} ({len(p['profile']['pts'])} " + tr("Punkte", "points") + ")"]
            if p.get("port") and p["profile"]["type"] == "poly":
                P = np.asarray(p["profile"]["pts"], float)
                fil = [x for x in p["profile"].get("fillets", []) if x > 0]
                lines.append(tr("Drehkontur an der Fläche: ", "Turned contour at the face: ")
                             + f"Ø {2 * P[:, 0].min():.2f} … {2 * P[:, 0].max():.2f} mm, "
                             + tr("Tiefe", "depth") + f" {P[:, 1].max():.3f} mm"
                             + ((tr(", Radien ", ", radii ") + ", ".join(f"{x:.2f}" for x in fil)) if fil else ""))
                lines.append(tr("Profilpunkte (r, z): ", "Profile points (r, z): ")
                             + "; ".join(f"({a:.3f}, {b:.3f})" for a, b in P))
        elif k == "hole":
            from .bores import hole_summary
            lines += [hole_summary(p), tr("Eintritt ", "Entry ") + f(p['point']) + tr(", Richtung ", ", direction ") + f(p['axis'])]
        st = self.stats
        if st:
            lines.append(f"RMS {st.get('rms', 0):.4f} mm, P95 {st.get('p95', 0):.4f}, max {st.get('max', 0):.4f} "
                         f"({st.get('n', 0)} " + tr("Punkte", "points") + ")")
        return "\n".join(lines)

    # -------------------------------------------------------------- Serialisierung
    def to_dict(self):
        return {"kind": self.kind, "params": to_jsonable(self.params), "role": self.role,
                "name": self.name, "stats": to_jsonable(self.stats), "visible": self.visible,
                "auto": self.auto, "id": self.id, "color": list(self.color)}

    @classmethod
    def from_dict(cls, d, face_ids=None):
        f = cls(kind=d["kind"], params=d["params"], role=d.get("role", "surface"),
                face_ids=np.asarray(face_ids if face_ids is not None else [], np.int64),
                name=d.get("name", ""), stats=d.get("stats", {}), visible=d.get("visible", True),
                auto=d.get("auto", False), id=int(d.get("id", 0)), color=tuple(d.get("color", (0.6, 0.6, 0.6))))
        return f


# ===================================================================== Fit -> Feature
def _stats(res, w=None):
    res = np.asarray(res, float)
    if len(res) == 0:
        return {"rms": 0.0, "p95": 0.0, "max": 0.0, "n": 0}
    if w is None:
        w = np.ones(len(res))
    return {"rms": float(np.sqrt(np.sum(w * res ** 2) / np.sum(w))),
            "p95": float(np.percentile(np.abs(res), 95)),
            "max": float(np.abs(res).max()), "n": int(len(res))}


def _selection_data(model, face_ids, max_pts=80000):
    return model.fit_samples(face_ids, max_pts)


def _patch_vertices(model, face_ids):
    return model.V[np.unique(model.F[face_ids])]


def _plane_ok(p):
    try:
        from .occ_build import build_shape, is_valid
        f = build_shape("plane", p, "surface")
        return is_valid(f)
    except Exception:
        return False


def _smoothed_region(model, face_ids, rings=1):
    """Morphologisches Öffnen der Region (entfernt Zacken/Inseln am Rand)."""
    m = np.zeros(model.n_faces, bool)
    m[face_ids] = True
    closed_ = model.shrink(model.grow(m, rings), rings)          # kleine Löcher schließen
    m2 = model.grow(model.shrink(closed_, rings), rings) & closed_  # Zacken/Inseln entfernen
    ids = np.nonzero(m2)[0]
    return ids if len(ids) > 0.5 * len(face_ids) else np.asarray(face_ids)


def _fitted_outline(model, face_ids, o, x, y, outline_tol, tol):
    """Umriss einer Ebene als Profile (Kreis / Rechteck / Polygon mit Radien / Spline)."""
    from .geom_utils import polyline_length, resample_polyline
    from .profiles import fit_profile, profile_to_dense
    ids = _smoothed_region(model, face_ids, 1)
    comps = model.components_of(ids)
    ids = comps[0] if comps else ids
    out = []
    for loop in model.boundary_loops(ids):
        q = model.V[loop]
        q2 = np.c_[(q - o) @ x, (q - o) @ y]
        L = polyline_length(q2, True)
        a_raw = abs(polygon_area_2d(q2))
        if len(q2) < 6 or L < 10 * tol or 4 * np.pi * a_raw / max(L * L, 1e-12) < 0.03:
            continue  # zu klein oder fadenförmig (Artefakt am Bereichsrand)
        q2 = resample_polyline(q2, max(tol, L / 800), True)
        fit_tol = max(tol, 0.5 * outline_tol)
        # Bereichsränder sind zackig -> stärker zur einfachen Form tendieren
        prof, kind, rms = fit_profile(q2, q2, "auto", fit_tol, "auto", 0.0, 0.5, max(1.0, L / 120),
                                      closed=True, simplicity=2.0, max_poly_corners=12)
        area = abs(polygon_area_2d(profile_to_dense(prof, max(fit_tol, L / 400))[0]))
        if area > 4 * fit_tol * fit_tol:
            out.append((area, prof))
    out.sort(key=lambda t: -t[0])
    if not out:
        return []
    a_outer = out[0][0]
    out = [out[0]] + [t for t in out[1:] if t[0] >= max(25 * tol * tol, 5e-4 * a_outer)]
    return [pr for _, pr in out]


def trim_primitive(kind, params, model, face_ids, opts):
    """Ergänzt Parameter um Parameterbereiche/Umriss passend zur Auswahl."""
    p = dict(params)
    Pv = _patch_vertices(model, face_ids)
    C = model.centroids[face_ids]
    A = model.areas[face_ids]
    cen = weighted_mean(C, A)
    if kind == "plane":
        o, n = p["origin"], p["normal"]
        x, y, _ = orthonormal_frame(n, opts.get("plane_xdir"))
        o = cen - ((cen - o) @ n) * n  # Ursprung = Auswahlschwerpunkt in der Ebene
        p.update(origin=o, xdir=x)
        mode = opts.get("outline", "fit")
        tol = float(opts.get("outline_tol", 0.2))
        loops2 = []
        if mode == "fit":
            try:
                prof_loops = _fitted_outline(model, face_ids, o, x, y, tol, float(opts.get("tol", 0.05)))
            except Exception:
                prof_loops = []
            if prof_loops:
                p["loops"] = prof_loops
                if _plane_ok(p):
                    return p
                # z. B. Risse im Netz erzeugen Schlitze im Rand -> Profil-Fit ungültig
                q = dict(p)
                q["loops"] = prof_loops[:1]
                if _plane_ok(q):
                    return q
            mode = "boundary"
        if mode == "boundary":
            for loop in model.boundary_loops(face_ids):
                q = model.V[loop]
                q2 = np.c_[(q - o) @ x, (q - o) @ y]
                if len(q2) < 3:
                    continue
                idx = douglas_peucker(q2, tol, closed=True)
                q2 = q2[idx]
                if len(q2) >= 3 and abs(polygon_area_2d(q2)) > (tol * tol):
                    loops2.append(q2)
        Q = np.c_[(Pv - o) @ x, (Pv - o) @ y]
        if mode == "rect" or (mode == "boundary" and not loops2):
            from .geom_utils import min_area_rect
            mid, ang, w, h = min_area_rect(Q)
            c, s = np.cos(ang), np.sin(ang)
            R = np.array([[c, -s], [s, c]])
            loc = np.array([[-w / 2, -h / 2], [w / 2, -h / 2], [w / 2, h / 2], [-w / 2, h / 2]])
            loops2 = [loc @ R.T + mid]
        elif mode == "hull":
            from scipy.spatial import ConvexHull
            loops2 = [Q[ConvexHull(Q).vertices]]
        loops2.sort(key=lambda l: -abs(polygon_area_2d(l)))
        p["loops"] = [l.tolist() for l in loops2]
        if mode == "boundary" and not _plane_ok(p):
            from scipy.spatial import ConvexHull
            p["loops"] = [Q[ConvexHull(Q).vertices].tolist()]
        return p
    if kind in ("cylinder", "cone", "torus"):
        a = p["axis"]
        base = p["point"] if kind == "cylinder" else (p["apex"] if kind == "cone" else p["center"])
        r0 = cen - base
        x, y, _ = orthonormal_frame(a, r0 - (r0 @ a) * a)
        rel = Pv - base
        h = rel @ a
        rad = rel - np.outer(h, a)
        ang = np.arctan2(rad @ y, rad @ x)
        u0, u1 = angular_range(ang, opts.get("full_gap_deg", 15.0))
        p.update(xdir=x, u0=u0, u1=u1)
        if kind == "cylinder":
            p.update(v0=float(h.min()), v1=float(h.max()))
        elif kind == "cone":
            p.update(h0=float(max(h.min(), 0.0)), h1=float(h.max()))
        else:
            rho = np.linalg.norm(rad, axis=1)
            v = np.arctan2(h, rho - p["R"])
            v0, v1 = angular_range(v, 10.0)
            p.update(v0=v0, v1=v1)
        return p
    if kind == "sphere":
        c = p["center"]
        m = unit(cen - c)
        z, xx, _ = orthonormal_frame(m)
        x = m
        y = np.cross(z, x)
        rel = Pv - c
        u = np.arctan2(rel @ y, rel @ x)
        v = np.arcsin(np.clip((rel @ z) / np.maximum(np.linalg.norm(rel, axis=1), 1e-12), -1, 1))
        u0, u1 = angular_range(u, 15.0)
        p.update(zdir=z, xdir=x, u0=u0, u1=u1, v0=float(v.min()), v1=float(v.max()))
        if u1 - u0 >= 2 * np.pi - 1e-6:
            p.update(v0=-np.pi / 2 if v.min() < -1.2 else float(v.min()),
                     v1=np.pi / 2 if v.max() > 1.2 else float(v.max()))
        return p
    return p


def snap_extent_to_planes(o, d, z0, z1, planes, max_gap=None, ang_deg=3.0):
    """Verlängert/kürzt einen Bereich [z0, z1] entlang d auf parallele Ebenen-Features."""
    if not planes:
        return z0, z1, []
    h = z1 - z0
    max_gap = max_gap if max_gap is not None else max(2.0, 0.3 * h)
    zs = []
    for po, pn in planes:
        pn = unit(pn)
        if abs(pn @ d) >= np.cos(np.radians(ang_deg)):
            zs.append(float((np.asarray(po) - o) @ d))
    used = []
    if zs:
        zs = np.array(zs)
        k0 = int(np.argmin(np.abs(zs - z0)))
        if abs(zs[k0] - z0) < max_gap:
            z0 = float(zs[k0])
            used.append("Anfang")
        k1 = int(np.argmin(np.abs(zs - z1)))
        if abs(zs[k1] - z1) < max_gap and k1 != k0:
            z1 = float(zs[k1])
            used.append("Ende")
    return z0, z1, used


def fit_feature(model, face_ids, kind, opts):
    """Fit eines Features an die ausgewählten Flächen. Gibt (Feature, Zusatzinfo) zurück."""
    face_ids = np.asarray(face_ids, np.int64)
    if len(face_ids) < 3:
        raise RuntimeError("Auswahl zu klein.")
    tol = float(opts.get("tol", 0.05))
    robust = tol if opts.get("robust", True) else None
    info = {}
    if kind in ("auto", "plane", "sphere", "cylinder", "cone", "torus", "ellipsoid"):
        C, N, A = _selection_data(model, face_ids)
        if kind == "auto":
            fr, allr = fit.auto_fit(C, N, A, tol, kinds=fit.PRIMITIVE_ORDER + ["ellipsoid"], robust=robust)
            info["alternatives"] = {k: r.rms for k, r in allr.items()}
        else:
            fr = fit.FIT_FUNCS[kind](C, N, A, robust)
        params = trim_primitive(fr.kind, fr.params, model, face_ids, opts)
        if fr.kind == "cylinder" and opts.get("planes"):
            v0, v1, used = snap_extent_to_planes(params["point"], params["axis"], params["v0"], params["v1"],
                                                 opts["planes"])
            if used:
                params["v0"], params["v1"] = v0, v1
                info["note"] = "Länge an Ebenen angepasst (" + ", ".join(used) + ")"
        role = "surface"
        feat = Feature(fr.kind, params, role, face_ids, stats=_stats(fr.residuals, fr.weights))
        return feat, info
    if kind == "freeform":
        comps = model.components_of(_smoothed_region(model, face_ids, 1))
        ids = comps[0]
        if len(comps) > 1:
            info["note"] = f"{len(comps)} Teilstücke – nur das größte ({len(ids)} Dreiecke) verwendet."
        V, F, _ = model.submesh(ids)
        r = fit_freeform_patch(V, F, int(opts.get("ff_ctrl", 12)), float(opts.get("ff_smooth", 0.3)),
                               opts.get("ff_mode", "auto"), trim_tol=tol)
        params = {"poles": r["poles"], "knots_u": r["knots_u"], "knots_v": r["knots_v"],
                  "degree": 3, "mode": r["mode"],
                  "trim_uv": None if r["trim_uv"] is None else np.asarray(r["trim_uv"]).tolist()}
        feat = Feature("freeform", params, "surface", ids, stats=_stats(r["residuals"], r["weights"]))
        return feat, info
    if kind == "extrusion":
        params, res, side_ids = fit_extrusion(
            model, face_ids, tol, opts.get("ex_profile", "auto"), opts.get("ex_fillets", "auto"),
            float(opts.get("snap", 0.0)), float(opts.get("ex_smooth", 0.5)), float(opts.get("ex_step", 2.0)),
            fixed_dir=opts.get("ex_dir"))
        if opts.get("planes"):
            z0, z1, used = snap_extent_to_planes(params["origin"], params["dir"], params["z0"], params["z1"],
                                                 opts["planes"])
            if used:
                params["z0"], params["z1"] = z0, z1
                info["note"] = "Höhe an Ebenen angepasst (" + ", ".join(used) + ")"
        params["z0"] -= float(opts.get("ex_extend0", 0.0))
        params["z1"] += float(opts.get("ex_extend1", 0.0))
        role = "surface" if params["open"] else ("sub" if params["inside"] else "add")
        feat = Feature("extrusion", params, role, face_ids, stats=_stats(res))
        return feat, info
    if kind == "revolve":
        params, res = fit_revolve(model, face_ids, tol, opts.get("rv_profile", "auto"),
                                  float(opts.get("rv_smooth", 0.4)), float(opts.get("ex_step", 2.0)),
                                  fixed_axis=opts.get("rv_axis"), snap=float(opts.get("snap", 0.0)))
        role = "sub" if params["inside"] else "add"
        feat = Feature("revolve", params, role, face_ids, stats=_stats(res))
        return feat, info
    raise ValueError(kind)


def grow_selection_to_fit(model, feature, face_ids, tol_factor=2.5, tol=0.05, angle_deg=20.0,
                          exclude=None):
    """Erweitert eine Auswahl um alle angrenzenden Flächen, die zum Primitiv passen."""
    k = feature.kind
    if k not in ("plane", "sphere", "cylinder", "cone", "torus", "ellipsoid"):
        return np.asarray(face_ids)
    p = feature.params
    d = np.abs(fit.model_distance(k, p, model.centroids))
    ok = d < tol_factor * tol
    idx = np.nonzero(ok)[0]
    g = fit.model_normals(k, p, model.centroids[idx])
    c = np.abs(np.einsum("ij,ij->i", g, model.fit_normals[idx]))
    ok[idx[c < np.cos(np.radians(angle_deg))]] = False
    if exclude is not None:
        ok &= ~exclude
    ok[face_ids] = True
    cand = np.nonzero(ok)[0]
    return model.component_containing(cand, np.asarray(face_ids))


def snap_feature(model, feature, angle_tol_deg=3.0, dim_step=0.0, robust_tol=None, axes=None):
    """Konstruktionsabsicht: Achsen/Normalen auf Hauptachsen rasten (mit Neu-Fit), Maße runden."""
    axes = np.eye(3) if axes is None else np.asarray(axes)
    k = feature.kind
    p = feature.params
    dir_key = {"plane": "normal", "cylinder": "axis", "cone": "axis", "torus": "axis",
               "extrusion": "dir", "revolve": "axis"}.get(k)
    changed = []
    if dir_key:
        v = unit(p[dir_key])
        dots = axes @ v
        j = int(np.argmax(np.abs(dots)))
        ang = np.degrees(np.arccos(min(1.0, abs(dots[j]))))
        if 1e-9 < ang <= angle_tol_deg:
            newv = axes[j] * np.sign(dots[j])
            ids = feature.face_ids
            if len(ids) >= 3 and k in ("plane", "cylinder", "cone", "torus"):
                C, N, A = _selection_data(model, ids)
                fr = fit.refit_constrained(k, C, N, A, p, fix_axis=newv, robust=robust_tol)
                newp = trim_primitive(k, fr.params, model, ids, {})
                feature.params = _arr(newp)
                feature.stats = _stats(fr.residuals, fr.weights)
            else:
                p[dir_key] = newv
                if "xdir" in p:
                    x, _, _ = orthonormal_frame(newv, p["xdir"])
                    p["xdir"] = x
            changed.append(tr(f"Richtung um {ang:.2f}° eingerastet", f"direction snapped by {ang:.2f}°"))
    if dim_step and dim_step > 0:
        p = feature.params

        def rnd(v):
            return float(np.round(v / dim_step) * dim_step)
        for key in ("radius", "R", "r"):
            if key in p:
                old = p[key]
                p[key] = max(rnd(old), dim_step)
                changed.append(f"{key}: {old:.4f} → {p[key]:.4f}")
        if k == "extrusion":
            for l in p["loops"]:
                if l["type"] == "circle":
                    l["r"] = max(rnd(l["r"]), dim_step)
                if l["type"] == "poly":
                    l["fillets"] = [rnd(r) if r > 0 else 0.0 for r in l.get("fillets", [])]
            changed.append(tr("Profilmaße gerundet", "profile dimensions rounded"))
    feature.invalidate()
    return changed


def fixed_plane_feature(model, face_ids, plane, name=None, tol=0.05):
    """Ebene OHNE Messdaten (Boden einer offenen Basis, 'als Ebene erzwungene' Füllung):
    wird nie aus dem Scan neu gefittet, nur über Bedingungen (z. B. Abstand) verändert."""
    o, n = np.asarray(plane[0], float), unit(plane[1])
    params = {"origin": o, "normal": n}
    ids = np.asarray(face_ids, np.int64)
    if len(ids) >= 3:
        try:
            params = trim_primitive("plane", params, model, ids, {"outline_tol": max(0.05, 2 * tol)})
        except Exception:
            pass
    params["normal"] = n          # Ursprung kann (in der Ebene) zum Flächenschwerpunkt gewandert sein
    params["fixed"] = True
    return Feature("plane", params, "surface", ids, name=name or tr("Boden", "Floor"))
