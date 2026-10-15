"""Fits analytischer Primitive an orientierte Punktwolken (Punkte + Normalen + Gewichte).

Konventionen (alle Längen in mm, Winkel in rad):
  plane     origin, normal
  sphere    center, radius
  cylinder  point (auf Achse), axis, radius
  cone      apex, axis (von Spitze zur Öffnung), half_angle
  torus     center, axis, R (Ringradius), r (Rohrradius)
  ellipsoid center, axes (3x3, Zeilen = Halbachsenrichtungen), radii (3)
Jeder Fit setzt zusätzlich 'inside' (True = Hohlform, Netznormalen zeigen
zur Primitiv-Innenseite, z. B. Bohrung).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from .i18n import LangDict, tr
from .geom_utils import (fit_circle_2d, orthonormal_frame, pca_frame, perturb_dir,
                         unit, weighted_cov, weighted_mean)

# Reihenfolge = Vorrang bei gleich guter Passung. Zylinder vor Kugel: ein kurzes Zylinderband
# (Bohrung in dünner Platte) passt fast genauso gut auf eine Kugel – Zylinder sind in
# technischen Teilen aber viel häufiger.
PRIMITIVE_ORDER = ["plane", "cylinder", "sphere", "cone", "torus"]
KIND_LABELS = LangDict({
    "plane": ("Ebene", "Plane"),
    "sphere": ("Kugel", "Sphere"),
    "cylinder": ("Zylinder", "Cylinder"),
    "cone": ("Kegel", "Cone"),
    "torus": ("Torus/Rundung", "Torus/fillet"),
    "ellipsoid": ("Ellipsoid", "Ellipsoid"),
    "freeform": ("Freiform", "Freeform"),
    "extrusion": ("Extrusion", "Extrusion"),
    "revolve": ("Rotation", "Revolve"),
    "hole": ("Bohrung", "Hole"),
})


@dataclass
class FitResult:
    kind: str
    params: dict
    residuals: np.ndarray
    weights: np.ndarray
    info: dict = field(default_factory=dict)

    @property
    def rms(self):
        w = self.weights
        return float(np.sqrt((w * self.residuals ** 2).sum() / max(w.sum(), 1e-300)))

    @property
    def max_abs(self):
        return float(np.abs(self.residuals).max()) if len(self.residuals) else 0.0

    def p95(self):
        return float(np.percentile(np.abs(self.residuals), 95)) if len(self.residuals) else 0.0


def _prep(P, N, w):
    P = np.asarray(P, float)
    N = None if N is None else unit(np.asarray(N, float))
    w = np.ones(len(P)) if w is None else np.asarray(w, float)
    w = w / w.mean()
    return P, N, w


def _ls(fun, x0, robust):
    kw = dict(x_scale="jac", max_nfev=200)
    if robust and robust > 0:
        kw.update(loss="soft_l1", f_scale=robust)
    return least_squares(fun, x0, **kw).x


# ============================================================ Modellfunktionen
def model_distance(kind, p, P):
    P = np.asarray(P, float)
    if kind == "plane":
        return (P - p["origin"]) @ p["normal"]
    if kind == "sphere":
        return np.linalg.norm(P - p["center"], axis=1) - p["radius"]
    if kind == "cylinder":
        a = p["axis"]
        d = P - p["point"]
        d = d - np.outer(d @ a, a)
        return np.linalg.norm(d, axis=1) - p["radius"]
    if kind == "cone":
        a = p["axis"]
        q = P - p["apex"]
        h = q @ a
        rho = np.linalg.norm(q - np.outer(h, a), axis=1)
        al = p["half_angle"]
        return rho * np.cos(al) - h * np.sin(al)
    if kind == "torus":
        a = p["axis"]
        q = P - p["center"]
        h = q @ a
        rho = np.linalg.norm(q - np.outer(h, a), axis=1)
        return np.sqrt((rho - p["R"]) ** 2 + h ** 2) - p["r"]
    if kind == "ellipsoid":
        Rm = np.asarray(p["axes"])
        y = (P - p["center"]) @ Rm.T
        t = np.linalg.norm(y / p["radii"], axis=1)
        ny = np.linalg.norm(y, axis=1)
        return (t - 1.0) * ny / np.maximum(t, 1e-12)
    raise ValueError(kind)


def model_normals(kind, p, P):
    P = np.asarray(P, float)
    if kind == "plane":
        return np.tile(p["normal"], (len(P), 1))
    if kind == "sphere":
        return unit(P - p["center"])
    if kind in ("cylinder", "cone", "torus"):
        a = p["axis"]
        base = p.get("point", p.get("apex", p.get("center")))
        q = P - base
        h = q @ a
        radial = q - np.outer(h, a)
        rho = np.linalg.norm(radial, axis=1)
        radial = radial / np.maximum(rho, 1e-12)[:, None]
        if kind == "cylinder":
            return radial
        if kind == "cone":
            al = p["half_angle"]
            return unit(np.cos(al) * radial - np.sin(al) * a[None, :])
        hh = (P - p["center"]) @ a
        return unit((rho - p["R"])[:, None] * radial + hh[:, None] * a[None, :])
    if kind == "ellipsoid":
        Rm = np.asarray(p["axes"])
        y = (P - p["center"]) @ Rm.T
        g = y / np.asarray(p["radii"]) ** 2
        return unit(g @ Rm)
    raise ValueError(kind)


def _finish(kind, params, P, N, w, info=None):
    res = model_distance(kind, params, P)
    info = dict(info or {})
    if N is not None:
        try:
            g = model_normals(kind, params, P)
            dots = np.einsum("ij,ij->i", g, N)
            if kind != "plane":
                params["inside"] = bool(np.sum(w * dots) < 0)
            # mittlere Normalenabweichung (Grad) – unterscheidet z. B. kurzes Zylinderband von Kugel
            ww = np.asarray(w, float)
            ang = np.degrees(np.arccos(np.clip(np.abs(dots), 0, 1)))
            info["ang_rms"] = float(np.sqrt((ww * ang ** 2).sum() / max(ww.sum(), 1e-300)))
        except Exception:
            pass
    if kind == "plane" or N is None:
        params.setdefault("inside", False)
    return FitResult(kind, params, res, w, info)


# ============================================================ Ebene
def fit_plane(P, N=None, w=None, fixed_normal=None, robust=None):
    P, N, w = _prep(P, N, w)
    c = weighted_mean(P, w)
    if fixed_normal is None:
        C = weighted_cov(P - c, w, center=False)
        ev, evec = np.linalg.eigh(C)
        n = evec[:, 0]
    else:
        n = unit(fixed_normal)
    if N is not None and np.sum(w[:, None] * N, axis=0) @ n < 0:
        n = -n
    if robust:
        # robuste Nachiteration (IRLS mit Cauchy-Gewichten)
        for _ in range(5):
            d = (P - c) @ n
            ww = w / (1.0 + (d / robust) ** 2)
            c = weighted_mean(P, ww)
            if fixed_normal is None:
                ev, evec = np.linalg.eigh(weighted_cov(P - c, ww, center=False))
                n2 = evec[:, 0]
                n = n2 if n2 @ n > 0 else -n2
    return _finish("plane", {"origin": c, "normal": n}, P, N, w)


# ============================================================ Kugel
def fit_sphere(P, N=None, w=None, fixed_radius=None, robust=None):
    P, N, w = _prep(P, N, w)
    c0 = weighted_mean(P, w)
    X = P - c0
    sw = np.sqrt(w)
    A = np.c_[2 * X, np.ones(len(X))] * sw[:, None]
    b = (X * X).sum(1) * sw
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    c = sol[:3]
    R = np.sqrt(max(sol[3] + c @ c, 1e-12))
    if fixed_radius is None:
        def res(q):
            return sw * (np.linalg.norm(X - q[:3], axis=1) - q[3])
        q = _ls(res, np.r_[c, R], robust)
        c, R = q[:3], abs(q[3])
    else:
        R = float(fixed_radius)

        def res(q):
            return sw * (np.linalg.norm(X - q[:3], axis=1) - R)
        c = _ls(res, c, robust)
    return _finish("sphere", {"center": c + c0, "radius": float(R)}, P, N, w)


# ============================================================ Zylinder
def cylinder_axis_from_normals(N, w):
    M = weighted_cov(N, w, center=False)
    ev, evec = np.linalg.eigh(M)
    return evec[:, 0], ev


def fit_cylinder(P, N, w=None, fixed_axis=None, fixed_radius=None, robust=None):
    P, N, w = _prep(P, N, w)
    if fixed_axis is None:
        a0, _ = cylinder_axis_from_normals(N, w)
    else:
        a0 = unit(fixed_axis)
    x, y, _ = orthonormal_frame(a0)
    c0 = weighted_mean(P, w)
    Q = np.c_[(P - c0) @ x, (P - c0) @ y]
    cc, R, _ = fit_circle_2d(Q, w, robust)
    p0 = c0 + cc[0] * x + cc[1] * y
    sw = np.sqrt(w)
    if fixed_radius is not None:
        R = float(fixed_radius)

    def axis_of(q):
        return a0 if fixed_axis is not None else perturb_dir(a0, x, y, q[0], q[1])

    def res(q):
        a = axis_of(q)
        pnt = p0 + q[2] * x + q[3] * y
        d = P - pnt
        d = d - np.outer(d @ a, a)
        rad = R if fixed_radius is not None else q[4]
        return sw * (np.linalg.norm(d, axis=1) - rad)

    q0 = [0, 0, 0, 0] + ([] if fixed_radius is not None else [R])
    q = _ls(res, q0, robust)
    a = axis_of(q)
    pnt = p0 + q[2] * x + q[3] * y
    if fixed_radius is None:
        R = abs(q[4])
    # Achspunkt auf Höhe des Schwerpunkts
    pnt = pnt + ((c0 - pnt) @ a) * a
    return _finish("cylinder", {"point": pnt, "axis": unit(a), "radius": float(R)}, P, N, w)


# ============================================================ Kegel
def fit_cone(P, N, w=None, fixed_axis=None, robust=None):
    P, N, w = _prep(P, N, w)
    if fixed_axis is None:
        C = weighted_cov(N, w, center=True)
        ev, evec = np.linalg.eigh(C)
        a0 = evec[:, 0]
    else:
        a0 = unit(fixed_axis)
    # Spitze: alle Tangentialebenen gehen durch sie hindurch
    sw = np.sqrt(w)
    A = N * sw[:, None]
    b = np.einsum("ij,ij->i", N, P) * sw
    apex, *_ = np.linalg.lstsq(A, b, rcond=None)
    if np.sum(w * ((P - apex) @ a0)) < 0:
        a0 = -a0
    q = P - apex
    h = q @ a0
    rho = np.linalg.norm(q - np.outer(h, a0), axis=1)
    al0 = float(np.clip(np.average(np.arctan2(rho, np.maximum(h, 1e-9)), weights=w), 1e-3, np.pi / 2 - 1e-3))
    x, y, _ = orthonormal_frame(a0)

    def unpack(q):
        a = a0 if fixed_axis is not None else perturb_dir(a0, x, y, q[3], q[4])
        return apex + q[:3], a, q[5]

    def res(q):
        ap, a, al = unpack(q)
        qq = P - ap
        hh = qq @ a
        r = np.linalg.norm(qq - np.outer(hh, a), axis=1)
        return sw * (r * np.cos(al) - hh * np.sin(al))

    qs = _ls(res, [0, 0, 0, 0, 0, al0], robust)
    ap, a, al = unpack(qs)
    if al < 0:
        al = -al
    return _finish("cone", {"apex": ap, "axis": unit(a), "half_angle": float(al)}, P, N, w)


# ============================================================ Torus
def fit_torus(P, N, w=None, fixed_axis=None, robust=None):
    P, N, w = _prep(P, N, w)
    rng = np.random.default_rng(0)
    idx = rng.choice(len(P), size=min(len(P), 2500), replace=False)
    Ps, Ns, ws = P[idx], N[idx], w[idx]
    L = float(np.linalg.norm(P.max(0) - P.min(0))) + 1e-9
    best = None
    for s in (1.0, -1.0):
        for r in np.geomspace(0.005 * L, 2.0 * L, 48):
            Cc = Ps - s * r * Ns
            if fixed_axis is None:
                c0, ev, evec = pca_frame(Cc, ws)
                a = evec[:, 2]
            else:
                a = unit(fixed_axis)
                c0 = weighted_mean(Cc, ws)
            x, y, _ = orthonormal_frame(a)
            Q = np.c_[(Cc - c0) @ x, (Cc - c0) @ y]
            sw = np.sqrt(ws)
            Aa = np.c_[2 * Q, np.ones(len(Q))] * sw[:, None]
            bb = (Q * Q).sum(1) * sw
            sol, *_ = np.linalg.lstsq(Aa, bb, rcond=None)
            R = np.sqrt(max(sol[2] + sol[0] ** 2 + sol[1] ** 2, 1e-12))
            d2 = np.hypot(Q[:, 0] - sol[0], Q[:, 1] - sol[1]) - R
            dz = (Cc - c0) @ a
            score = np.sqrt(np.average(d2 ** 2 + dz ** 2, weights=ws))
            if best is None or score < best[0]:
                center = c0 + sol[0] * x + sol[1] * y
                best = (score, center, a, R, r, s)
    _, center0, a0, R0, r0, s0 = best
    x, y, _ = orthonormal_frame(a0)
    sw = np.sqrt(w)

    def unpack(q):
        a = a0 if fixed_axis is not None else perturb_dir(a0, x, y, q[3], q[4])
        return center0 + q[:3], a, q[5], q[6]

    def res(q):
        c, a, R, r = unpack(q)
        qq = P - c
        h = qq @ a
        rho = np.linalg.norm(qq - np.outer(h, a), axis=1)
        return sw * (np.sqrt((rho - R) ** 2 + h ** 2) - abs(r))

    qs = _ls(res, [0, 0, 0, 0, 0, R0, r0], robust)
    c, a, R, r = unpack(qs)
    return _finish("torus", {"center": c, "axis": unit(a), "R": float(abs(R)), "r": float(abs(r))}, P, N, w)


# ============================================================ Ellipsoid
def fit_ellipsoid(P, N=None, w=None, robust=None):
    P, N, w = _prep(P, N, w)
    c0 = weighted_mean(P, w)
    s = np.sqrt(np.average(((P - c0) ** 2).sum(1), weights=w))
    X = (P - c0) / s
    x, y, z = X.T
    D = np.c_[x * x, y * y, z * z, 2 * x * y, 2 * x * z, 2 * y * z, 2 * x, 2 * y, 2 * z]
    sw = np.sqrt(w)
    v, *_ = np.linalg.lstsq(D * sw[:, None], sw, rcond=None)
    A = np.array([[v[0], v[3], v[4]], [v[3], v[1], v[5]], [v[4], v[5], v[2]]])
    g = v[6:9]
    ok = False
    try:
        c = -np.linalg.solve(A, g)
        k = 1.0 + c @ A @ c
        M = A / k
        ev, evec = np.linalg.eigh(M)
        if np.all(ev > 0):
            radii = 1.0 / np.sqrt(ev)
            Rm = evec.T
            ok = True
    except np.linalg.LinAlgError:
        pass
    if not ok:
        sph = fit_sphere(X, None, w)
        c = sph.params["center"]
        radii = np.full(3, sph.params["radius"])
        Rm = np.eye(3)
    if np.linalg.det(Rm) < 0:
        Rm[2] = -Rm[2]
    rv0 = Rotation.from_matrix(Rm).as_rotvec()

    def res(q):
        Rm_ = Rotation.from_rotvec(q[3:6]).as_matrix()
        rr = np.exp(q[6:9])
        yy = (X - q[:3]) @ Rm_.T
        t = np.linalg.norm(yy / rr, axis=1)
        ny = np.linalg.norm(yy, axis=1)
        return sw * (t - 1.0) * ny / np.maximum(t, 1e-12)

    q = _ls(res, np.r_[c, rv0, np.log(radii)], None if not robust else robust / s)
    Rm = Rotation.from_rotvec(q[3:6]).as_matrix()
    params = {"center": q[:3] * s + c0, "axes": Rm, "radii": np.exp(q[6:9]) * s}
    return _finish("ellipsoid", params, P, N, w)


# ============================================================ Rotationsachse
def fit_rotation_axis(P, N, w=None):
    """Achse einer Rotationsfläche aus Normalenlinien (Pottmann/Randrup, lineare Komplexe).

    Alle Flächennormalen einer Rotationsfläche schneiden die Achse.
    Rückgabe: (Achspunkt, Achsrichtung, Steigung 'pitch', Gütemaß).
    """
    P, N, w = _prep(P, N, w)
    c0 = weighted_mean(P, w)
    Pc = P - c0
    L = np.sqrt(np.average((Pc ** 2).sum(1), weights=w)) + 1e-12
    Pc = Pc / L
    Mo = np.cross(Pc, N)
    X = np.c_[Mo, N]
    W = w / w.sum()
    M = (X * W[:, None]).T @ X
    A = M[:3, :3]
    B = M[:3, 3:]
    C = M[3:, 3:]
    # Pseudoinverse mit Abschneiden: fast singuläres C (z. B. Zylinder, alle n ⟂ Achse)
    # betrifft nur die Steigungskomponente (pitch) und darf das Ergebnis nicht dominieren.
    Ci = np.linalg.pinv(C, rcond=1e-2)
    S = A - B @ Ci @ B.T
    ev, evec = np.linalg.eigh(S)
    cvec = evec[:, 0]
    cbar = -Ci @ B.T @ cvec
    pitch = float(cvec @ cbar)
    abar = cbar - pitch * cvec
    q = np.cross(cvec, abar)
    inits = [(unit(cvec), q)]
    # weitere Startwerte: Zylinderachse aus Normalen, PCA-Achsen
    a_cyl, _ = cylinder_axis_from_normals(N, w)
    _, _, evp = pca_frame(Pc, w)
    for a0 in (a_cyl, evp[:, 0], evp[:, 1], evp[:, 2]):
        inits.append((unit(a0), None))
    sw = np.sqrt(w)

    def axis_point_for(a):
        G = np.cross(N, a)
        rhs = np.einsum("ij,ij->i", G, Pc)
        qq, *_ = np.linalg.lstsq(G * sw[:, None], rhs * sw, rcond=None)
        return qq - (qq @ a) * a

    best = None
    for a0, q0 in inits:
        if q0 is None:
            q0 = axis_point_for(a0)
        x, y, _ = orthonormal_frame(a0)

        def res(t, a0=a0, q0=q0, x=x, y=y):
            a = perturb_dir(a0, x, y, t[0], t[1])
            qq = q0 + t[2] * x + t[3] * y
            return sw * np.einsum("ij,ij->i", Pc - qq, np.cross(N, a))
        try:
            t = least_squares(res, np.zeros(4), x_scale="jac", max_nfev=200).x
        except Exception:
            continue
        cost = float(np.sum(res(t) ** 2))
        if best is None or cost < best[0]:
            a = perturb_dir(a0, x, y, t[0], t[1])
            best = (cost, a, q0 + t[2] * x + t[3] * y)
    _, a, q = best
    point = c0 + q * L
    a = unit(a)
    if a[np.argmax(np.abs(a))] < 0:
        a = -a
    return point, a, pitch * L, float(best[0] / max(np.sum(w), 1e-12))


# ============================================================ Auto-Auswahl
FIT_FUNCS = {
    "plane": lambda P, N, w, r: fit_plane(P, N, w, robust=r),
    "sphere": lambda P, N, w, r: fit_sphere(P, N, w, robust=r),
    "cylinder": lambda P, N, w, r: fit_cylinder(P, N, w, robust=r),
    "cone": lambda P, N, w, r: fit_cone(P, N, w, robust=r),
    "torus": lambda P, N, w, r: fit_torus(P, N, w, robust=r),
    "ellipsoid": lambda P, N, w, r: fit_ellipsoid(P, N, w, robust=r),
}


def is_plausible(fr: FitResult, P):
    L = float(np.linalg.norm(P.max(0) - P.min(0))) + 1e-9
    p = fr.params
    k = fr.kind
    if not np.all(np.isfinite(fr.residuals)):
        return False
    if k == "sphere":
        return p["radius"] < 25 * L
    if k == "cylinder":
        return p["radius"] < 25 * L
    if k == "cone":
        al = np.degrees(p["half_angle"])
        dist = np.linalg.norm(weighted_mean(P) - p["apex"])
        return 1.0 < al < 88.0 and dist < 30 * L
    if k == "torus":
        return p["R"] < 25 * L and p["r"] < 5 * L and p["r"] > 1e-4 * L
    if k == "ellipsoid":
        return np.max(p["radii"]) < 25 * L
    return True


def auto_fit(P, N, w=None, tol=0.05, kinds=None, robust=None):
    """Modellauswahl: das einfachste Primitiv, dessen RMS höchstens
    max(1.3*bestes RMS, bestes RMS + tol/4) beträgt (Reihenfolge Ebene < Kugel <
    Zylinder < Kegel < Torus). So gewinnt z. B. eine Ebene gegen einen riesigen
    Zylinder, aber eine Kugel nicht gegen einen klar besser passenden Kegel."""
    kinds = kinds or PRIMITIVE_ORDER
    results = {}
    for k in kinds:
        try:
            fr = FIT_FUNCS[k](P, N, w, robust)
            if is_plausible(fr, np.asarray(P)):
                results[k] = fr
        except Exception:
            continue
    if not results:
        raise RuntimeError(tr("Kein Primitiv-Fit möglich.", "No primitive fit possible."))
    best = min(r.rms for r in results.values())
    best_ang = min(r.info.get("ang_rms", 0.0) for r in results.values())
    # einfachstes Modell, das nicht deutlich schlechter ist als das beste – im Abstand UND in
    # der Normalenrichtung (eine Kugel durch ein kurzes Zylinderband passt im Abstand, aber ihre
    # Normalen kippen zum Rand hin)
    thr = max(1.3 * best, best + 0.25 * tol)
    thr_ang = max(1.5 * best_ang, best_ang + 2.0)
    for k in kinds:
        if k in results and results[k].rms <= thr and results[k].info.get("ang_rms", 0.0) <= thr_ang:
            return results[k], results
    k = min(results, key=lambda kk: results[kk].rms)
    return results[k], results


def refit_constrained(kind, P, N, w, params, fix_axis=None, fix_radius=None, robust=None):
    """Neu-Fit mit festgehaltener Achse/Normalen bzw. festem Radius."""
    if kind == "plane":
        return fit_plane(P, N, w, fixed_normal=fix_axis, robust=robust)
    if kind == "cylinder":
        return fit_cylinder(P, N, w, fixed_axis=fix_axis, fixed_radius=fix_radius, robust=robust)
    if kind == "sphere":
        return fit_sphere(P, N, w, fixed_radius=fix_radius, robust=robust)
    if kind == "cone":
        return fit_cone(P, N, w, fixed_axis=fix_axis, robust=robust)
    if kind == "torus":
        return fit_torus(P, N, w, fixed_axis=fix_axis, robust=robust)
    raise ValueError(kind)
