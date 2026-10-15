"""Bedingungen (Constraints) am fertigen Modell – rückwirkend über den Scan erzwungen.

Statt Parameter nachträglich zu verbiegen, werden alle betroffenen Features GEMEINSAM neu
gefittet (Benkő et al. 2002, 'Constrained fitting in reverse engineering'):

    min  Σ_Features Σ_Punkte w·d(Punkt, Fläche)²/tol²  +  W·Σ_Bedingungen r_c(x)²  +  ε·Σ (x − x0)²

  * d   Abstand der Scanpunkte zur jeweiligen Fläche (nur echte Messdaten, keine Füllungen)
  * r_c Bedingungs-Residuen (Winkel in Bogenmaß × Bauteilgröße, Abstände in mm), sehr stark gewichtet
  * ε   schwache Bindung an den Ausgangszustand -> Elemente OHNE Messdaten (Symmetrieachsen,
        festgelegte Ebenen wie ein extrapolierter Boden, Bohrungen) werden allein durch die
        Bedingungen bestimmt.

Referenzen (Strings):
  'f:<id>'          Feature (Ebene -> Ebene; Zylinder/Kegel/Torus/Bohrung -> Achse; Kugel -> Punkt)
  'mid:<id>:<id>'   Mittelebene zweier Ebenen
  's:<id>'          Symmetrie-Element (Spiegelebene / Drehachse)
  'w:X|Y|Z'         Weltachse durch den Ursprung,  'w:XY|YZ|XZ' Weltebene
"""
from __future__ import annotations

import math

import numpy as np

from .geom_utils import unit
from .i18n import LangDict, tr

TYPES = LangDict({
    "parallel": ("parallel", "parallel"),
    "perpendicular": ("senkrecht", "perpendicular"),
    "angle": ("Winkel", "angle"),
    "coaxial": ("koaxial", "coaxial"),
    "coplanar": ("koplanar", "coplanar"),
    "distance": ("Abstand", "distance"),
    "diameter": ("Ø", "Ø"),
    "radius": ("Radius", "radius"),
    "equal_radius": ("gleiche Radien", "equal radii"),
    "axis_in_plane": ("Achse in Ebene", "axis in plane"),
    "symmetric": ("symmetrisch zu Ebene", "symmetric about plane"),
    "invariant": ("invariant unter Symmetrie", "invariant under symmetry"),
    "fix": ("fixieren", "fix"),
})
ARITY = {"parallel": 2, "perpendicular": 2, "angle": 2, "coaxial": 2, "coplanar": 2, "distance": 2,
         "diameter": 1, "radius": 1, "equal_radius": -2, "axis_in_plane": 2, "symmetric": 3, "invariant": 2,
         "fix": 1}
NEEDS_VALUE = {"angle", "distance", "diameter", "radius"}
SUPPORTED_KINDS = ("plane", "cylinder", "cone", "sphere", "torus", "hole")

_CID = {"next": 1}


def new_constraint(ctype, refs, value=None, existing=()):
    used = {c["id"] for c in existing}
    while _CID["next"] in used:
        _CID["next"] += 1
    c = {"id": _CID["next"], "type": ctype, "refs": list(refs), "value": value, "weight": 1.0}
    _CID["next"] += 1
    return c


def describe(c, names):
    def nm(r):
        return names.get(r, r)
    t = TYPES.get(c["type"], c["type"])
    s = f"{t}: " + ", ".join(nm(r) for r in c["refs"])
    if c.get("value") is not None:
        s += f" = {c['value']:g}" + ("°" if c["type"] == "angle" else " mm")
    return s


# ===================================================================== Variablenblöcke
class _Block:
    """Parameter eines Features/Symmetrie-Elements als Vektor."""

    def __init__(self, key, kind, params, data=None, prior=1e-2):
        self.key, self.kind, self.p0 = key, kind, params
        self.data = data          # (P, w) oder None
        self.prior = prior
        self.x0 = self.pack(params)
        self.n = len(self.x0)

    def pack(self, p):
        k = self.kind
        if k in ("plane", "mirror"):
            n = unit(p["normal"] if k == "plane" else p["nrm"])
            o = np.asarray(p["origin"] if k == "plane" else p["o"], float)
            return np.r_[n, o @ n]
        if k == "cylinder":
            return np.r_[p["point"], unit(p["axis"]), p["radius"]]
        if k == "cone":
            return np.r_[p["apex"], unit(p["axis"]), p["half_angle"]]
        if k == "sphere":
            return np.r_[p["center"], p["radius"]]
        if k == "torus":
            return np.r_[p["center"], unit(p["axis"]), p["R"], p["r"]]
        if k == "hole":
            return np.r_[p["point"], unit(p["axis"]), p["d"]]
        if k == "rot":
            return np.r_[p["p"], unit(p["d"])]
        raise ValueError(k)

    def geom(self, x):
        """Geometrisches Primitiv + Fit-Parameter für model_distance."""
        k = self.kind
        if k in ("plane", "mirror"):
            n = unit(x[:3])
            o = x[3] * n
            return {"type": "plane", "o": o, "n": n}, {"origin": o, "normal": n}
        if k == "cylinder":
            d = unit(x[3:6])
            return ({"type": "axis", "p": x[:3], "d": d, "r": x[6]},
                    {"point": x[:3], "axis": d, "radius": x[6]})
        if k == "cone":
            d = unit(x[3:6])
            return ({"type": "axis", "p": x[:3], "d": d, "alpha": x[6]},
                    {"apex": x[:3], "axis": d, "half_angle": x[6]})
        if k == "sphere":
            return {"type": "point", "p": x[:3], "r": x[3]}, {"center": x[:3], "radius": x[3]}
        if k == "torus":
            d = unit(x[3:6])
            return ({"type": "axis", "p": x[:3], "d": d, "r": x[6], "rt": x[7]},
                    {"center": x[:3], "axis": d, "R": x[6], "r": x[7]})
        if k == "hole":
            d = unit(x[3:6])
            return {"type": "axis", "p": x[:3], "d": d, "r": 0.5 * x[6]}, None
        if k == "rot":
            d = unit(x[3:6])
            return {"type": "axis", "p": x[:3], "d": d}, None
        raise ValueError(k)

    def gauge(self, x, L):
        """Normierung + Lage entlang der Achse festhalten (sonst unbestimmt)."""
        k = self.kind
        out = []
        if k in ("plane", "mirror"):
            out.append(10.0 * (np.linalg.norm(x[:3]) - 1.0))
        if k in ("cylinder", "cone", "torus", "hole", "rot"):
            out.append(10.0 * (np.linalg.norm(x[3:6]) - 1.0))
            if k in ("cylinder", "rot", "hole"):
                d = unit(x[3:6])
                out.append(((x[:3] - self.x0[:3]) @ d) / max(L, 1.0))
        return out


# ===================================================================== Geometrie-Helfer
def _dir(g):
    return g["n"] if g["type"] == "plane" else g.get("d")


def _reflect(g, m):
    """Primitiv g an der Ebene m spiegeln."""
    n, o = m["n"], m["o"]

    def rp(p):
        return p - 2 * ((p - o) @ n) * n

    def rv(v):
        return v - 2 * (v @ n) * n
    if g["type"] == "plane":
        return {"type": "plane", "o": rp(g["o"]), "n": rv(g["n"])}
    if g["type"] == "axis":
        return dict(g, p=rp(g["p"]), d=rv(g["d"]))
    return dict(g, p=rp(g["p"]))


def _axis_offset(p, a, q):
    """Senkrechter Versatz des Punktes q von der Achse (p, a) (3 Komponenten)."""
    w = q - p
    return w - (w @ a) * a


def _world(ref):
    k = ref.split(":", 1)[1].upper()
    if k in ("X", "Y", "Z"):
        return {"type": "axis", "p": np.zeros(3), "d": np.eye(3)["XYZ".index(k)]}
    n = {"XY": [0, 0, 1], "YZ": [1, 0, 0], "XZ": [0, 1, 0]}[k]
    return {"type": "plane", "o": np.zeros(3), "n": np.asarray(n, float)}


# ===================================================================== Residuen
def constraint_residual(c, G, L):
    """G: ref -> Primitiv. Rückgabe Residuen in 'mm' (Winkel × L/2)."""
    t = c["type"]
    refs = c["refs"]
    g = [G[r] for r in refs]
    s = 0.5 * L
    v = c.get("value")
    if t == "parallel":
        return np.cross(_dir(g[0]), _dir(g[1])) * s
    if t == "perpendicular":
        return np.array([_dir(g[0]) @ _dir(g[1])]) * s
    if t == "angle":
        u, w = _dir(g[0]), _dir(g[1])
        th = math.radians(v)
        if {g[0]["type"], g[1]["type"]} == {"plane", "axis"}:
            ang = math.asin(min(1.0, abs(float(u @ w))))       # Winkel Achse–Ebene
        else:
            ang = math.acos(min(1.0, abs(float(u @ w))))
        return np.array([ang - th]) * s
    if t == "coaxial":
        a, b = g
        return np.r_[np.cross(a["d"], b["d"]) * s, _axis_offset(a["p"], a["d"], b["p"])]
    if t == "coplanar":
        a, b = g
        return np.r_[np.cross(a["n"], b["n"]) * s, (b["o"] - a["o"]) @ a["n"]]
    if t == "distance":
        a, b = g
        ta, tb = a["type"], b["type"]
        if ta == "plane" and tb == "plane":
            return np.r_[np.cross(a["n"], b["n"]) * s, abs(float((b["o"] - a["o"]) @ a["n"])) - v]
        if {ta, tb} == {"plane", "axis"}:
            pl, ax = (a, b) if ta == "plane" else (b, a)
            return np.r_[(ax["d"] @ pl["n"]) * s, abs(float((ax["p"] - pl["o"]) @ pl["n"])) - v]
        if {ta, tb} == {"plane", "point"}:
            pl, pt = (a, b) if ta == "plane" else (b, a)
            return np.array([abs(float((pt["p"] - pl["o"]) @ pl["n"])) - v])
        if ta == "axis" and tb == "axis":
            return np.r_[np.cross(a["d"], b["d"]) * s, np.linalg.norm(_axis_offset(a["p"], a["d"], b["p"])) - v]
        if {ta, tb} == {"axis", "point"}:
            ax, pt = (a, b) if ta == "axis" else (b, a)
            return np.array([np.linalg.norm(_axis_offset(ax["p"], ax["d"], pt["p"])) - v])
        return np.array([np.linalg.norm(b["p"] - a["p"]) - v])
    if t == "diameter":
        return np.array([2 * g[0]["r"] - v])
    if t == "radius":
        return np.array([g[0]["r"] - v])
    if t == "equal_radius":
        return np.array([gi["r"] - g[0]["r"] for gi in g[1:]])
    if t == "axis_in_plane":
        ax, pl = (g[0], g[1]) if g[0]["type"] == "axis" else (g[1], g[0])
        return np.r_[(ax["d"] @ pl["n"]) * s, (ax["p"] - pl["o"]) @ pl["n"]]
    if t == "symmetric":
        a, b, m = g
        ar = _reflect(a, m)
        if a["type"] == "plane":
            return np.r_[np.cross(ar["n"], b["n"]) * s, (b["o"] - ar["o"]) @ b["n"]]
        if a["type"] == "axis":
            out = np.r_[np.cross(ar["d"], b["d"]) * s, _axis_offset(b["p"], b["d"], ar["p"])]
            if a.get("r") is not None and b.get("r") is not None:
                out = np.r_[out, a["r"] - b["r"]]
            return out
        return ar["p"] - b["p"]
    if t == "invariant":
        f, sy = g
        if sy["type"] == "plane":           # Spiegelung
            if f["type"] == "axis":
                return np.r_[(f["d"] @ sy["n"]) * s, (f["p"] - sy["o"]) @ sy["n"]]
            if f["type"] == "plane":
                if abs(float(f["n"] @ sy["n"])) > 0.9:
                    return np.r_[np.cross(f["n"], sy["n"]) * s, (f["o"] - sy["o"]) @ sy["n"]]
                return np.array([(f["n"] @ sy["n"]) * s])
            return np.array([(f["p"] - sy["o"]) @ sy["n"]])
        if f["type"] == "axis":             # Drehung
            return np.r_[np.cross(f["d"], sy["d"]) * s, _axis_offset(sy["p"], sy["d"], f["p"])]
        if f["type"] == "plane":
            return np.cross(f["n"], sy["d"]) * s
        return _axis_offset(sy["p"], sy["d"], f["p"])
    if t == "fix":
        return np.zeros(0)
    raise ValueError(t)


# ===================================================================== Löser
def _feature_block(model, f, tol, n_max=400, seed=0):
    k = f.kind
    if k not in SUPPORTED_KINDS:
        raise ValueError(tr(f"{f.name}: Art '{k}' wird von Bedingungen nicht unterstützt",
                            f"{f.name}: kind '{k}' is not supported by constraints"))
    data = None
    prior = 1e-2
    if k == "hole" or f.params.get("fixed"):
        prior = 0.3                      # keine (brauchbaren) Messdaten: nur Bedingungen + schwache Bindung
    else:
        ids = np.asarray(f.face_ids, np.int64)
        ids = ids[(ids >= 0) & (ids < model.n_faces)]
        ids = ids[~model.synthetic[ids]]
        if len(ids) >= 6:
            rng = np.random.default_rng(seed + f.id)
            if len(ids) > n_max:
                ids = rng.choice(ids, n_max, replace=False)
            w = model.areas[ids]
            data = (model.centroids[ids], w / w.mean())
        else:
            prior = 0.3
    return _Block(f"f:{f.id}", k, f.params, data, prior)


def _sym_block(s):
    if s["type"] == "mirror":
        return _Block(f"s:{s['id']}", "mirror", s, None, 0.3)
    return _Block(f"s:{s['id']}", "rot", s, None, 0.3)


def solve(model, features, symmetries, constraints, tol=0.05, weight=1000.0, log=None):
    """Gemeinsamer Neu-Fit. Rückgabe dict(params: {key: neue params}, report, ok)."""
    from scipy.optimize import least_squares
    from .fitting import model_distance
    log = log or (lambda s: None)
    fmap = {f"f:{f.id}": f for f in features}
    smap = {f"s:{s['id']}": s for s in symmetries}
    L = float(np.linalg.norm(model.bbox[1] - model.bbox[0])) if model is not None else 100.0
    # beteiligte Blöcke
    keys = []
    for c in constraints:
        for r in c["refs"]:
            if r.startswith("mid:"):
                _, a, b = r.split(":")
                keys += [f"f:{a}", f"f:{b}"]
            elif r.startswith(("f:", "s:")):
                keys.append(r)
    keys = list(dict.fromkeys(keys))
    blocks = []
    for k in keys:
        if k.startswith("f:"):
            if k not in fmap:
                raise ValueError(tr(f"Feature {k} existiert nicht mehr", f"feature {k} no longer exists"))
            blocks.append(_feature_block(model, fmap[k], tol))
        else:
            if k not in smap:
                raise ValueError(tr(f"Symmetrie {k} existiert nicht mehr", f"symmetry {k} no longer exists"))
            blocks.append(_sym_block(smap[k]))
    offs = np.cumsum([0] + [b.n for b in blocks])
    x0 = np.concatenate([b.x0 for b in blocks]) if blocks else np.zeros(0)
    fixed_keys = {r for c in constraints if c["type"] == "fix" for r in c["refs"]}

    def prims(x):
        G = {}
        for i, b in enumerate(blocks):
            G[b.key] = b.geom(x[offs[i]:offs[i + 1]])[0]
        for c in constraints:
            for r in c["refs"]:
                if r in G:
                    continue
                if r.startswith("w:"):
                    G[r] = _world(r)
                elif r.startswith("mid:"):
                    _, a, bb = r.split(":")
                    pa, pb = G[f"f:{a}"], G[f"f:{bb}"]
                    na, nb = pa["n"], pb["n"]
                    if na @ nb < 0:
                        nb = -nb
                    n = unit(na + nb)
                    o = 0.5 * (pa["o"] + pb["o"])
                    G[r] = {"type": "plane", "o": o, "n": n}
        return G

    def resid(x, w_c=weight):
        out = []
        for i, b in enumerate(blocks):
            xb = x[offs[i]:offs[i + 1]]
            if b.data is not None:
                _, fp = b.geom(xb)
                P, w = b.data
                out.append(np.sqrt(w) * model_distance(b.kind, fp, P) / tol)
            pw = b.prior if b.key not in fixed_keys else 1e3
            dx = xb - b.x0
            out.append(pw * dx * np.where(np.arange(b.n) < 6, 1.0, 1.0) / max(tol, 1e-6) * 0.05)
            out.append(np.asarray(b.gauge(xb, L)) / max(tol, 1e-6))
        G = prims(x)
        for c in constraints:
            if c["type"] == "fix":
                continue
            out.append(w_c * c.get("weight", 1.0) * constraint_residual(c, G, L) / max(tol, 1e-6))
        return np.concatenate(out) if out else np.zeros(1)

    def report_constraints(x):
        G = prims(x)
        rows = []
        for c in constraints:
            if c["type"] == "fix":
                continue
            r = constraint_residual(c, G, L)
            rows.append(float(np.max(np.abs(r))) if len(r) else 0.0)
        return rows

    def data_rms(x):
        out = {}
        for i, b in enumerate(blocks):
            if b.data is None:
                continue
            _, fp = b.geom(x[offs[i]:offs[i + 1]])
            P, w = b.data
            d = model_distance(b.kind, fp, P)
            out[b.key] = float(np.sqrt(np.average(d ** 2, weights=w)))
        return out

    before_c = report_constraints(x0)
    before_d = data_rms(x0)
    if len(x0) == 0:
        return {"params": {}, "report": [tr("Nichts zu lösen.", "Nothing to solve.")], "ok": True}
    sol = least_squares(resid, x0, method="trf", x_scale="jac", max_nfev=400 * max(1, len(x0) // 6))
    x = sol.x
    after_c = report_constraints(x)
    after_d = data_rms(x)
    # Ergebnisparameter
    new = {}
    for i, b in enumerate(blocks):
        xb = x[offs[i]:offs[i + 1]]
        geo, fp = b.geom(xb)
        if b.key.startswith("f:"):
            f = fmap[b.key]
            p = dict(f.params)
            if b.kind == "plane":
                o_old = np.asarray(f.params["origin"], float)
                p["normal"] = geo["n"]
                p["origin"] = o_old - ((o_old - geo["o"]) @ geo["n"]) * geo["n"]
            elif b.kind == "cylinder":
                q = np.asarray(f.params["point"], float)
                p["axis"] = geo["d"]
                p["point"] = geo["p"] + ((q - geo["p"]) @ geo["d"]) * geo["d"]
                p["radius"] = float(geo["r"])
            elif b.kind == "cone":
                p.update(apex=geo["p"], axis=geo["d"], half_angle=float(geo["alpha"]))
            elif b.kind == "sphere":
                p.update(center=geo["p"], radius=float(geo["r"]))
            elif b.kind == "torus":
                p.update(center=geo["p"], axis=geo["d"], R=float(geo["r"]), r=float(geo["rt"]))
            elif b.kind == "hole":
                q = np.asarray(f.params["point"], float)
                p["axis"] = geo["d"]
                p["point"] = geo["p"] + ((q - geo["p"]) @ geo["d"]) * geo["d"]
                p["d"] = float(2 * geo["r"])
            new[b.key] = p
        else:
            s = dict(smap[b.key])
            if b.kind == "mirror":
                s["nrm"], s["o"] = geo["n"], geo["o"] + ((np.asarray(s["o"]) - geo["o"]) @ geo["n"]) * 0
                # Ursprung in der Nähe des alten halten
                o_old = np.asarray(smap[b.key]["o"], float)
                s["o"] = o_old - ((o_old - geo["o"]) @ geo["n"]) * geo["n"]
            else:
                s["d"] = geo["d"]
                s["p"] = geo["p"]
            new[b.key] = s
    rep = []
    viol = max(after_c) if after_c else 0.0
    ok = viol < max(0.01, tol * 0.2)
    rep.append(tr(f"Bedingungen: max. Abweichung vorher {max(before_c) if before_c else 0:.4f}, nachher {viol:.5f} (mm bzw. Winkel×L/2)",
                  f"constraints: max deviation before {max(before_c) if before_c else 0:.4f}, after {viol:.5f} (mm or angle×L/2)"))
    for k in after_d:
        nm = fmap[k].name if k in fmap else k
        rep.append(tr(f"  {nm}: RMS {before_d[k]:.4f} → {after_d[k]:.4f} mm",
                      f"  {nm}: RMS {before_d[k]:.4f} → {after_d[k]:.4f} mm"))
    if not ok:
        rep.append(tr("⚠ Bedingungen nicht vollständig erfüllbar (widersprüchlich oder zu stark gegen die Messdaten)",
                      "⚠ constraints not fully satisfiable (contradictory or too strong against the scan data)"))
    return {"params": new, "report": rep, "ok": ok, "violations": after_c}


def apply_solution(model, features, symmetries, result, tol=0.05):
    """Neue Parameter übernehmen (Features neu trimmen). Ändert Objekte in place."""
    from .features import _arr, trim_primitive
    fmap = {f"f:{f.id}": f for f in features}
    smap = {f"s:{s['id']}": s for s in symmetries}
    for k, p in result["params"].items():
        if k in fmap:
            f = fmap[k]
            if f.kind in ("plane", "cylinder", "cone", "sphere", "torus") and len(f.face_ids) >= 3 and not p.get("fixed"):
                try:
                    p = trim_primitive(f.kind, p, model, np.asarray(f.face_ids), {"outline_tol": max(0.05, 2 * tol)})
                except Exception:
                    pass
            f.params = _arr(p)
            f.params["constrained"] = True
            f.invalidate()
        elif k in smap:
            smap[k].update(p)


# ===================================================================== Vorschläge
def suggest(features, symmetries=(), tol=0.05, ang_tol=1.0, max_n=40):
    """Naheliegende Bedingungen: (Bedingung, Begründung, Abweichung)."""
    from .bores import snap_diameter
    out = []
    pl = [f for f in features if f.kind == "plane" and f.role == "surface"]
    ax = [f for f in features if f.kind in ("cylinder", "cone", "torus", "hole")]
    rad = [f for f in features if f.kind in ("cylinder", "hole", "sphere")]
    ca = math.cos(math.radians(ang_tol))

    def d_of(f):
        p = f.params
        return unit(p["normal"] if f.kind == "plane" else p["axis"])

    def add(t, refs, val, why, dev):
        out.append(({"type": t, "refs": refs, "value": val}, why, dev))

    for i, a in enumerate(pl):
        for b in pl[i + 1:]:
            c = abs(float(d_of(a) @ d_of(b)))
            if c > ca:
                off = abs(float((np.asarray(b.params["origin"]) - a.params["origin"]) @ d_of(a)))
                dev = math.degrees(math.acos(min(1, c)))
                if off < 3 * tol:
                    add("coplanar", [f"f:{a.id}", f"f:{b.id}"], None, tr("fast koplanar", "nearly coplanar"), dev)
                else:
                    add("parallel", [f"f:{a.id}", f"f:{b.id}"], None, tr(f"fast parallel, Abstand {off:.3f}",
                                                                          f"nearly parallel, distance {off:.3f}"), dev)
                    r1 = round(off * 10) / 10
                    if abs(r1 - off) < max(2 * tol, 0.03):
                        add("distance", [f"f:{a.id}", f"f:{b.id}"], r1,
                            tr(f"Abstand {off:.3f} ≈ {r1:g}", f"distance {off:.3f} ≈ {r1:g}"), abs(r1 - off))
            elif c < math.sin(math.radians(ang_tol)):
                add("perpendicular", [f"f:{a.id}", f"f:{b.id}"], None, tr("fast senkrecht", "nearly perpendicular"),
                    math.degrees(math.asin(c)))
    for i, a in enumerate(ax):
        for b in ax[i + 1:]:
            c = abs(float(d_of(a) @ d_of(b)))
            if c > ca:
                pa = np.asarray(a.params.get("point", a.params.get("apex", a.params.get("center"))), float)
                pb = np.asarray(b.params.get("point", b.params.get("apex", b.params.get("center"))), float)
                off = float(np.linalg.norm(_axis_offset(pa, d_of(a), pb)))
                if off < max(4 * tol, 0.1):
                    add("coaxial", [f"f:{a.id}", f"f:{b.id}"], None, tr(f"fast koaxial ({off:.3f})", f"nearly coaxial ({off:.3f})"), off)
                else:
                    add("parallel", [f"f:{a.id}", f"f:{b.id}"], None, tr("Achsen fast parallel", "axes nearly parallel"),
                        math.degrees(math.acos(min(1, c))))
        for p in pl:
            c = abs(float(d_of(a) @ d_of(p)))
            if c > ca:
                add("parallel", [f"f:{a.id}", f"f:{p.id}"], None, tr(f"Achse ⟂ {p.name}", f"axis ⟂ {p.name}"),
                    math.degrees(math.acos(min(1, c))))
    for i, a in enumerate(rad):
        ra = a.params["radius"] if a.kind != "hole" else 0.5 * a.params["d"]
        sn = snap_diameter(2 * ra)
        if sn is not None and abs(sn[0] - 2 * ra) > 1e-6:
            add("diameter", [f"f:{a.id}"], sn[0], f"Ø {2 * ra:.3f} ≈ {sn[0]:g} ({sn[1]})", abs(sn[0] - 2 * ra))
        for b in rad[i + 1:]:
            rb = b.params["radius"] if b.kind != "hole" else 0.5 * b.params["d"]
            if a.kind == b.kind and abs(ra - rb) < max(2 * tol, 0.02 * ra) and abs(ra - rb) > 1e-9:
                add("equal_radius", [f"f:{a.id}", f"f:{b.id}"], None, tr("fast gleiche Radien", "nearly equal radii"), abs(ra - rb))
    for s in symmetries:
        for f in s.get("features") or []:
            add("invariant", [f"f:{f}", f"s:{s['id']}"], None, tr(f"Teil von {s['name']}", f"part of {s['name']}"), 0.0)
    out.sort(key=lambda t: t[2])
    return out[:max_n]


def constraints_to_json(cs):
    return [dict(c) for c in cs]
