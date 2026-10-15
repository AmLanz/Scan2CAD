"""Kleine geometrische Hilfsfunktionen (reines numpy/scipy)."""
from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares

TWO_PI = 2.0 * np.pi


# ---------------------------------------------------------------- Vektoren
def unit(v):
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.maximum(n, 1e-300)


def orthonormal_frame(z, x_hint=None):
    """Rechtshändiges Orthonormalsystem (x, y, z) mit z = unit(z)."""
    z = unit(z)
    if x_hint is not None:
        x = np.asarray(x_hint, float) - np.dot(x_hint, z) * z
        if np.linalg.norm(x) < 1e-9:
            x_hint = None
    if x_hint is None:
        a = np.eye(3)[np.argmin(np.abs(z))]
        x = a - np.dot(a, z) * z
    x = unit(x)
    y = np.cross(z, x)
    return x, y, z


def weighted_mean(P, w=None):
    if w is None:
        return P.mean(axis=0)
    return (P * w[:, None]).sum(axis=0) / max(w.sum(), 1e-300)


def weighted_cov(X, w=None, center=True):
    if w is None:
        w = np.ones(len(X))
    w = w / max(w.sum(), 1e-300)
    if center:
        X = X - (X * w[:, None]).sum(axis=0)
    return (X * w[:, None]).T @ X


def pca_frame(P, w=None):
    """Mittelpunkt + Eigenvektoren (Spalten, absteigende Varianz)."""
    c = weighted_mean(P, w)
    C = weighted_cov(P - c, w, center=False)
    ev, evec = np.linalg.eigh(C)
    order = np.argsort(ev)[::-1]
    return c, ev[order], evec[:, order]


def rotation_between(a, b):
    """Rotationsmatrix, die Einheitsvektor a auf b dreht."""
    a, b = unit(a), unit(b)
    v = np.cross(a, b)
    c = float(np.dot(a, b))
    if np.linalg.norm(v) < 1e-12:
        if c > 0:
            return np.eye(3)
        x, _, _ = orthonormal_frame(a)
        return 2 * np.outer(x, x) - np.eye(3)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx * (1.0 / (1.0 + c))


def perturb_dir(a0, u, v, alpha, beta):
    return unit(a0 + alpha * u + beta * v)


# ---------------------------------------------------------------- Winkel
def angular_range(theta, full_gap_deg=12.0, margin=0.0):
    """Abgedeckter Winkelbereich einer Winkelmenge.

    Gibt (start, ende) mit ende > start zurück; (0, 2pi) wenn keine
    nennenswerte Lücke existiert.
    """
    t = np.sort(np.mod(np.asarray(theta, float), TWO_PI))
    if len(t) < 3:
        return 0.0, TWO_PI
    gaps = np.diff(np.concatenate([t, [t[0] + TWO_PI]]))
    k = int(np.argmax(gaps))
    g = gaps[k]
    if g < np.radians(full_gap_deg):
        return 0.0, TWO_PI
    start = t[(k + 1) % len(t)]
    end = t[k]
    if end <= start:
        end += TWO_PI
    start -= margin
    end += margin
    if end - start >= TWO_PI - 1e-6:
        return 0.0, TWO_PI
    if start > np.pi:
        start -= TWO_PI
        end -= TWO_PI
    return float(start), float(end)


# ---------------------------------------------------------------- 2D-Kreis
def fit_circle_2d(Q, w=None, robust_scale=None):
    """Kreisfit in 2D: algebraisch (Kasa) + geometrische Verfeinerung."""
    Q = np.asarray(Q, float)
    if w is None:
        w = np.ones(len(Q))
    sw = np.sqrt(w / w.mean())
    x, y = Q[:, 0], Q[:, 1]
    A = np.c_[2 * x, 2 * y, np.ones_like(x)] * sw[:, None]
    b = (x * x + y * y) * sw
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    cx, cy, c = sol
    R = np.sqrt(max(c + cx * cx + cy * cy, 1e-12))

    def res(p):
        return sw * (np.hypot(x - p[0], y - p[1]) - p[2])

    kw = {}
    if robust_scale:
        kw = dict(loss="soft_l1", f_scale=robust_scale)
    r = least_squares(res, [cx, cy, R], **kw)
    cx, cy, R = r.x
    d = np.hypot(x - cx, y - cy) - abs(R)
    return np.array([cx, cy]), abs(R), d


# ---------------------------------------------------------------- Polylinien
def polyline_length(P, closed=False):
    P = np.asarray(P, float)
    if closed:
        P = np.vstack([P, P[:1]])
    return float(np.linalg.norm(np.diff(P, axis=0), axis=1).sum())


def resample_polyline(P, step, closed=False):
    P = np.asarray(P, float)
    if closed:
        P = np.vstack([P, P[:1]])
    seg = np.linalg.norm(np.diff(P, axis=0), axis=1)
    keep = np.r_[True, seg > 1e-12]
    P = P[keep]
    seg = np.linalg.norm(np.diff(P, axis=0), axis=1)
    s = np.r_[0.0, np.cumsum(seg)]
    L = s[-1]
    if L <= 0:
        return P[:1]
    n = max(int(np.ceil(L / step)), 4 if closed else 2)
    if closed:
        si = np.linspace(0, L, n, endpoint=False)
    else:
        si = np.linspace(0, L, n)
    out = np.empty((len(si), P.shape[1]))
    for d in range(P.shape[1]):
        out[:, d] = np.interp(si, s, P[:, d])
    return out


def douglas_peucker(P, tol, closed=False):
    """Indizes der beibehaltenen Punkte (Ramer-Douglas-Peucker)."""
    P = np.asarray(P, float)
    n = len(P)
    if n < 3:
        return np.arange(n)
    if closed:
        # an den zwei am weitesten entfernten Punkten aufteilen
        c = P.mean(axis=0)
        i0 = int(np.argmax(np.linalg.norm(P - c, axis=1)))
        i1 = int(np.argmax(np.linalg.norm(P - P[i0], axis=1)))
        a, b = sorted((i0, i1))
        if a == b:
            return np.arange(n)
        part1 = np.arange(a, b + 1)
        part2 = np.r_[np.arange(b, n), np.arange(0, a + 1)]
        k1 = part1[_dp_open(P[part1], tol)]
        k2 = part2[_dp_open(P[part2], tol)]
        keep = np.unique(np.r_[k1, k2])
        return keep
    return _dp_open(P, tol)


def _dp_open(P, tol):
    n = len(P)
    keep = np.zeros(n, bool)
    keep[0] = keep[-1] = True
    stack = [(0, n - 1)]
    while stack:
        a, b = stack.pop()
        if b <= a + 1:
            continue
        seg = P[b] - P[a]
        L = np.linalg.norm(seg)
        Q = P[a + 1:b] - P[a]
        if L < 1e-12:
            d = np.linalg.norm(Q, axis=1)
        else:
            t = np.clip(Q @ seg / (L * L), 0, 1)
            d = np.linalg.norm(Q - t[:, None] * seg, axis=1)
        k = int(np.argmax(d))
        if d[k] > tol:
            m = a + 1 + k
            keep[m] = True
            stack.append((a, m))
            stack.append((m, b))
    return np.nonzero(keep)[0]


def polygon_area_2d(P):
    x, y = P[:, 0], P[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def point_in_polygon(pt, poly):
    x, y = pt
    inside = False
    n = len(poly)
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        if (yi > y) != (yj > y):
            xint = (xj - xi) * (y - yi) / (yj - yi + 1e-300) + xi
            if x < xint:
                inside = not inside
        j = i
    return inside


def dist_points_to_polyline(Q, P, closed=False):
    """Exakter Abstand Punkte->Polylinie (vektorisiert über Segmente, blockweise)."""
    Q = np.asarray(Q, float)
    P = np.asarray(P, float)
    if closed:
        P = np.vstack([P, P[:1]])
    A = P[:-1]
    D = P[1:] - A
    L2 = np.maximum((D * D).sum(1), 1e-300)
    out = np.full(len(Q), np.inf)
    block = max(1, int(2e6 // max(len(A), 1)))
    for s in range(0, len(Q), block):
        q = Q[s:s + block]
        W = q[:, None, :] - A[None, :, :]
        t = np.clip((W * D[None]).sum(2) / L2[None], 0, 1)
        d = np.linalg.norm(W - t[..., None] * D[None], axis=2)
        out[s:s + block] = d.min(axis=1)
    return out


def line_fit_2d(Q, w=None):
    c, ev, evec = pca_frame(Q, w)
    return c, evec[:, 0]


def intersect_lines_2d(p1, d1, p2, d2):
    A = np.array([d1, -d2]).T
    if abs(np.linalg.det(A)) < 1e-12:
        return None
    t = np.linalg.solve(A, p2 - p1)
    return p1 + t[0] * d1


def min_area_rect(Q):
    """Minimales umschließendes Rechteck: (Mitte, Winkel, Breite, Höhe)."""
    from scipy.spatial import ConvexHull
    Q = np.asarray(Q, float)
    try:
        H = Q[ConvexHull(Q).vertices]
    except Exception:
        H = Q
    best = None
    E = np.diff(np.vstack([H, H[:1]]), axis=0)
    for e in E:
        if np.linalg.norm(e) < 1e-12:
            continue
        ang = np.arctan2(e[1], e[0])
        c, s = np.cos(ang), np.sin(ang)
        R = np.array([[c, s], [-s, c]])
        X = H @ R.T
        mn, mx = X.min(0), X.max(0)
        area = np.prod(mx - mn)
        if best is None or area < best[0]:
            mid = (mn + mx) / 2
            best = (area, R.T @ mid, ang, mx[0] - mn[0], mx[1] - mn[1])
    _, mid, ang, w, h = best
    return mid, ang, w, h
