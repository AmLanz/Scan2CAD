"""Freiform: Parametrisierung eines Netz-Patches + glättender Tensorprodukt-B-Spline-Fit.

Modi der Parametrisierung
  'harmonic' : Tutte/harmonische Abbildung des Patches auf das Einheitsquadrat
               (Rand -> Quadratrand). Fläche braucht keine Trimmung.
  'planar'   : Projektion auf die Ausgleichsebene; der Patch-Rand wird als
               Trimmkurve im Parameterraum mitgegeben. Gut für 'Höhenfeld'-artige Patches.
  'auto'     : 'planar', wenn alle Normalen < 70° zur Ebenennormalen stehen, sonst 'harmonic'.
"""
from __future__ import annotations

import numpy as np
from scipy import sparse
from scipy.interpolate import BSpline
from scipy.sparse.linalg import spsolve

from .geom_utils import douglas_peucker, pca_frame, polygon_area_2d, unit
from .mesh_model import boundary_loops_of


def clamped_knots(n_ctrl, k=3):
    n_int = n_ctrl - k - 1
    inner = np.linspace(0, 1, n_int + 2)[1:-1] if n_int > 0 else np.array([])
    return np.r_[np.zeros(k + 1), inner, np.ones(k + 1)]


def _design(x, t, k):
    x = np.clip(x, 0.0, 1.0 - 1e-12)
    return BSpline.design_matrix(x, t, k).tocsr()


def _khatri_rao(Bu, Bv):
    """Zeilenweises Kronecker-Produkt zweier CSR-Matrizen."""
    Bu = Bu.tocsr()
    Bv = Bv.tocsr()
    nv = Bv.shape[1]
    nu_nnz = np.diff(Bu.indptr)
    nv_nnz = np.diff(Bv.indptr)
    ku, kv = int(nu_nnz.max()), int(nv_nnz.max())
    N = Bu.shape[0]
    iu = np.zeros((N, ku), np.int64)
    du = np.zeros((N, ku))
    iv = np.zeros((N, kv), np.int64)
    dv = np.zeros((N, kv))
    for M, I, D, kk in ((Bu, iu, du, ku), (Bv, iv, dv, kv)):
        cnt = np.diff(M.indptr)
        pos = np.arange(M.nnz) - np.repeat(M.indptr[:-1], cnt)
        r = np.repeat(np.arange(N), cnt)
        I[r, pos] = M.indices
        D[r, pos] = M.data
    rr = np.repeat(np.arange(N), ku * kv)
    cc = (iu[:, :, None] * nv + iv[:, None, :]).reshape(-1)
    vv = (du[:, :, None] * dv[:, None, :]).reshape(-1)
    return sparse.csr_matrix((vv, (rr, cc)), shape=(N, Bu.shape[1] * nv))


def _diff(n, order):
    D = sparse.eye(n, format="csr")
    for _ in range(order):
        D = (D[1:] - D[:-1])
    return D


def smoothing_matrix(nu, nv):
    Iu, Iv = sparse.eye(nu), sparse.eye(nv)
    D2u = sparse.kron(_diff(nu, 2), Iv)
    D2v = sparse.kron(Iu, _diff(nv, 2))
    Duv = sparse.kron(_diff(nu, 1), _diff(nv, 1))
    return (D2u.T @ D2u + D2v.T @ D2v + 2 * Duv.T @ Duv).tocsr()


# ------------------------------------------------------------ Parametrisierung
def tutte_square(V, F, loop):
    """Harmonische Abbildung (uniforme Gewichte) mit Rand auf das Einheitsquadrat."""
    nV = len(V)
    Pl = V[loop]
    c, ev, evec = pca_frame(V)
    X = (Pl - c) @ evec[:, :2]
    mn, mx = X.min(0), X.max(0)
    box = np.array([[mn[0], mn[1]], [mx[0], mn[1]], [mx[0], mx[1]], [mn[0], mx[1]]])
    corners = [int(np.argmin(np.linalg.norm(X - b, axis=1))) for b in box]
    # Reihenfolge entlang der Schleife sichern
    order = np.argsort(corners)
    corners = [corners[i] for i in order]
    if len(set(corners)) < 4:
        m = len(loop)
        corners = [0, m // 4, m // 2, 3 * m // 4]
    # Orientierung: Quadrat gegen den Uhrzeigersinn
    sq = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], float)
    m = len(loop)
    seg = np.linalg.norm(np.diff(np.vstack([Pl, Pl[:1]]), axis=0), axis=1)
    uv_b = np.zeros((m, 2))
    for s in range(4):
        a, b = corners[s], corners[(s + 1) % 4]
        idx = np.arange(a, b + 1) if b > a else np.r_[np.arange(a, m), np.arange(0, b + 1)]
        lens = np.r_[0, np.cumsum(seg[idx[:-1]])]
        t = lens / max(lens[-1], 1e-12)
        uv_b[idx] = sq[s] + t[:, None] * (sq[(s + 1) % 4] - sq[s])
    # Laplace
    e = F[:, [0, 1, 1, 2, 2, 0]].reshape(-1, 2)
    e = np.unique(np.sort(e, axis=1), axis=0)
    W = sparse.csr_matrix((np.ones(2 * len(e)), (np.r_[e[:, 0], e[:, 1]], np.r_[e[:, 1], e[:, 0]])), shape=(nV, nV))
    W.data[:] = 1.0
    deg = np.asarray(W.sum(1)).ravel()
    L = sparse.diags(deg) - W
    fixed = np.zeros(nV, bool)
    fixed[loop] = True
    free = np.nonzero(~fixed)[0]
    uv = np.zeros((nV, 2))
    uv[loop] = uv_b
    if len(free):
        Lff = L[free][:, free].tocsc()
        Lfb = L[free][:, loop]
        rhs = -Lfb @ uv_b
        sol = spsolve(Lff, rhs)
        uv[free] = sol.reshape(-1, 2)
    # Fläche sollte nicht gespiegelt sein (für konsistente Flächennormale)
    return np.clip(uv, 0, 1)


def planar_params(V, normals_hint=None, margin=0.02):
    c, ev, evec = pca_frame(V)
    ax_u, ax_v, ax_n = evec[:, 0], evec[:, 1], evec[:, 2]
    if normals_hint is not None and np.mean(normals_hint @ ax_n) < 0:
        ax_v = -ax_v
        ax_n = -ax_n
    X = np.c_[(V - c) @ ax_u, (V - c) @ ax_v]
    mn, mx = X.min(0), X.max(0)
    span = mx - mn
    mn = mn - margin * span
    mx = mx + margin * span
    uv = (X - mn) / (mx - mn)
    return uv, (c, ax_u, ax_v, ax_n, mn, mx)


# ------------------------------------------------------------ Fit
def fit_bspline_surface(P, uv, w, nu, nv, smooth=0.3, k=3, param_iters=2):
    tu, tv = clamped_knots(nu, k), clamped_knots(nv, k)
    w = np.asarray(w, float)
    w = w / w.mean()
    R = smoothing_matrix(nu, nv)
    uv = uv.copy()
    ctrl = None
    for it in range(param_iters + 1):
        B = _khatri_rao(_design(uv[:, 0], tu, k), _design(uv[:, 1], tv, k))
        BtW = B.T.multiply(w[None, :]).tocsr()
        A = (BtW @ B).tocsr()
        lam_scale = A.diagonal().sum() / max(R.diagonal().sum(), 1e-300)
        lam = lam_scale * (10 ** (-6 + 6 * float(np.clip(smooth, 0, 1)))) + 1e-12 * lam_scale
        M = (A + lam * R).tocsc()
        rhs = BtW @ P
        ctrl = np.column_stack([spsolve(M, rhs[:, d]) for d in range(3)])
        if it < param_iters:
            uv = _reproject(P, uv, ctrl.reshape(nu, nv, 3), tu, tv, k)
    S = (B @ ctrl)
    res = np.linalg.norm(P - S, axis=1)
    # Vorzeichen über Flächennormale (zur Anzeige)
    return ctrl.reshape(nu, nv, 3), tu, tv, uv, res


def _reproject(P, uv, C, tu, tv, k):
    try:
        from scipy.interpolate import NdBSpline
    except ImportError:
        return uv
    spl = NdBSpline((tu, tv), C, k)
    for _ in range(3):
        xi = np.clip(uv, 0, 1)
        S = spl(xi)
        Su = spl(xi, nu=(1, 0))
        Sv = spl(xi, nu=(0, 1))
        r = P - S
        a11 = (Su * Su).sum(1)
        a12 = (Su * Sv).sum(1)
        a22 = (Sv * Sv).sum(1)
        b1 = (Su * r).sum(1)
        b2 = (Sv * r).sum(1)
        det = a11 * a22 - a12 * a12
        ok = np.abs(det) > 1e-12
        du = np.where(ok, (a22 * b1 - a12 * b2) / np.where(ok, det, 1), 0)
        dv = np.where(ok, (a11 * b2 - a12 * b1) / np.where(ok, det, 1), 0)
        step = np.c_[du, dv]
        nstep = np.linalg.norm(step, axis=1)
        step *= np.minimum(1.0, 0.05 / np.maximum(nstep, 1e-12))[:, None]
        uv = np.clip(xi + step, 0, 1)
    return uv


def fit_freeform_patch(V, F, n_ctrl=12, smooth=0.3, mode="auto", trim_tol=None):
    """V, F: Teilnetz (zusammenhängend). Rückgabe: dict mit Flächenparametern + Residuen."""
    V = np.asarray(V, float)
    F = np.asarray(F, np.int64)
    loops = boundary_loops_of(F, len(V))
    if not loops:
        raise RuntimeError("Freiform braucht einen Patch mit Rand (keine geschlossene Fläche).")
    # Flächennormalen
    tri = V[F]
    cr = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    area = 0.5 * np.linalg.norm(cr, axis=1)
    fn = unit(cr)
    va = np.zeros(len(V))
    vn = np.zeros((len(V), 3))
    for j in range(3):
        va += np.bincount(F[:, j], weights=area / 3, minlength=len(V))
        for d in range(3):
            vn[:, d] += np.bincount(F[:, j], weights=cr[:, d], minlength=len(V))
    vn = unit(vn)
    c, ev, evec = pca_frame(V, va)
    nrm = evec[:, 2]
    if np.mean(fn @ nrm) < 0:
        nrm = -nrm
    if mode == "auto":
        cosang = fn @ nrm
        mode = "planar" if np.percentile(cosang, 2) > np.cos(np.radians(70)) else "harmonic"
    trim = None
    if mode == "planar":
        uv, frame = planar_params(V, fn)
        _, ax_u, ax_v, ax_n, mn, mx = frame
        span = (mx - mn)
        aspect = span[0] / max(span[1], 1e-9)
    else:
        uv = tutte_square(V, F, loops[0])
        # Seitenverhältnis aus mittleren 3D-Längen
        aspect = 1.0
    if aspect >= 1:
        nu, nv = n_ctrl, max(4, int(round(n_ctrl / aspect)))
    else:
        nv, nu = n_ctrl, max(4, int(round(n_ctrl * aspect)))
    nu, nv = max(nu, 4), max(nv, 4)
    C, tu, tv, uv_fit, res = fit_bspline_surface(V, uv, va, nu, nv, smooth)
    if mode == "planar":
        from .profiles import fit_spline_profile
        loop = loops[0]
        # Rand in mm (Ebenenkoordinaten), glätten, zurück in (u, v)
        luv = uv_fit[loop]
        sc_ = (mx - mn)
        lmm = luv * sc_
        tol_mm = max(trim_tol or 0.05, 0.02)
        try:
            spl = fit_spline_profile(lmm, True, 0.8, 4 * tol_mm, max(0.5, 20 * tol_mm))
            lmm = np.asarray(spl["pts"], float)
        except Exception:
            lmm = lmm[douglas_peucker(lmm, tol_mm, closed=True)]
        luv = np.clip(lmm / sc_, 0.0, 1.0)
        if polygon_area_2d(luv) < 0:
            luv = luv[::-1]
        trim = luv
    # Orientierung der Fläche: Normale soll wie Netznormalen zeigen
    return {
        "poles": C, "knots_u": tu, "knots_v": tv, "degree": 3, "mode": mode,
        "trim_uv": trim, "residuals": res, "weights": va, "n_u": nu, "n_v": nv,
        "normal_hint": nrm,
    }
