"""Erzeugt einen realistischen 'Scan' (STL) aus einem CAD-Testteil.

Wie ein Streifenlicht-Scanner:
  * Abtastung über ein Distanzfeld + Marching Cubes  -> unregelmäßige Dreiecke, keine CAD-Kanten
  * Rauschen entlang der Normalen
  * Bohrungen: der tiefe Teil wird nicht erfasst (offen), am Rand der Lücke Ausreißer ("Murks")
  * optional offener Boden (--open-bottom): unterhalb z_cut fehlen alle Daten
  * ein paar kleine Fehlstellen, zufällige Lage im Raum

python tests/make_scan_like.py scan.stl [voxel=0.25] [rauschen=0.015] [--part testteil|flansch] [--open-bottom]
Schreibt zusätzlich scan.stl.T.npy (Lage) und scan.stl.json (Sollwerte).
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "app"))
import pyvista as pv  # noqa: E402
from OCP.BRepAlgoAPI import BRepAlgoAPI_Cut, BRepAlgoAPI_Fuse  # noqa: E402
from OCP.BRepPrimAPI import (BRepPrimAPI_MakeCone, BRepPrimAPI_MakeCylinder,  # noqa: E402
                             BRepPrimAPI_MakeSphere)
from OCP.gp import gp_Ax2, gp_Dir, gp_Pnt  # noqa: E402

from core import occ_build as ob  # noqa: E402


def _fillet_top_edge(shape, z, r):
    from OCP.BRepAdaptor import BRepAdaptor_Curve
    from OCP.BRepFilletAPI import BRepFilletAPI_MakeFillet
    from OCP.TopAbs import TopAbs_EDGE
    from OCP.TopExp import TopExp_Explorer
    from core.occ_compat import to_edge
    mf = BRepFilletAPI_MakeFillet(shape)
    e = TopExp_Explorer(shape, TopAbs_EDGE)
    n = 0
    while e.More():
        ed = to_edge(e.Current())
        ad = BRepAdaptor_Curve(ed)
        p = ad.Value(0.5 * (ad.FirstParameter() + ad.LastParameter()))
        if abs(p.Z() - z) < 1e-6:
            mf.Add(r, ed)
            n += 1
        e.Next()
    return mf.Shape() if n else shape


def drill(solid, x, y, z_top, dia, depth, point_deg=118.0, through=False, csk=None):
    """Bohrung von z_top nach unten (-Z)."""
    if through:
        cyl = BRepPrimAPI_MakeCylinder(gp_Ax2(gp_Pnt(x, y, -50), gp_Dir(0, 0, 1)), dia / 2, 200).Shape()
        s = BRepAlgoAPI_Cut(solid, cyl).Shape()
    else:
        cyl = BRepPrimAPI_MakeCylinder(gp_Ax2(gp_Pnt(x, y, z_top - depth), gp_Dir(0, 0, 1)), dia / 2,
                                       depth + 5).Shape()
        s = BRepAlgoAPI_Cut(solid, cyl).Shape()
        h = (dia / 2) / np.tan(np.radians(point_deg / 2))
        cone = BRepPrimAPI_MakeCone(gp_Ax2(gp_Pnt(x, y, z_top - depth - h), gp_Dir(0, 0, 1)),
                                    0.0, dia / 2, h).Shape()
        s = BRepAlgoAPI_Cut(s, cone).Shape()
    if csk:
        d_csk, ang = csk
        hh = (d_csk / 2) / np.tan(np.radians(ang / 2))
        cone = BRepPrimAPI_MakeCone(gp_Ax2(gp_Pnt(x, y, z_top - hh), gp_Dir(0, 0, 1)), 0.0, d_csk / 2 * 1.6,
                                    hh * 1.6).Shape()
        s = BRepAlgoAPI_Cut(s, cone).Shape()
    return s


def part_testteil():
    prof = {"type": "poly", "pts": [[-30, -20], [30, -20], [30, 20], [-30, 20]], "fillets": [4, 4, 4, 4]}
    base = ob.build_extrusion({"origin": np.zeros(3), "dir": np.array([0, 0, 1.]), "xdir": np.array([1, 0, 0.]),
                               "z0": 0.0, "z1": 15.0, "loops": [prof]}, "add")
    boss = BRepPrimAPI_MakeCylinder(gp_Ax2(gp_Pnt(-15, 0, 14), gp_Dir(0, 0, 1)), 8, 9).Shape()
    boss = _fillet_top_edge(boss, 23.0, 2.0)
    s = BRepAlgoAPI_Fuse(base, boss).Shape()
    s = BRepAlgoAPI_Fuse(s, BRepPrimAPI_MakeSphere(gp_Pnt(15, 0, 9), 9).Shape()).Shape()
    s = drill(s, 15, 12, 15, 6.0, 0, through=True, csk=(10.0, 90.0))
    truth = {"part": "testteil", "volume_hint": None,
             "holes": [{"xy": [15, 12], "dia": 6.0, "through": True, "csk": [10.0, 90.0]}],
             "bbox": [60, 40, 23]}
    return ob.fix_shape(s), truth


def part_flansch():
    prof = {"type": "poly", "pts": [[-30, -30], [30, -30], [30, 30], [-30, 30]], "fillets": [6, 6, 6, 6]}
    base = ob.build_extrusion({"origin": np.zeros(3), "dir": np.array([0, 0, 1.]), "xdir": np.array([1, 0, 0.]),
                               "z0": 0.0, "z1": 10.0, "loops": [prof]}, "add")
    boss = BRepPrimAPI_MakeCylinder(gp_Ax2(gp_Pnt(0, 0, 9), gp_Dir(0, 0, 1)), 15, 13).Shape()
    s = BRepAlgoAPI_Fuse(base, boss).Shape()
    s = drill(s, 0, 0, 22, 12.0, 0, through=True)
    holes = [{"xy": [0, 0], "dia": 12.0, "through": True}]
    for k in range(4):
        a = np.radians(45 + 90 * k)
        x, y = 22 * np.cos(a), 22 * np.sin(a)
        s = drill(s, x, y, 10, 6.6, 0, through=True)
        holes.append({"xy": [round(x, 4), round(y, 4)], "dia": 6.6, "through": True})
    for x in (-9.5, 9.5):
        s = drill(s, x, 0, 22, 4.2, 8.0, point_deg=118.0)
        holes.append({"xy": [x, 0], "dia": 4.2, "depth": 8.0, "point": 118.0, "thread": "M5"})
    truth = {"part": "flansch", "holes": holes, "bbox": [60, 60, 22],
             "symmetry": "2-zählig um Z (4-zählig nur ohne die beiden Gewindelöcher), 2 Spiegelebenen"}
    return ob.fix_shape(s), truth


def volume(shape):
    from OCP.BRepGProp import BRepGProp
    from OCP.GProp import GProp_GProps
    g = GProp_GProps()
    BRepGProp.VolumeProperties_s(shape, g)
    return float(g.Mass())


def scan_like(shape, voxel=0.25, noise=0.015, open_bottom=False, holes=(), seed=3):
    rng = np.random.default_rng(seed)
    V, F = ob.tessellate(shape, 0.01, 0.1)
    ref = pv.PolyData(V, np.hstack([np.full((len(F), 1), 3), F]).ravel()).clean()
    b = np.array(ref.bounds).reshape(3, 2)
    lo, hi = b[:, 0] - 3 * voxel, b[:, 1] + 3 * voxel
    dims = np.ceil((hi - lo) / voxel).astype(int) + 1
    grid = pv.ImageData(dimensions=dims, spacing=(voxel,) * 3, origin=lo)
    grid = grid.compute_implicit_distance(ref)
    surf = grid.contour([0.0], scalars="implicit_distance").triangulate().clean()
    surf = surf.compute_normals(point_normals=True, cell_normals=False, split_vertices=False,
                                auto_orient_normals=True)
    P = np.asarray(surf.points).copy()
    N = np.asarray(surf.point_data["Normals"])
    F2 = np.asarray(surf.regular_faces)
    C = P[F2].mean(1)
    drop = np.zeros(len(F2), bool)
    noisy = np.zeros(len(P), bool)
    # Bohrungen: tiefer Teil unerfasst (Sichtschatten), Rand der Lücke mit Ausreißern
    for h in holes:
        x, y = h["xy"]
        r = h["dia"] / 2
        z_top = h.get("z_top")
        seen = h.get("seen_depth")
        if z_top is None or seen is None:
            continue
        inside = (np.hypot(C[:, 0] - x, C[:, 1] - y) < r + 0.6 * voxel) & (C[:, 2] < z_top - seen)
        if h.get("both_sides") and not open_bottom:
            inside &= C[:, 2] > h["z_bottom"] + seen
        drop |= inside
        band = (np.hypot(P[:, 0] - x, P[:, 1] - y) < r + voxel) & (np.abs(P[:, 2] - (z_top - seen)) < 1.5)
        noisy |= band
    if open_bottom:
        drop |= C[:, 2] < 1.2
    # einige kleine Fehlstellen
    for _ in range(4):
        c = C[rng.integers(len(C))]
        drop |= np.linalg.norm(C - c, axis=1) < rng.uniform(0.8, 1.6)
    F2 = F2[~drop]
    used = np.unique(F2)
    remap = -np.ones(len(P), np.int64)
    remap[used] = np.arange(len(used))
    P, N, noisy, F2 = P[used], N[used], noisy[used], remap[F2]
    sig = np.full(len(P), noise)
    sig[noisy] = 6 * noise
    P = P + N * rng.normal(0, 1, (len(P), 1)) * sig[:, None]
    return P, F2


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    out = args[0]
    voxel = float(args[1]) if len(args) > 1 else 0.25
    noise = float(args[2]) if len(args) > 2 else 0.015
    part = "testteil"
    for a in sys.argv[1:]:
        if a.startswith("--part="):
            part = a.split("=", 1)[1]
    shape, truth = part_flansch() if part == "flansch" else part_testteil()
    truth["volume"] = volume(shape)
    open_bottom = "--open-bottom" in sys.argv
    holes = []
    for h in truth["holes"]:
        hh = dict(h)
        if part == "flansch":
            if h["dia"] == 12.0:
                hh.update(z_top=22.0, z_bottom=0.0, seen_depth=9.0, both_sides=True)
            elif h.get("through"):
                hh.update(z_top=10.0, z_bottom=0.0, seen_depth=3.5, both_sides=True)
            else:
                hh.update(z_top=22.0, seen_depth=4.0)
        else:
            hh.update(z_top=15.0, z_bottom=0.0, seen_depth=6.0, both_sides=True)
        holes.append(hh)
    P, F = scan_like(shape, voxel, noise, open_bottom, holes)
    from scipy.spatial.transform import Rotation
    R = Rotation.from_euler("xyz", [23, -41, 67], degrees=True).as_matrix()
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = [120, -35, 48]
    P = P @ R.T + T[:3, 3]
    np.save(out + ".T.npy", T)
    truth["open_bottom"] = open_bottom
    with open(out + ".json", "w", encoding="utf-8") as fh:
        json.dump(truth, fh, indent=1, ensure_ascii=False)
    pd = pv.PolyData(P, np.hstack([np.full((len(F), 1), 3), F]).ravel())
    pd.save(out)
    print("faces", len(F), "points", len(P), "volume", round(truth["volume"], 2))
