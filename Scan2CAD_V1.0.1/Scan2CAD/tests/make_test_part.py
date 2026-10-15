"""Erzeugt ein synthetisches 'Scan'-Testteil (STL mit Rauschen, beliebige Lage)."""
import sys, os
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "app"))
from OCP.gp import gp_Pnt, gp_Dir, gp_Ax2
from OCP.BRepPrimAPI import BRepPrimAPI_MakeCylinder, BRepPrimAPI_MakeSphere, BRepPrimAPI_MakeCone
from OCP.BRepFilletAPI import BRepFilletAPI_MakeFillet
from OCP.BRepAlgoAPI import BRepAlgoAPI_Fuse, BRepAlgoAPI_Cut
from OCP.TopExp import TopExp_Explorer
from OCP.TopAbs import TopAbs_EDGE
from OCP.BRep import BRep_Tool
from core import occ_build as ob
from core.occ_compat import to_edge
import pyvista as pv

def part():
    prof = {"type": "poly", "pts": [[-30, -20], [30, -20], [30, 20], [-30, 20]], "fillets": [4, 4, 4, 4]}
    base = ob.build_extrusion({"origin": np.zeros(3), "dir": np.array([0, 0, 1.]), "xdir": np.array([1, 0, 0.]),
                               "z0": 0.0, "z1": 15.0, "loops": [prof]}, "add")
    boss = BRepPrimAPI_MakeCylinder(gp_Ax2(gp_Pnt(-15, 0, 14), gp_Dir(0, 0, 1)), 8, 9).Shape()
    # obere Kante des Zapfens verrunden (Torus)
    mf = BRepFilletAPI_MakeFillet(boss)
    # Kante bei z=23 finden
    exp = TopExp_Explorer(boss, TopAbs_EDGE)
    from OCP.BRepAdaptor import BRepAdaptor_Curve
    while exp.More():
        e = to_edge(exp.Current())
        ad = BRepAdaptor_Curve(e)
        p = ad.Value(ad.FirstParameter())
        if abs(p.Z() - 23) < 1e-6:
            mf.Add(2.0, e)
        exp.Next()
    boss = mf.Shape()
    s = BRepAlgoAPI_Fuse(base, boss).Shape()
    dome = BRepPrimAPI_MakeSphere(gp_Pnt(15, 0, 9), 9).Shape()
    s = BRepAlgoAPI_Fuse(s, dome).Shape()
    hole = BRepPrimAPI_MakeCylinder(gp_Ax2(gp_Pnt(15, 12, -1), gp_Dir(0, 0, 1)), 3, 30).Shape()
    s = BRepAlgoAPI_Cut(s, hole).Shape()
    csk = BRepPrimAPI_MakeCone(gp_Ax2(gp_Pnt(15, 12, 10), gp_Dir(0, 0, 1)), 0.0, 6.0, 6.0).Shape()
    s = BRepAlgoAPI_Cut(s, csk).Shape()
    return ob.fix_shape(s)

if __name__ == "__main__":
    out = sys.argv[1]
    noise = float(sys.argv[2]) if len(sys.argv) > 2 else 0.02
    shp = part()
    print("valid", ob.is_valid(shp))
    V, F = ob.tessellate(shp, 0.01, 0.15)
    pd = pv.PolyData(V, np.hstack([np.full((len(F), 1), 3), F]).ravel()).clean(lines_to_points=True, polys_to_lines=False).triangulate()
    pd = pd.extract_cells(np.nonzero(pd.compute_cell_sizes(length=False, volume=False).cell_data["Area"] > 1e-12)[0]).extract_surface().triangulate()
    if len(sys.argv) > 3 and sys.argv[3] == "uniform":
        import pyacvd
        pd = pd.subdivide_adaptive(max_edge_len=0.5).clean()
        cl = pyacvd.Clustering(pd)
        cl.cluster(int(sys.argv[4]) if len(sys.argv) > 4 else 150000)
        pd = cl.create_mesh().clean()
    else:
        pd = pd.subdivide_adaptive(max_edge_len=0.7)
        pd = pd.clean()
    pd = pd.compute_normals(point_normals=True, cell_normals=False, split_vertices=False)
    rng = np.random.default_rng(3)
    pts = pd.points + pd.point_data["Normals"] * rng.normal(0, noise, (pd.n_points, 1))
    # zufällige Lage
    from scipy.spatial.transform import Rotation
    R = Rotation.from_euler("xyz", [23, -41, 67], degrees=True).as_matrix()
    T = np.eye(4); T[:3, :3] = R; T[:3, 3] = [120, -35, 48]
    pts = pts @ R.T + T[:3, 3]
    np.save(out + ".T.npy", T)
    pd.points = pts
    pd = pd.triangulate()
    pd.save(out)
    print("faces", pd.n_cells, "points", pd.n_points)
