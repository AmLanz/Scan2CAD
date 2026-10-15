"""Kompatibilitätsschicht für OCP 7.x und 8.x (OCCT 8 hat Collections und TopoDS-Casts umbenannt)."""
from __future__ import annotations

from OCP.TopoDS import TopoDS

try:  # OCCT <= 7.9
    from OCP.TColgp import TColgp_Array2OfPnt as Array2Pnt  # noqa: F401
    from OCP.TColgp import TColgp_HArray1OfPnt as HArray1Pnt  # noqa: F401
    from OCP.TColgp import TColgp_HArray1OfPnt2d as HArray1Pnt2d  # noqa: F401
    from OCP.TColStd import TColStd_Array1OfReal as Array1Real  # noqa: F401
    from OCP.TColStd import TColStd_Array1OfInteger as Array1Int  # noqa: F401
except ImportError:  # OCCT 8
    from OCP.collections import Array2_gp_Pnt as Array2Pnt  # noqa: F401
    from OCP.collections import HArray1_gp_Pnt as HArray1Pnt  # noqa: F401
    from OCP.collections import HArray1_gp_Pnt2d as HArray1Pnt2d  # noqa: F401
    from OCP.collections import Array1_double as Array1Real  # noqa: F401
    from OCP.collections import Array1_int as Array1Int  # noqa: F401


try:
    from OCP.TopTools import TopTools_ListOfShape as ListOfShape  # noqa: F401
except ImportError:
    from OCP.collections import List_TopoDS_Shape as ListOfShape  # noqa: F401


def _cast(name):
    f = getattr(TopoDS, name + "_s", None)
    if f is None:
        f = getattr(TopoDS, name)
    return f


to_edge = _cast("Edge")
to_face = _cast("Face")
to_wire = _cast("Wire")
to_vertex = _cast("Vertex")
to_shell = _cast("Shell")
to_solid = _cast("Solid")


try:
    from OCP.TopTools import TopTools_IndexedDataMapOfShapeListOfShape as ShapeListMap  # noqa: F401
except ImportError:
    from OCP.collections import (  # noqa: F401
        IndexedDataMap_TopoDS_Shape_List_TopoDS_Shape_TopTools_ShapeMapHasher as ShapeListMap)
