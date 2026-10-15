"""3D-Ansicht (pyvista/VTK in Qt) mit Mal-Interaktion auf dem Netz."""
from __future__ import annotations

import time

import numpy as np
import pyvista as pv
from pyvistaqt import QtInteractor
from qtpy.QtCore import Signal
from qtpy.QtWidgets import QVBoxLayout, QWidget
from vtkmodules.vtkCommonDataModel import vtkStaticCellLocator
from vtkmodules.vtkInteractionStyle import vtkInteractorStyleTrackballCamera
from vtkmodules.vtkRenderingCore import vtkCellPicker

from core.i18n import tr

from .workers import VTK_LOCK


def _guarded(fn):
    def g(*a, **k):
        if VTK_LOCK.acquire(blocking=False):
            try:
                return fn(*a, **k)
            finally:
                VTK_LOCK.release()
        return None
    g._s2c_guarded = True
    return g


def _install_render_guard():
    """paintGL der pyvistaqt-Widgets nur rendern lassen, wenn kein Hintergrund-Job VTK benutzt."""
    try:
        from pyvistaqt.rwi import QVTKRenderWindowInteractor as _RWI
        if not getattr(_RWI.paintGL, "_s2c_guarded", False):
            orig = _RWI.paintGL

            def paintGL(self):
                if VTK_LOCK.acquire(blocking=False):
                    try:
                        return orig(self)
                    finally:
                        VTK_LOCK.release()
                self._s2c_missed = True
            paintGL._s2c_guarded = True
            _RWI.paintGL = paintGL
    except Exception:
        pass

BASE_RGB = np.array([196, 199, 204], np.uint8)
SEL_RGB = np.array([255, 128, 20], np.uint8)


class PaintStyle(vtkInteractorStyleTrackballCamera):
    """Linke Maustaste malt (in Mal-Modi), rechte dreht, mittlere verschiebt, Rad zoomt.
    Im Navigationsmodus Standardverhalten (links drehen, rechts zoomen)."""

    def __init__(self, viewer):
        super().__init__()
        self.viewer = viewer
        self.painting = False
        self.rotating_right = False
        self.AddObserver("LeftButtonPressEvent", self._lp)
        self.AddObserver("LeftButtonReleaseEvent", self._lr)
        self.AddObserver("MouseMoveEvent", self._mm)
        self.AddObserver("RightButtonPressEvent", self._rp)
        self.AddObserver("RightButtonReleaseEvent", self._rr)
        # VTK-Standardtasten (w/s/e/q/3/p …) abschalten – Tasten gehören der GUI
        self.AddObserver("CharEvent", lambda o, e: None)
        self.AddObserver("KeyPressEvent", lambda o, e: None)

    def _pos(self):
        return self.GetInteractor().GetEventPosition()

    def _lp(self, obj, ev):
        if self.viewer.tool == "navigate" or self.GetInteractor().GetAltKey():
            self.OnLeftButtonDown()
            return
        iren = self.GetInteractor()
        erase = bool(iren.GetControlKey()) or self.viewer.tool == "erase"
        self.painting = True
        x, y = self._pos()
        self.viewer._stroke_begin(x, y, erase)

    def _lr(self, obj, ev):
        if self.painting:
            self.painting = False
            self.viewer._stroke_end()
        else:
            self.OnLeftButtonUp()

    def _mm(self, obj, ev):
        x, y = self._pos()
        if self.painting:
            self.viewer._stroke_move(x, y)
        else:
            self.viewer._hover(x, y)
            self.OnMouseMove()

    def _rp(self, obj, ev):
        if self.viewer.tool == "navigate":
            self.OnRightButtonDown()
        else:
            self.rotating_right = True
            self.OnLeftButtonDown()

    def _rr(self, obj, ev):
        if self.rotating_right:
            self.rotating_right = False
            self.OnLeftButtonUp()
        else:
            self.OnRightButtonUp()


class Viewer(QWidget):
    stroke = Signal(object)        # dict(event='begin'|'move'|'end', cell, pos, erase)
    picked = Signal(object)        # dict(cell, pos, erase) für Klick-Werkzeuge

    def __init__(self, parent=None):
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        _install_render_guard()
        self.plotter = QtInteractor(self)
        try:
            self.plotter.render_signal.disconnect(self.plotter._render)
        except Exception:
            pass
        self.plotter.render_signal.connect(_guarded(self.plotter._render))
        lay.addWidget(self.plotter.interactor)
        self.plotter.set_background("#2b2f36", top="#4a5361")
        # FXAA statt SSAA: Kantenglättung als billiger Nachbearbeitungsschritt. SSAA rendert jedes
        # Bild in mehrfacher Auflösung – bei Scans mit Millionen Dreiecken wird das Drehen zäh.
        try:
            self.plotter.enable_anti_aliasing("fxaa")
        except Exception:
            pass
        self.plotter.add_axes(interactive=False, line_width=2, color="white")
        self.style = PaintStyle(self)
        self.plotter.iren.interactor.SetInteractorStyle(self.style)
        try:
            self.plotter.iren.clear_key_event_callbacks()
        except Exception:
            pass
        self.tool = "navigate"
        self.brush_radius = 2.0
        self.pd = None
        self.actor = None
        self.picker = vtkCellPicker()
        self.picker.SetTolerance(1e-5)
        self.picker.PickFromListOn()
        self._locator = None
        self._last_render = 0.0
        self._cursor = None
        self._show_edges = False
        self._overlays = {}
        self._scalar_actor = None

    # ------------------------------------------------------------------ Netz
    def set_mesh(self, V, F, rgb=None, reset_camera=True):
        faces = np.hstack([np.full((len(F), 1), 3, dtype=np.int64), F]).ravel()
        pd = pv.PolyData(np.asarray(V, float), faces)
        if rgb is None:
            rgb = np.tile(BASE_RGB, (len(F), 1))
        pd.cell_data["rgb"] = np.ascontiguousarray(rgb, dtype=np.uint8)
        if self.actor is not None:
            self.plotter.remove_actor(self.actor, render=False)
        self.pd = pd
        self.actor = self.plotter.add_mesh(pd, scalars="rgb", rgb=True, show_edges=self._show_edges,
                                           edge_color="#30343a", line_width=0.5, smooth_shading=False,
                                           specular=0.25, specular_power=20, name="scan",
                                           reset_camera=False, show_scalar_bar=False,
                                           copy_mesh=False)
        try:
            mp = self.actor.GetMapper()
            mp.SetResolveCoincidentTopologyToPolygonOffset()
            mp.SetRelativeCoincidentTopologyPolygonOffsetParameters(4.0, 8.0)
        except Exception:
            pass
        self.picker.InitializePickList()
        self.picker.AddPickList(self.actor)
        loc = vtkStaticCellLocator()
        loc.SetDataSet(pd)
        loc.BuildLocator()
        self._locator = loc
        try:
            self.picker.RemoveAllLocators()
        except Exception:
            pass
        self.picker.AddLocator(loc)
        if reset_camera:
            self.plotter.reset_camera()
        self.plotter.render()

    def _mesh_dataset(self):
        try:
            return self.actor.mapper.dataset
        except Exception:
            return self.pd

    def set_colors(self, rgb, idx=None):
        ds = self._mesh_dataset()
        if ds is None:
            return
        arr = ds.cell_data["rgb"]
        if idx is None:
            arr[:] = rgb
        else:
            arr[idx] = rgb
        ds.GetCellData().GetArray("rgb").Modified()
        ds.Modified()

    def set_mesh_visible(self, vis):
        if self.actor is not None:
            self.actor.SetVisibility(bool(vis))
            self.render()

    def set_mesh_opacity(self, op):
        if self.actor is not None:
            self.actor.GetProperty().SetOpacity(float(op))
            self.render()

    def set_edges(self, on):
        self._show_edges = bool(on)
        if self.actor is not None:
            self.actor.GetProperty().SetEdgeVisibility(self._show_edges)
            self.render()

    def set_busy(self, busy):
        """Während Hintergrundrechnungen keine Maus-Interaktion (Picking/Rendern mit VTK)."""
        try:
            self.plotter.interactor.setEnabled(not busy)
        except Exception:
            pass
        if not busy:
            self.render()

    def render(self, throttle=False):
        now = time.perf_counter()
        if throttle and now - self._last_render < 1 / 45:
            return
        self._last_render = now
        self.plotter.render()

    # ------------------------------------------------------------------ Overlays
    def set_overlay(self, name, V, F, color=(0.3, 0.6, 1.0), opacity=0.6, edges=False, visible=True):
        self.remove_overlay(name, render=False)
        if V is None or len(F) == 0:
            return
        faces = np.hstack([np.full((len(F), 1), 3, dtype=np.int64), F]).ravel()
        pd = pv.PolyData(np.asarray(V, float), faces)
        act = self.plotter.add_mesh(pd, color=color, opacity=opacity, smooth_shading=True,
                                    show_edges=edges, name=name, reset_camera=False,
                                    pickable=False, specular=0.3)
        act.SetVisibility(bool(visible))
        self._overlays[name] = act
        self.render()

    def overlay_visible(self, name, vis):
        if name in self._overlays:
            self._overlays[name].SetVisibility(bool(vis))

    def overlay_opacity(self, op, prefix="feat_"):
        for n, a in self._overlays.items():
            if n.startswith(prefix):
                a.GetProperty().SetOpacity(float(op))
        self.render()

    def remove_overlay(self, name, render=True):
        act = self._overlays.pop(name, None)
        if act is not None:
            self.plotter.remove_actor(act, render=False)
        if render:
            self.render()

    def clear_overlays(self, prefix=""):
        for n in [n for n in self._overlays if n.startswith(prefix)]:
            self.remove_overlay(n, render=False)
        self.render()

    def set_lines(self, name, polylines, color="#ff3b30", width=3.0, visible=True):
        """Polylinien (Liste von Nx3-Arrays) als ein Linien-Overlay."""
        self.remove_overlay(name, render=False)
        polylines = [np.asarray(p, float) for p in polylines if len(p) >= 2]
        if not polylines:
            self.render()
            return
        pts = np.vstack(polylines)
        cells = []
        off = 0
        for p in polylines:
            cells.append(np.r_[len(p), np.arange(off, off + len(p))])
            off += len(p)
        pd = pv.PolyData(pts, lines=np.concatenate(cells))
        act = self.plotter.add_mesh(pd, color=color, line_width=width, name=name, reset_camera=False,
                                    pickable=False, lighting=False, render_lines_as_tubes=False)
        try:
            mp = act.GetMapper()
            mp.SetResolveCoincidentTopologyToPolygonOffset()
            mp.SetRelativeCoincidentTopologyLineOffsetParameters(-2.0, -2.0)
        except Exception:
            pass
        act.SetVisibility(bool(visible))
        self._overlays[name] = act
        self.render()

    def set_origin_axes(self, on, length=20.0):
        for n in ("__ox", "__oy", "__oz"):
            self.remove_overlay(n, render=False)
        if on:
            for n, d, c in (("__ox", (1, 0, 0), "#e5484d"), ("__oy", (0, 1, 0), "#30a46c"),
                            ("__oz", (0, 0, 1), "#3e63dd")):
                self.set_lines(n, [np.array([[0, 0, 0], np.array(d, float) * length])], color=c, width=4)
        self.render()

    def set_points(self, name, P, color="#ffd60a", size=12.0):
        self.remove_overlay(name, render=False)
        P = np.atleast_2d(np.asarray(P, float))
        if len(P) == 0:
            self.render()
            return
        act = self.plotter.add_points(pv.PolyData(P), color=color, point_size=size, name=name,
                                      reset_camera=False, pickable=False, render_points_as_spheres=True)
        self._overlays[name] = act
        self.render()

    def set_plane_glyph(self, name, origin, normal, size, color="#4cc2ff", opacity=0.22):
        """Ebene als halbtransparentes Quadrat (z. B. Spiegelebene)."""
        self.remove_overlay(name, render=False)
        pl = pv.Plane(center=np.asarray(origin, float), direction=np.asarray(normal, float),
                      i_size=float(size), j_size=float(size), i_resolution=1, j_resolution=1)
        act = self.plotter.add_mesh(pl, color=color, opacity=opacity, name=name, reset_camera=False,
                                    pickable=False, lighting=False, show_edges=True, edge_color=color)
        self._overlays[name] = act
        self.render()

    def set_text(self, name, P, labels, color="white"):
        self.remove_overlay(name, render=False)
        if not len(labels):
            return
        act = self.plotter.add_point_labels(np.atleast_2d(np.asarray(P, float)), list(labels), font_size=14,
                                            text_color=color, point_size=1, shape_opacity=0.35,
                                            name=name, reset_camera=False, always_visible=True)
        self._overlays[name] = act
        self.render()

    def focus(self, point, distance=None, view_dir=None):
        """Kamera auf point richten. view_dir: Blickrichtung (Kamera → Punkt), sonst bleibt die
        aktuelle Richtung."""
        cam = self.plotter.camera
        p = np.asarray(point, float)
        fp = np.asarray(cam.focal_point)
        pos = np.asarray(cam.position)
        d = pos - fp
        if view_dir is not None:
            v = np.asarray(view_dir, float)
            if np.linalg.norm(v) > 1e-9:
                d = -v / np.linalg.norm(v) * max(np.linalg.norm(d), 1e-9)
        if distance is not None:
            d = d / max(np.linalg.norm(d), 1e-9) * distance
        cam.focal_point = tuple(p)
        cam.position = tuple(p + d)
        up = np.asarray(cam.up, float)
        dn = d / max(np.linalg.norm(d), 1e-9)
        if abs(float(up @ dn)) > 0.95:          # Blickrichtung fast parallel zur Hochachse
            alt = np.array([0.0, 0.0, 1.0]) if abs(dn[2]) < 0.9 else np.array([0.0, 1.0, 0.0])
            cam.up = tuple(alt - (alt @ dn) * dn)
        try:
            self.plotter.renderer.ResetCameraClippingRange()
        except Exception:
            pass
        self.render()

    def overlays_with_prefix(self, prefix):
        return [n for n in self._overlays if n.startswith(prefix)]

    def show_cell_scalars(self, V, F, scalars, clim, title=None, cmap="coolwarm"):
        title = title or tr("Abw. [mm]", "Dev. [mm]")
        self.hide_scalars(render=False)
        faces = np.hstack([np.full((len(F), 1), 3, dtype=np.int64), F]).ravel()
        pd = pv.PolyData(np.asarray(V, float), faces)
        pd.cell_data["dev"] = np.asarray(scalars, float)
        if isinstance(cmap, (list, tuple)):
            # Klassen (z. B. Herkunft der Flächen): feste Farben, keine Skala
            self._scalar_actor = self.plotter.add_mesh(
                pd, scalars="dev", cmap=list(cmap), clim=clim, n_colors=len(cmap), name="__scalars",
                reset_camera=False, smooth_shading=False, pickable=False, show_scalar_bar=False, specular=0.3)
            self.render()
            return
        self._scalar_actor = self.plotter.add_mesh(
            pd, scalars="dev", cmap=cmap, clim=clim, name="__scalars", reset_camera=False,
            smooth_shading=False,
            pickable=False, scalar_bar_args={"title": title, "vertical": True, "position_x": 0.84,
                                             "position_y": 0.1, "height": 0.6, "width": 0.08,
                                             "fmt": "%+.3f", "color": "white", "title_font_size": 14,
                                             "label_font_size": 12})
        self.render()

    def show_point_scalars(self, V, F, scalars, clim, title=None):
        title = title or tr("Abw. [mm]", "Dev. [mm]")
        self.hide_scalars(render=False)
        faces = np.hstack([np.full((len(F), 1), 3, dtype=np.int64), F]).ravel()
        pd = pv.PolyData(np.asarray(V, float), faces)
        pd.point_data["dev"] = np.asarray(scalars, float)
        self._scalar_actor = self.plotter.add_mesh(
            pd, scalars="dev", cmap="coolwarm", clim=clim, name="__scalars", reset_camera=False,
            smooth_shading=False,
            pickable=False, scalar_bar_args={"title": title, "vertical": True, "position_x": 0.84,
                                             "position_y": 0.1, "height": 0.6, "width": 0.08,
                                             "fmt": "%+.3f", "color": "white", "title_font_size": 14,
                                             "label_font_size": 12})
        self.render()

    def hide_scalars(self, render=True):
        if self._scalar_actor is not None:
            self.plotter.remove_actor(self._scalar_actor, render=False)
            self._scalar_actor = None
            try:
                self.plotter.remove_scalar_bar()
            except Exception:
                pass
        if render:
            self.render()

    # ------------------------------------------------------------------ Kamera
    def view(self, which):
        p = self.plotter
        {"iso": p.view_isometric, "+x": p.view_yz, "-x": lambda: p.view_yz(negative=True),
         "+y": p.view_xz, "-y": lambda: p.view_xz(negative=True), "+z": p.view_xy,
         "-z": lambda: p.view_xy(negative=True)}[which]()
        self.render()

    def reset_camera(self):
        self.plotter.reset_camera()
        self.render()

    def set_parallel(self, on):
        if on:
            self.plotter.enable_parallel_projection()
        else:
            self.plotter.disable_parallel_projection()
        self.render()

    # ------------------------------------------------------------------ Picking / Malen
    def pick(self, x, y):
        if self.actor is None:
            return None
        self.picker.Pick(x, y, 0, self.plotter.renderer)
        cid = self.picker.GetCellId()
        if cid < 0:
            return None
        return int(cid), np.array(self.picker.GetPickPosition())

    def set_tool(self, tool):
        self.tool = tool
        if tool in ("brush", "erase"):
            self._ensure_cursor()
        elif self._cursor is not None:
            self._cursor.SetVisibility(False)
            self.render()

    def _ensure_cursor(self):
        if self._cursor is None:
            sph = pv.Sphere(radius=1.0, theta_resolution=24, phi_resolution=16)
            self._cursor = self.plotter.add_mesh(sph, color="#ffd24a", opacity=0.25, pickable=False,
                                                 name="__cursor", reset_camera=False, lighting=False)
            self._cursor.SetVisibility(False)
        self._cursor.SetScale(self.brush_radius, self.brush_radius, self.brush_radius)

    def set_brush_radius(self, r):
        self.brush_radius = float(r)
        if self._cursor is not None:
            self._cursor.SetScale(r, r, r)
            self.render()

    def _hover(self, x, y):
        if self.tool in ("brush", "erase") and self._cursor is not None:
            hit = self.pick(x, y)
            if hit is None:
                if self._cursor.GetVisibility():
                    self._cursor.SetVisibility(False)
                    self.render(throttle=True)
                return
            self._cursor.SetPosition(*hit[1])
            self._cursor.SetVisibility(True)
            self.render(throttle=True)

    def _stroke_begin(self, x, y, erase):
        hit = self.pick(x, y)
        if self.tool in ("fill", "pick_feature", "seed", "hole", "bore", "measure"):
            if hit is not None:
                self.picked.emit({"cell": hit[0], "pos": hit[1], "erase": erase, "tool": self.tool})
            return
        self._erase = erase
        if hit is not None:
            self.stroke.emit({"event": "begin", "cell": hit[0], "pos": hit[1], "erase": erase})

    def _stroke_move(self, x, y):
        if self.tool not in ("brush", "erase"):
            return
        hit = self.pick(x, y)
        if hit is not None:
            if self._cursor is not None:
                self._cursor.SetPosition(*hit[1])
                self._cursor.SetVisibility(True)
            self.stroke.emit({"event": "move", "cell": hit[0], "pos": hit[1], "erase": self._erase})

    def _stroke_end(self):
        if self.tool in ("brush", "erase"):
            self.stroke.emit({"event": "end"})

    def close(self):
        try:
            self.plotter.close()
        except Exception:
            pass
