"""Zusammenbau des Gesamtmodells aus Features und STEP-Export."""
from __future__ import annotations

import json

import numpy as np

from . import occ_build as ob
from .features import to_jsonable
from .i18n import tr


def _surface_on_body(surf_shape, body_tess, thr):
    """True, wenn die Fläche (fast) vollständig auf der Körperoberfläche liegt."""
    import pyvista as pv
    from .preprocess import vf_to_pd
    V, F = ob.tessellate(surf_shape, max(thr, 0.05))
    if len(V) == 0:
        return False
    tgt = vf_to_pd(*body_tess)
    d = np.abs(np.asarray(pv.PolyData(V).compute_implicit_distance(tgt).point_data["implicit_distance"]))
    return float(np.percentile(d, 90)) < thr


def assemble(features, apply_booleans=True, include_surfaces=True, sew_tol=0.0,
             rest_mesh=None, drop_redundant=True, trim=False, scan_model=None, trim_ext=1.5,
             keep_dist=None, tol=0.05, filter_body=True, log=None, blends=None, fillet_min_r=None):
    """Gibt Liste (Name, Shape, Farbe) zurück.

    Rollen:  add -> vereinigt, sub -> abgezogen, cut -> Halbraum abgezogen,
             surface -> einzelne Flächen (optional vernäht oder gegeneinander verschnitten).
    Reihenfolge der Liste = Reihenfolge der Booleschen Operationen.
    """
    log = log or (lambda s: None)
    items = []
    body = None
    adds, ops, surfs = [], [], []
    for f in features:
        try:
            shp = f.shape()
        except Exception as e:  # noqa: BLE001
            log(f"⚠ {f.name}: Geometrie fehlgeschlagen ({e}) – übersprungen")
            continue
        if f.role == "add":
            adds.append((f, shp))
        elif f.role in ("sub", "cut"):
            ops.append((f, shp))
        else:
            surfs.append((f, shp))
    if apply_booleans and (adds or ops):
        if adds:
            body = adds[0][1]
            for f, s in adds[1:]:
                try:
                    body = ob.fuse_all([body, s], fuzzy=max(0.5 * tol, 1e-4))
                except Exception as e:  # noqa: BLE001
                    log(f"⚠ Vereinigung mit {f.name} fehlgeschlagen: {e}")
            for f, s in ops:
                try:
                    body = ob.cut(body, s, fuzzy=max(0.5 * tol, 1e-4))
                except Exception as e:  # noqa: BLE001
                    log(f"⚠ Abziehen von {f.name} fehlgeschlagen: {e}")
            body = ob.fix_shape(body)
            log("Körper " + ("gültig" if ob.is_valid(body) else "NICHT gültig (bitte prüfen)"))
        else:
            log("Nur abziehende Features ohne Grundkörper – exportiert als einzelne Körper.")
            for f, s in ops:
                items.append((f.name, s, f.color))
    elif adds or ops:
        for f, s in adds + ops:
            items.append((f.name, s, f.color))

    if not include_surfaces:
        surfs = []
    # Flächen, die auf dem Körper liegen (z. B. erkannte Ebenen unter einer Extrusion), weglassen
    if body is not None and surfs and drop_redundant:
        thr = max(2.5 * tol, 0.05)
        try:
            bt = ob.tessellate(body, max(tol, 0.02))
            kept = []
            for f, s in surfs:
                if f.kind != "freeform" and _surface_on_body(s, bt, thr):
                    log(f"{f.name}: liegt auf dem Körper – weggelassen")
                else:
                    kept.append((f, s))
            surfs = kept
        except Exception as e:  # noqa: BLE001
            log(f"⚠ Redundanzprüfung übersprungen: {e}")

    if trim and scan_model is not None and (surfs or body is not None) and (surfs and len(surfs) + (body is not None) >= 2):
        from . import trim as tr
        from .features import ROLES
        args = []
        edge = scan_model.mean_edge_length()
        try:
            grown = tr.grow_regions(scan_model, [f for f, _ in surfs], d_max=max(6 * tol, 2 * edge))
        except Exception as e:  # noqa: BLE001
            log(f"⚠ Regionen-Wachstum fehlgeschlagen ({e})")
            grown = [np.asarray(f.face_ids) for f, _ in surfs]
        rings = max(1, int(round(trim_ext / max(edge, 1e-6))))
        for (f, s), ids in zip(surfs, grown):
            try:
                if len(ids) >= 3:
                    es = tr.surface_for_trimming(f, scan_model, ids, rings, tol, trim_ext)
                else:
                    es = ob.build_shape(f.kind, tr.extended_params(f.kind, f.params, trim_ext), "surface")
            except Exception as e:  # noqa: BLE001
                log(f"⚠ {f.name}: Verlängerung fehlgeschlagen ({e}) – unverlängert")
                es = s
            args.append((f.name, es, f.color, False))
        if body is not None:
            args.append(("Koerper", body, (0.75, 0.78, 0.82), True))
        kd = keep_dist if keep_dist else max(4 * tol, 0.15)
        try:
            res = tr.split_and_select(args, scan_model, kd, fuzzy=max(tol * 0.2, 1e-4),
                                      body_support=0.3 if filter_body else -1.0, log=log)
            all_faces = [f for _, faces, _, _ in res for f in faces]
            merged = tr.make_shell_or_solid(all_faces, max(sew_tol, 2 * tol, 0.05), log=log)
            if merged is not None:
                items.append(("Modell_verschnitten", merged, (0.75, 0.78, 0.82)))
            body = None
            surfs = []
        except Exception as e:  # noqa: BLE001
            log(f"⚠ Verschneiden fehlgeschlagen ({e}) – Export ohne Verschneidung")

    if body is not None:
        items.append(("Koerper", body, (0.75, 0.78, 0.82)))
    if surfs:
        if sew_tol and sew_tol > 0 and len(surfs) > 1:
            try:
                sewn = ob.sew([s for _, s in surfs], sew_tol)
                items.append(("Flaechen_vernaeht", sewn, (0.55, 0.7, 0.9)))
                log(f"{len(surfs)} Flächen vernäht (Toleranz {sew_tol} mm)")
            except Exception as e:  # noqa: BLE001
                log(f"⚠ Nähen fehlgeschlagen: {e}")
                items += [(f.name, s, f.color) for f, s in surfs]
        else:
            items += [(f.name, s, f.color) for f, s in surfs]
    if rest_mesh is not None and len(rest_mesh[1]):
        V, F = rest_mesh
        log(f"Restnetz facettiert: {len(F)} Dreiecke")
        items.append(("Restnetz", ob.mesh_to_faceted(V, F, make_solid=False), (0.6, 0.6, 0.6)))
    return items


def plug_solid_features(model, features, tol=0.05, log=None):
    """Kopie des Netzes, in der die Dreiecke von Körper-Features (Bohrungen, Zapfen, …) durch
    ebene Deckel ersetzt sind ('verstopft'). Die Körper werden danach exakt per Boolean
    angebracht. Rückgabe: (Netz, face_map alt->neu (-1 = entfernt))."""
    from .mesh_ops import delete_and_fill
    log = log or (lambda s: None)
    nF = model.n_faces
    mask = np.zeros(nF, bool)
    for f in features:
        if f.role in ("add", "sub") and len(f.face_ids):
            ids = np.asarray(f.face_ids, np.int64)
            mask[ids[(ids >= 0) & (ids < nF)]] = True
    if not mask.any():
        return model, np.arange(nF), np.zeros(0, np.int64)
    m2, fmap, nloops = delete_and_fill(model, np.nonzero(mask)[0], mode="flat")
    n_kept = int((fmap >= 0).sum())
    caps = np.arange(n_kept, m2.n_faces)
    log(tr(f"{int(mask.sum())} Dreiecke von Körper-Features durch {nloops} ebene Deckel ersetzt",
           f"{int(mask.sum())} triangles of solid features replaced by {nloops} flat caps"))
    return m2, fmap, caps


def apply_blends(body, blends, features, min_r=0.0, tol=0.05, log=None):
    """Erkannte Kantenrundungen (blends.detect_edge_blends) als echte Verrundung auf die Kanten
    zwischen den beiden Ebenen legen. Rückgabe: (Körper, Anzahl verrundeter Kanten)."""
    from OCP.BRepAdaptor import BRepAdaptor_Surface
    from OCP.BRepFilletAPI import BRepFilletAPI_MakeFillet
    from OCP.GeomAbs import GeomAbs_Plane
    from .brep_seg import _edge_face_map
    from .geom_utils import unit
    from .occ_compat import to_edge, to_face
    log = log or (lambda s: None)
    planes = {f.id: f for f in features if f.kind == "plane"}
    want = [b for b in blends or () if b["r"] >= min_r and all(int(i) in planes for i in b["planes"])]
    if not want:
        return body, 0
    mp = _edge_face_map(body)

    def plane_of(face):
        ad = BRepAdaptor_Surface(face)
        if ad.GetType() != GeomAbs_Plane:
            return None
        pl = ad.Plane()
        ax = pl.Axis()
        n = np.array([ax.Direction().X(), ax.Direction().Y(), ax.Direction().Z()])
        o = np.array([pl.Location().X(), pl.Location().Y(), pl.Location().Z()])
        return n, o

    def matches(pd, f):
        if pd is None:
            return False
        n, o = pd
        nf = unit(f.params["normal"])
        if abs(float(n @ nf)) < math.cos(math.radians(2.0)):
            return False
        return abs(float((o - np.asarray(f.params["origin"], float)) @ nf)) < max(3 * tol, 0.1)

    import math
    sel = []
    for i in range(1, mp.Extent() + 1):
        fl = mp.FindFromIndex(i)
        if fl.Size() != 2:
            continue
        fa, fb = [to_face(x) for x in fl]
        pa, pb = plane_of(fa), plane_of(fb)
        if pa is None or pb is None:
            continue
        for b in want:
            p1, p2 = planes[int(b["planes"][0])], planes[int(b["planes"][1])]
            if (matches(pa, p1) and matches(pb, p2)) or (matches(pa, p2) and matches(pb, p1)):
                sel.append((to_edge(mp.FindKey(i)), float(b["r"])))
                break
    if not sel:
        return body, 0

    def run(items):
        mk = BRepFilletAPI_MakeFillet(body)
        for e, r in items:
            mk.Add(r, e)
        mk.Build()
        if not mk.IsDone():
            return None
        out = ob.fix_shape(mk.Shape())
        return out if ob.is_valid(out) else None

    res = run(sel)
    if res is not None:
        return res, len(sel)
    # einzeln versuchen, problematische Kanten auslassen
    n_ok = 0
    for e, r in sel:
        try:
            mk = BRepFilletAPI_MakeFillet(body)
            mk.Add(r, e)
            mk.Build()
            if mk.IsDone():
                cand = ob.fix_shape(mk.Shape())
                if ob.is_valid(cand):
                    body = cand
                    n_ok += 1
        except Exception:  # noqa: BLE001
            continue
    return body, n_ok


def extend_open_ends(f, body, tol=0.05):
    """Werkzeug eines abzuziehenden Körper-Features an offenen Enden verlängern, bis es dort ganz
    aus dem Körper herausragt. Eine Bohrung, deren Achse schräg auf die Ein-/Austrittsfläche
    trifft, endet sonst mit einem Deckel senkrecht zur Achse, und neben der Achse bleibt ein Keil
    stehen (Länge ≈ r·tan(Neigung)). Geprüft wird ein Ring von Punkten in der Deckelebene: liegt
    ein Teil davon im Körper (Keil), wird das Ende schrittweise verlängert, bis keiner mehr drin
    liegt. Ganz im Körper liegende Enden (Bohrungsgrund) bleiben unverändert.
    Rückgabe: (Werkzeug, Liste (Ende, Verlängerung in mm)) oder (None, []), wenn nichts zu tun ist."""
    import copy
    import math
    from OCP.BRepClass3d import BRepClass3d_SolidClassifier
    from OCP.TopAbs import TopAbs_IN
    from .bores import build_hole, hole_profile
    from .geom_utils import orthonormal_frame, unit
    p = copy.deepcopy(f.params)
    ends = []
    if f.kind == "hole":
        o = np.asarray(p["point"], float)
        a = unit(p["axis"])
        prof = hole_profile(p)
        z_top = float(prof[0][1])
        r_top = max(float(prof[1][0]), 0.5 * float(p["d"]))
        ends.append(("entry", z_top, -1.0, r_top))
        if p.get("bottom") == "through":
            ends.append(("exit", float(prof[-1][1]), 1.0, 0.5 * float(p["d"])))
    elif f.kind == "cylinder":
        o = np.asarray(p["point"], float)
        a = unit(p["axis"])
        R = float(p["radius"])
        ends += [("v0", float(p["v0"]), -1.0, R), ("v1", float(p["v1"]), 1.0, R)]
    else:
        return None, []
    x, y, _ = orthonormal_frame(a)
    cl = BRepClass3d_SolidClassifier(body)

    def inside_frac(z, R):
        ang = np.linspace(0, 2 * math.pi, 24, endpoint=False)
        ring = [(0.97 * R, t) for t in ang] + [(0.6 * R, t) for t in ang[::2]]
        n_in = 0
        for rr, t in ring:
            q = o + z * a + rr * (math.cos(t) * x + math.sin(t) * y)
            cl.Perform(ob.P(q), max(tol, 1e-4))
            if cl.State() == TopAbs_IN:
                n_in += 1
        return n_in / len(ring)

    done = []
    for name, z0, sgn, R in ends:
        f0 = inside_frac(z0, R)
        if f0 == 0.0 or f0 > 0.6:
            continue                  # frei bzw. Bohrungsgrund im Material
        step = max(0.15, 0.08 * R)
        ext = None
        for k in range(1, int(math.ceil(max(2.0, 2.0 * R + 1.0) / step)) + 1):
            if inside_frac(z0 + sgn * k * step, R) == 0.0:
                ext = k * step + max(0.3, 2 * tol)
                break
        if ext is None:
            continue                  # Überhang/Wand: nicht durchbohren
        done.append((name, ext))
        if name == "entry":
            p["ext_entry"] = -z0 + ext
        elif name == "exit":
            p["ext_exit"] = max(1.0, z0 - float(p["depth"])) + ext
        elif name == "v0":
            p["v0"] = float(p["v0"]) - ext
        else:
            p["v1"] = float(p["v1"]) + ext
    if not done:
        return None, []
    tool = build_hole(p) if f.kind == "hole" else ob.build_shape("cylinder", p, "sub")
    return tool, done

def build_scan_topology(model, features, tol=0.05, log=None, progress=None, blends=None, fillet_min_r=None):
    """Exakt geschlossener Volumenkörper aus der Scan-Topologie (siehe brep_seg) plus
    Boolesche Körper-Features. Rückgabe dict(items, solid, report, open_edges, stats, aux)."""
    import copy
    from .brep_seg import build_closed_brep
    report = []
    lg = (lambda s: (report.append(s), log and log(s)))
    plugged, fmap, caps = plug_solid_features(model, features, tol, lg)
    surf = []
    for f in features:
        if f.role == "surface" and f.kind in ("plane", "cylinder", "cone", "sphere", "torus", "freeform", "ellipsoid"):
            g = copy.copy(f)
            ids = np.asarray(f.face_ids, np.int64)
            ids = ids[(ids >= 0) & (ids < len(fmap))]
            ids = fmap[ids]
            g.face_ids = ids[ids >= 0]
            surf.append(g)
    res = build_closed_brep(plugged, surf, tol=tol, log=lg, progress=progress, passive=caps)
    shape = res["shape"]
    solid = res["solid"]
    adds = [f for f in features if f.role == "add"]
    subs = [f for f in features if f.role in ("sub", "cut")]
    items = []

    def _fillet(body):
        if not blends or fillet_min_r is None:
            return body
        try:
            body2, nf = apply_blends(body, blends, features, fillet_min_r, tol, lg)
            if nf:
                lg(tr(f"{nf} Kanten verrundet (erkannte Kantenrundungen ab R {fillet_min_r:g} mm)",
                      f"{nf} edges filleted (detected edge blends from R {fillet_min_r:g} mm)"))
                return body2
        except Exception as e:  # noqa: BLE001
            lg(tr(f"⚠ Verrunden fehlgeschlagen: {e}", f"⚠ filleting failed: {e}"))
        return body

    if solid and not (adds or subs):
        shape = _fillet(shape)
        from .brep_seg import _n_faces as _nf
        if _nf(shape) > 3000 and not ob.is_valid(shape):     # kleinere hat brep_seg schon geprüft
            lg(tr("⚠ Körper geschlossen, die Formprüfung (BRepCheck) meldet aber Fehler – im CAD ggf. reparieren",
                  "⚠ solid closed, but the shape check (BRepCheck) reports errors – repair in CAD if needed"))
    if solid and (adds or subs):
        from .brep_seg import _volume as _vol
        body = shape
        n_app = 0
        vcache = {}
        if (res.get("stats") or {}).get("volume"):
            vcache[id(shape)] = (shape, float(res["stats"]["volume"]))

        def vol_of(x):
            k = id(x)
            if k not in vcache:
                vcache[k] = (x, _vol(x))       # Form mit ablegen, damit die id gültig bleibt
            return vcache[k][1]

        from .brep_seg import _free_edges as _fe

        def plausible(before, after, tool, sign):
            # Ergebnis muss zwischen 'unverändert' und 'ganzes Werkzeug' liegen (sonst ist das
            # Boolean an einer fehlerhaften Fläche entgleist) und darf keine offenen Kanten haben
            try:
                vb, va, vt = vol_of(before), vol_of(after), abs(_vol(tool))
            except Exception:
                return True
            if va <= 0:
                return False
            try:
                if _fe(after) > 0 and _fe(before) == 0:
                    return False
            except Exception:  # noqa: BLE001
                pass
            if sign < 0:
                return vb - 1.5 * vt - 1e-6 <= va <= vb + 1e-3 * max(vb, 1.0)
            return vb - 1e-3 * max(vb, 1.0) <= va <= vb + 1.5 * vt + 1e-6

        for f in adds:
            try:
                t = f.shape()
                nb = ob.fuse_all([body, t], fuzzy=max(0.5 * tol, 1e-4))
                if plausible(body, nb, t, +1):
                    body = nb
                    n_app += 1
                else:
                    lg(tr(f"⚠ Vereinigung mit {f.name} unplausibel – ausgelassen", f"⚠ fusing {f.name} implausible – skipped"))
            except Exception as e:  # noqa: BLE001
                lg(tr(f"⚠ Vereinigung mit {f.name} fehlgeschlagen: {e}", f"⚠ fusing {f.name} failed: {e}"))
        for f in subs:
            try:
                # Bohrungen stehen über die Eintrittsfläche hinaus -> kein unscharfes Boolean nötig
                t = f.shape()
                try:
                    t2, ext = extend_open_ends(f, body, tol)
                except Exception:  # noqa: BLE001
                    t2, ext = None, []
                if t2 is not None:
                    t = t2
                    lg(tr(f"{f.name}: Werkzeug am schräg angeschnittenen Ende verlängert ("
                          + ", ".join(f"{e:.2f} mm" for _, e in ext) + ")",
                          f"{f.name}: tool extended at the obliquely cut end ("
                          + ", ".join(f"{e:.2f} mm" for _, e in ext) + ")"))
                nb = ob.cut(body, t, fuzzy=0.0 if f.kind == "hole" else max(0.5 * tol, 1e-4))
                if plausible(body, nb, t, -1):
                    body = nb
                    n_app += 1
                else:
                    lg(tr(f"⚠ Abziehen von {f.name} unplausibel – ausgelassen", f"⚠ subtracting {f.name} implausible – skipped"))
            except Exception as e:  # noqa: BLE001
                lg(tr(f"⚠ Abziehen von {f.name} fehlgeschlagen: {e}", f"⚠ subtracting {f.name} failed: {e}"))
        from .brep_seg import _free_edges, _n_faces, _volume
        if _n_faces(body) <= 4000:
            body = ob.fix_shape(body)      # ShapeFix_Shape wächst bei facettierten Körpern quadratisch
        ok = ob.is_valid(body)
        vfin = _volume(body)
        try:
            if vfin < 0:
                body = body.Reversed()
                vfin = -vfin
        except Exception:
            pass
        if not ok:
            try:
                if vfin > 0 and _free_edges(body) == 0:
                    ok = True
                    lg(tr("⚠ Körper geschlossen, die Formprüfung (BRepCheck) meldet aber Fehler",
                          "⚠ solid closed, but the shape check (BRepCheck) reports errors"))
            except Exception:
                pass
        if ok:
            body = _fillet(body)
        if ok:
            if fillet_min_r is not None and blends:
                vfin = _volume(body)
            lg(tr(f"✔ Mit {n_app} Körper-Features: Volumen {vfin:.1f} mm³",
                  f"✔ with {n_app} solid features: volume {vfin:.1f} mm³"))
        else:
            lg(tr("⚠ Ergebnis der Booleschen Operationen ungültig – bitte prüfen",
                  "⚠ result of boolean operations invalid – please check"))
        items.append((tr("Bauteil", "Part"), body, (0.75, 0.78, 0.82)))
        solid = ok
    else:
        items.append((tr("Bauteil", "Part") if solid else tr("Huelle_offen", "Shell_open"), shape, (0.75, 0.78, 0.82)))
        if not solid:
            for f in adds + subs:
                try:
                    items.append((f.name, f.shape(), f.color))
                except Exception:
                    pass
    aux = [f for f in res["cfeat"] if f.params.get("aux")]
    return {"items": items, "solid": solid, "report": report, "open_edges": res["open_edges"],
            "stats": res["stats"], "aux": aux, "plugged": plugged, "labels": res["labels"]}


def export_step(path, features, schema="AP214IS", mode="bool", model=None, tol=0.05, **kw):
    log_lines = []
    if mode == "scan":
        if model is None:
            raise RuntimeError(tr("Scan-Topologie braucht das Netz.", "Scan topology needs the mesh."))
        items = build_scan_topology(model, features, tol=tol, log=log_lines.append, blends=kw.get("blends"),
                                    fillet_min_r=kw.get("fillet_min_r"))["items"]
    else:
        items = assemble(features, log=log_lines.append, trim=(mode == "trim"), scan_model=model, tol=tol, **kw)
    if not items:
        raise RuntimeError("Nichts zu exportieren.")
    named = ob.write_step(items, path, schema)
    log_lines.append(f"STEP geschrieben: {path}" + (" (mit Namen/Farben)" if named else ""))
    return log_lines


def export_report(path, features):
    data = [dict(f.to_dict(), n_faces=int(len(f.face_ids))) for f in features]
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(to_jsonable(data), fh, indent=2, ensure_ascii=False)
