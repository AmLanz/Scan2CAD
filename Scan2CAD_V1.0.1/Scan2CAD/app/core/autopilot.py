"""'Auf gut Glück': der ganze Weg vom Scan zum geschlossenen STEP in einzelnen Schritten.

Jeder Schritt bekommt ein Dokument (Kopie), verändert es und liefert einen kurzen Bericht.
Die GUI führt die Schritte einzeln aus, zeigt nach jedem Schritt das Zwischenergebnis und
wartet (wenn gewünscht) – man kann jederzeit eingreifen (Features löschen, Löcher anders füllen,
Bohrungen korrigieren …) und dann weitermachen; der nächste Schritt arbeitet mit dem
aktuellen Stand.
"""
from __future__ import annotations

import math

import numpy as np

from .geom_utils import unit
from .i18n import tr

STEPS = [
    ("repair", ("Netz reparieren", "Repair mesh")),
    ("noise", ("Rauschen messen → Toleranzen", "Measure noise → tolerances")),
    ("small_holes", ("Kleine Löcher füllen", "Fill small holes")),
    ("base", ("Offenen Boden schließen, Rest füllen", "Close open base, fill the rest")),
    ("simplify", ("Vereinfachen", "Simplify")),
    ("rough_align", ("Grob ausrichten", "Rough alignment")),
    ("detect", ("Formen erkennen (RANSAC)", "Detect shapes (RANSAC)")),
    ("merge", ("Ähnliche zusammenführen", "Merge similar")),
    ("optimize_align", ("Ausrichtung optimieren, Achsen gerade", "Optimise alignment, straighten axes")),
    ("rotational", ("Drehteil? Rotationsflächen aus dem Profil", "Turned part? Surfaces of revolution from the profile")),
    ("holes", ("Bohrungen erkennen, Werte glätten", "Detect holes, smooth values")),
    ("ports", ("Bohrungsumgebung: Ringnuten, Fasen, Achsen (Drehprofil)",
               "Hole surroundings: ring grooves, chamfers, axes (turned profile)")),
    ("cleanup", ("Kantenrundungen und Splitter → scharfe Kanten", "Edge blends and fragments → sharp edges")),
    ("symmetry", ("Symmetrien suchen", "Find symmetries")),
    ("freeform", ("Restflächen → Freiform", "Remaining areas → freeform")),
    ("preview", ("STEP-Vorschau (Scan-Topologie), Dichtheit prüfen", "STEP preview (scan topology), check closure")),
]
STEP_KEYS = [k for k, _ in STEPS]

DEFAULTS = {
    "small_hole_frac": 0.03,       # Löcher bis 3 % der Bauteilgröße
    "base_min_frac": 0.25,         # Rand gilt als offener Boden ab 25 % des Bauteilumfangs
    "max_faces": 150000,
    "max_faces_initial": 600000,
    "detect_min_area": 0.002,
    "detect_max": 80,
    "capture_deg": 6.0,
    "straighten_deg": 1.0,
    "freeform_min_frac": 0.01,
    "cad_target_faces": 80000,
    "rot_min_frac": 0.9,
    "rot_min_area": 0.0003,
    "blend_r_max": 1.0,
}


def step_title(key):
    for k, (de, en) in STEPS:
        if k == key:
            return tr(de, en)
    return key


def run_step(key, doc, opts=None, progress=None):
    """Schritt auf doc (Kopie!) ausführen. Rückgabe: Liste von Berichtszeilen."""
    o = dict(DEFAULTS)
    o.update(opts or {})
    prog = progress or (lambda s: None)
    return globals()["_step_" + key](doc, o, prog)


# ===================================================================== Schritte
def _step_repair(doc, o, prog):
    from . import preprocess as pp
    from .mesh_ops import manifold_report, repair_manifold
    out = []
    if doc.model.n_faces > o["max_faces_initial"] and not doc.model.exact:
        # sehr fein aufgelöste Scans gleich zu Beginn ausdünnen: die Auflösung liegt weit unter der
        # Fit-Toleranz, kostet aber in jedem Schritt Zeit und Speicher (und macht die Ansicht träge)
        n0 = doc.model.n_faces
        m0 = pp.decimate_to_count(doc.model, o["max_faces_initial"])
        doc.replace_model(m0)
        out.append(tr(f"sehr feines Netz: {n0:,} → {m0.n_faces:,} Dreiecke ausgedünnt".replace(",", "."),
                      f"very fine mesh: thinned {n0:,} → {m0.n_faces:,} triangles"))
    m, rep = pp.repair(doc.model, merge_tol=0.0, min_component_frac=0.01)
    doc.replace_model(m)
    mr = manifold_report(doc.model)
    out += list(rep)
    if mr["nonmanifold_edges"] or mr["flipped_edges"]:
        m2, rep2, fmap = repair_manifold(doc.model)
        doc.replace_model(m2, face_map=fmap)
        out += rep2
    mr = manifold_report(doc.model)
    out.append(tr(f"{mr['components']} Teil(e), {mr['boundary_edges']} offene Randkanten",
                  f"{mr['components']} part(s), {mr['boundary_edges']} open boundary edges"))
    return out


def _step_noise(doc, o, prog):
    from .mesh_ops import estimate_noise
    from .preprocess import mesh_kind, subdivide_long_edges
    m = doc.model
    kind = mesh_kind(m)
    out = []
    if kind["kind"] == "cad":
        # CAD-Export: Punkte liegen exakt auf der Fläche, Dreiecke sind sehr ungleich groß.
        # Gleichmäßig unterteilen (ebene Facetten bleiben exakt eben), Toleranzen aus dem
        # Facettenfehler gekrümmter Flächen statt aus Rauschen.
        A = float(m.areas.sum())
        h = 1.5 * np.sqrt(4 * A / (np.sqrt(3) * o["cad_target_faces"]))
        if kind["max_edge"] > 2.5 * h:
            m2 = subdivide_long_edges(m, h, max_faces=int(2.5 * o["cad_target_faces"]), only_coplanar=True)
            doc.replace_model(m2)
            out.append(tr(f"CAD-Export erkannt ({100 * kind['coplanar']:.0f} % exakt ebene Nachbarn) – gleichmäßig "
                          f"unterteilt: {m.n_faces:,} → {m2.n_faces:,} Dreiecke".replace(",", "."),
                          f"CAD export detected ({100 * kind['coplanar']:.0f} % exactly coplanar neighbours) – "
                          f"subdivided evenly: {m.n_faces:,} → {m2.n_faces:,} triangles"))
        else:
            out.append(tr(f"CAD-Export erkannt ({100 * kind['coplanar']:.0f} % exakt ebene Nachbarn)",
                          f"CAD export detected ({100 * kind['coplanar']:.0f} % exactly coplanar neighbours)"))
        sag = max(float(kind["sagitta"]), 1e-4)
        diag = doc.model.diag
        # Eckpunkte liegen exakt auf den Flächen → Erkennung über die Eckpunkte mit enger Toleranz;
        # die Fit-Toleranz (u. a. für Schwerpunkte gekrümmter Facetten) berücksichtigt den Facettenfehler
        ns = {"sigma": sag / 2, "edge": doc.model.mean_edge_length(), "kind": "cad", "sagitta": sag,
              "fit_tol": round(max(0.75 * sag, 0.01), 4), "detect_eps": round(max(2e-4 * diag, 0.02), 4),
              "simplify_tol": round(max(sag, 0.01), 4), "p90": sag}
        doc.tol = ns
        out.append(tr(f"Facettenfehler gekrümmter Flächen ≈ {sag:.4f} mm (Eckpunkte exakt) → Fit-Toleranz "
                      f"{ns['fit_tol']:.3f}, Erkennung {ns['detect_eps']:.3f} mm",
                      f"facet error of curved faces ≈ {sag:.4f} mm (vertices exact) → fit tolerance "
                      f"{ns['fit_tol']:.3f}, detection {ns['detect_eps']:.3f} mm"))
        return out
    ns = estimate_noise(m)
    ns["kind"] = "scan"
    doc.tol = ns
    return [tr(f"Rauschen σ ≈ {ns['sigma']:.4f} mm → Fit-Toleranz {ns['fit_tol']:.3f}, Erkennung {ns['detect_eps']:.3f}, "
               f"Vereinfachen {ns['simplify_tol']:.3f} mm",
               f"noise σ ≈ {ns['sigma']:.4f} mm → fit tolerance {ns['fit_tol']:.3f}, detection {ns['detect_eps']:.3f}, "
               f"simplify {ns['simplify_tol']:.3f} mm")]


def _size(model):
    mn, mx = model.bbox
    return float(np.linalg.norm(mx - mn))


def _step_small_holes(doc, o, prog):
    from .holes import fill_hole, find_holes
    m = doc.model
    lim = o["small_hole_frac"] * _size(m)
    holes = [h for h in find_holes(m) if h["diameter"] <= lim]
    mode = "flat" if doc.tol.get("kind") == "cad" else "smooth"
    n = 0
    for h in holes:
        try:
            m, _ = fill_hole(m, h["loop"], mode)
            n += 1
        except Exception:
            continue
    doc.replace_model(m)
    return [tr(f"{n} kleine Löcher (Ø ≤ {lim:.1f} mm) glatt gefüllt – bearbeitbar unter Scan › Füllungen",
               f"{n} small holes (Ø ≤ {lim:.1f} mm) filled smoothly – editable under Scan › Fills")]


def _step_base(doc, o, prog):
    from .features import fixed_plane_feature
    from .holes import fill_hole, find_holes
    from .mesh_ops import close_base, plane_from_rims, rim_info, rims_near_plane
    m = doc.model
    out = []
    holes = find_holes(m)
    if not holes:
        return [tr("Netz ist geschlossen.", "Mesh is closed.")]
    mn, mx = m.bbox
    ext = np.sort(mx - mn)
    girth = 2 * (ext[1] + ext[2])               # Umfang der beiden größten Ausdehnungen
    big = max(holes, key=lambda h: h["perimeter"])
    info = rim_info(m, big["loop"])
    cad = doc.tol.get("kind") == "cad"
    if (not cad and big["perimeter"] >= o["base_min_frac"] * girth and info["flatness"] < 0.05 * _size(m)
            and big["n"] >= 12):
        # Ebene am äußersten Randpunkt: alle Wände werden (tangential) verlängert, keiner wird gekappt.
        # Die wahre Lage des Bodens kennt der Scan nicht -> per Bedingung (Abstand) festlegen.
        plane = plane_from_rims(m, [big["loop"]], "lowest")
        others = [h for h in holes if h is not big]
        inner = rims_near_plane(m, others, plane, max_dist=max(1.0, 0.02 * _size(m)))
        m, s_ids, c_ids, rep = close_base(m, [big["loop"]] + inner, plane, extend=True)
        out += rep
        f = fixed_plane_feature(m, c_ids, plane, name=tr("Boden", "Floor"), tol=doc.fit_tol)
        m.fills[-1]["feature"] = f.id
        doc.replace_model(m)
        doc.features.append(f)
        out.append(tr("Boden als festgelegte Ebene am tiefsten Randpunkt angelegt – die echte Lage kennt der Scan "
                      "nicht: per Bedingung (z. B. Abstand zur Deckfläche) festlegen oder unter Scan › Löcher & Boden "
                      "neu wählen",
                      "floor created as fixed plane at the lowest rim point – the scan does not know its true position: "
                      "set it by a constraint (e.g. distance to the top face) or re-choose it under Scan › Holes & base"))
        m = doc.model
    rest = find_holes(m)
    n = 0
    for h in sorted(rest, key=lambda h: h["perimeter"]):
        try:
            # CAD-Export: fehlende Dreiecke liegen in ebenen Facetten → eben füllen
            m, _ = fill_hole(m, h["loop"], "flat" if (cad or h["flatness"] < 1e-6) else "smooth")
            n += 1
        except Exception:
            continue
    doc.replace_model(m)
    if n:
        out.append(tr(f"{n} weitere Löcher glatt gefüllt (prüfen: Scan › Füllungen – neu füllen/als Ebene erzwingen)",
                      f"{n} further holes filled smoothly (check: Scan › Fills – refill/force plane)"))
    if m.boundary_edge_count:
        out.append(tr(f"⚠ {m.boundary_edge_count} offene Randkanten bleiben", f"⚠ {m.boundary_edge_count} open edges remain"))
    return out


def _step_simplify(doc, o, prog):
    from . import preprocess as pp
    m = doc.model
    if m.n_faces <= o["max_faces"]:
        return [tr(f"{m.n_faces:,} Dreiecke – Vereinfachen nicht nötig".replace(",", "."),
                   f"{m.n_faces:,} triangles – no simplification needed")]
    tol = float(doc.tol.get("simplify_tol", 0.05))
    m2, info = pp.simplify_to_tolerance(m, tol, max_faces=o["max_faces"])
    if m2.n_faces > o["max_faces"]:
        m2 = pp.decimate_to_count(m, o["max_faces"])
    doc.replace_model(m2)
    return [tr(f"{m.n_faces:,} → {m2.n_faces:,} Dreiecke (Abweichung P99.9 {info.get('dev_max', 0):.4f} mm)".replace(",", "."),
               f"{m.n_faces:,} → {m2.n_faces:,} triangles (deviation P99.9 {info.get('dev_max', 0):.4f} mm)")]


def _step_rough_align(doc, o, prog):
    from .geom_utils import rotation_between
    m = doc.model
    floor = next((f for f in doc.features if f.kind == "plane" and f.params.get("fixed")), None)
    T = np.eye(4)
    out = []
    if floor is not None:
        # Boden nach unten (−Z), dann Hauptachsen in der XY-Ebene
        R1 = rotation_between(unit(floor.params["normal"]), np.array([0, 0, -1.0]))
        V = m.V @ R1.T
        c = V.mean(0)
        Q = (V - c)[:, :2]
        ev, evec = np.linalg.eigh(Q.T @ Q)
        x = evec[:, 1]
        ang = math.atan2(x[1], x[0])
        cz, sz = math.cos(-ang), math.sin(-ang)
        R2 = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1.0]])
        R = R2 @ R1
        out.append(tr("Boden nach unten, Hauptachsen in XY", "floor down, principal axes in XY"))
    else:
        from .mesh_ops import principal_axes_rotation
        R, _ = principal_axes_rotation(m)
        out.append(tr("Hauptachsen (größte Ausdehnung = X, größte ebene Fläche unten)",
                      "principal axes (largest extent = X, largest flat face down)"))
    T[:3, :3] = R
    doc.apply_transform(T)
    mn, mx = doc.model.bbox
    T2 = np.eye(4)
    T2[:2, 3] = -0.5 * (mn[:2] + mx[:2])
    T2[2, 3] = -mn[2]
    doc.apply_transform(T2)
    out.append(tr("Mitte XY → 0, Unterseite → z = 0", "centre XY → 0, bottom → z = 0"))
    return out


def _step_detect(doc, o, prog):
    from .features import Feature, _stats, trim_primitive
    from .ransac import detect_primitives
    m = doc.model
    excl = doc.assigned() > 0
    eps = doc.detect_eps
    eps_c = 1.5 * float(doc.tol.get("sagitta", 0.0)) if m.exact else None
    res = detect_primitives(m, eps=eps, angle_deg=15.0, min_area_frac=o["detect_min_area"],
                            max_shapes=o["detect_max"], exclude_mask=excl, eps_coarse=eps_c,
                            progress=lambda n, k: prog(tr(f"{n} Formen", f"{n} shapes")))
    n = {}

    flat = None
    if m.exact:
        from .preprocess import planar_patch_faces
        flat = planar_patch_faces(m)

    def take(res, strict=False):
        for kind, fr, ids in res:
            if strict and (fr.rms > 0.6 * eps or len(ids) < 20):
                continue
            # CAD-Export: eine einzelne ebene Facette einer gekrümmten Fläche ist keine Ebene
            if strict and kind == "plane" and flat is not None and np.mean(flat[ids]) < 0.5:
                continue
            try:
                pr = trim_primitive(kind, fr.params, m, ids, {"outline_tol": max(0.05, 2 * eps)})
                f = Feature(kind, pr, "surface", ids, stats=_stats(fr.residuals, fr.weights), auto=True)
                doc.features.append(f)
                n[kind] = n.get(kind, 0) + 1
            except Exception:
                continue

    take(res)
    # zweiter Durchgang für kleine Formen (kleine Bohrungen, Fasen, Sechskant-Taschen) – nur auf den
    # noch freien Dreiecken und nur sehr gut passende Formen
    excl = doc.assigned() > 0
    if (~excl & ~m.synthetic).sum() > 50:
        res2 = detect_primitives(m, eps=eps, angle_deg=15.0, min_area_frac=o["detect_min_area"] / 8.0,
                                 max_shapes=o["detect_max"], exclude_mask=excl, eps_coarse=eps_c,
                                 kinds=("plane", "cylinder", "cone", "sphere"), seed=1,
                                 progress=lambda k_, kk: prog(tr(f"{k_} kleine Formen", f"{k_} small shapes")))
        take(res2, strict=True)
    from .fitting import KIND_LABELS
    return [tr("Erkannt: ", "Detected: ") + (", ".join(f"{v}× {KIND_LABELS[k]}" for k, v in n.items()) or "–")]


def _step_merge(doc, o, prog):
    from .analysis import merge_similar
    res = merge_similar(doc.model, doc.features, doc.fit_tol)
    removed = set()
    for nf, old in res:
        removed |= set(old)
    if not res:
        return [tr("Nichts zusammenzuführen.", "Nothing to merge.")]
    first = {}
    for nf, old in res:
        first[old[0]] = nf
    new = []
    for f in doc.features:
        if f.id in first:
            new.append(first[f.id])
        elif f.id not in removed:
            new.append(f)
    doc.features = new
    return [tr(f"{len(removed)} Features zu {len(res)} zusammengeführt", f"{len(removed)} features merged into {len(res)}")]


def _step_optimize_align(doc, o, prog):
    from .analysis import floor_translation, manhattan_frame, optimize_alignment
    from .features import snap_feature
    out = []
    # liegen die Konstruktionsflächen schräg zu den Achsen (Hauptachsen ≠ Konstruktionsachsen),
    # erst grob so drehen, dass möglichst viel Fläche achsparallel ist
    floor = next((f for f in doc.features if f.kind == "plane" and f.params.get("fixed")), None)
    try:
        R, s0, s1 = manhattan_frame(doc.features, o["capture_deg"],
                                    floor_normal=None if floor is None else floor.params["normal"])
        if R is not None and s1 > max(1.15 * s0, s0 + 0.1):
            # „größte ebene Fläche unten“ beibehalten: Vorzeichen von Z nach der Bodenfläche wählen
            Rz = R
            if floor is None:
                from .geom_utils import unit as _u
                down = up = 0.0
                for f in doc.features:
                    if f.kind == "plane":
                        nz = float(_u(f.params["normal"]) @ Rz[2])
                        a = float(doc.model.areas[np.asarray(f.face_ids, np.int64)].sum()) if len(f.face_ids) else 0.0
                        if nz < -0.99:
                            down = max(down, a)
                        elif nz > 0.99:
                            up = max(up, a)
                if up > down:
                    Rz = np.vstack([Rz[0], -Rz[1], -Rz[2]])
            mn, mx = doc.model.bbox
            c = 0.5 * (mn + mx)
            T = np.eye(4)
            T[:3, :3] = Rz
            T[:3, 3] = c - Rz @ c
            doc.apply_transform(T)
            out.append(tr(f"an den Konstruktionsflächen ausgerichtet: achsparallel {100 * s0:.0f} % → {100 * s1:.0f} % "
                          "der Richtungen (gewichtet)",
                          f"aligned to the design faces: axis-parallel {100 * s0:.0f} % → {100 * s1:.0f} % "
                          "of directions (weighted)"))
    except Exception as e:  # noqa: BLE001
        out.append(tr(f"Grobausrichtung an Flächen übersprungen: {e}", f"face-based alignment skipped: {e}"))
    try:
        mn, mx = doc.model.bbox
        T, rep = optimize_alignment(doc.features, o["capture_deg"], center=0.5 * (mn + mx))
        doc.apply_transform(T)
        out.append(tr(f"Ausrichtung optimiert: {rep['n_used']}/{rep['n_total']} Richtungen, Drehung {rep['rotation_deg']:.3f}°, "
                      f"max. Abweichung {rep['max_before']:.3f}° → {rep['max_after']:.4f}°",
                      f"alignment optimised: {rep['n_used']}/{rep['n_total']} directions, rotation {rep['rotation_deg']:.3f}°, "
                      f"max deviation {rep['max_before']:.3f}° → {rep['max_after']:.4f}°"))
    except Exception as e:  # noqa: BLE001
        out.append(tr(f"Ausrichtung nicht optimiert: {e}", f"alignment not optimised: {e}"))
    Tf = floor_translation(doc.features)
    if Tf is not None and abs(Tf[2, 3]) > 1e-9:
        doc.apply_transform(Tf)
        out.append(tr(f"Bodenebene auf z = 0 ({Tf[2, 3]:+.4f} mm)", f"floor plane to z = 0 ({Tf[2, 3]:+.4f} mm)"))
    n = 0
    for f in doc.features:
        if f.params.get("fixed") or f.kind not in ("plane", "cylinder", "cone", "torus"):
            continue
        try:
            ch = snap_feature(doc.model, f, o["straighten_deg"], 0.0, robust_tol=2 * doc.fit_tol)
            if ch:
                n += 1
        except Exception:
            continue
    out.append(tr(f"{n} Achsen/Normalen exakt auf X/Y/Z gestellt (≤ {o['straighten_deg']:g}°, mit Neu-Fit)",
                  f"{n} axes/normals set exactly to X/Y/Z (≤ {o['straighten_deg']:g}°, refitted)"))
    return out


def _step_rotational(doc, o, prog):
    """Ist (fast) das ganze Teil rotationssymmetrisch um eine Achse, die Flächen aus dem (r, z)-Profil
    bestimmen: Strecken → Ebenen/Zylinder/Kegel, Bögen → Tori/Kugeln. Das ersetzt die einzeln per
    RANSAC gefundenen Stücke (Rundungen zerfallen dort gern in viele schmale Bänder)."""
    from .features import Feature, _stats, trim_primitive
    from .rotational import dominant_axis, rotational_fraction, segment_rotational
    m = doc.model
    ax = dominant_axis(doc.features, m, tol=doc.fit_tol)
    if ax is None:
        return [tr("Keine Drehachse – übersprungen.", "No axis of revolution – skipped.")]
    q, a, _w = ax
    frac = rotational_fraction(m, q, a)
    if frac < o["rot_min_frac"]:
        return [tr(f"Kein Drehteil ({100 * frac:.0f} % der Fläche rotationssymmetrisch) – übersprungen.",
                   f"Not a turned part ({100 * frac:.0f} % of the area rotationally symmetric) – skipped.")]
    faces = np.nonzero(~m.synthetic)[0]
    res = segment_rotational(m, faces, q, a, doc.detect_eps, o["rot_min_area"] * float(m.areas[faces].sum()))
    if not res:
        return [tr("Profil nicht auswertbar – übersprungen.", "Profile could not be evaluated – skipped.")]
    covered = np.zeros(m.n_faces, bool)
    new = []
    for kind, params, ids, rms in res:
        try:
            pr = trim_primitive(kind, params, m, ids, {"outline_tol": max(0.05, 2 * doc.detect_eps)})
        except Exception:
            pr = params
        f = Feature(kind, pr, "surface", ids, stats={"rms": rms, "p95": rms, "max": rms, "n": int(len(ids))}, auto=True)
        new.append(f)
        covered[ids] = True
    cov = float(m.areas[covered].sum() / max(m.areas[faces].sum(), 1e-300))
    keep = [f for f in doc.features if f.params.get("fixed") or f.role not in ("surface", "sub")]
    removed = len(doc.features) - len(keep)
    doc.features = keep + new
    doc.tol["rot_axis"] = (np.asarray(q, float), np.asarray(a, float))
    kinds = {}
    for f in new:
        kinds[f.kind] = kinds.get(f.kind, 0) + 1
    from .fitting import KIND_LABELS
    return [tr(f"Drehteil erkannt ({100 * frac:.0f} % rotationssymmetrisch): {len(new)} Flächen aus dem Profil "
               f"({', '.join(f'{v}× {KIND_LABELS.get(k, k)}' for k, v in kinds.items())}) ersetzen {removed} einzeln "
               f"erkannte; abgedeckt {100 * cov:.1f} %",
               f"turned part detected ({100 * frac:.0f} % rotationally symmetric): {len(new)} faces from the profile "
               f"({', '.join(f'{v}× {KIND_LABELS.get(k, k)}' for k, v in kinds.items())}) replace {removed} single "
               f"detections; covered {100 * cov:.1f} %")]


def _step_holes(doc, o, prog):
    from .bores import detect_holes, hole_summary, snap_hole
    from .features import Feature
    if doc.tol.get("rot_axis") is not None:
        # Drehteil: Bohrungen auf der Drehachse sind Profilflächen (exakte Werte, keine Normbohrung)
        q0, a0 = doc.tol["rot_axis"]
        feats = [f for f in doc.features
                 if not (f.kind == "cylinder" and abs(float(np.asarray(f.params["axis"]) @ a0)) > 0.9999
                         and np.linalg.norm(np.cross(np.asarray(f.params["point"]) - q0, a0)) < 10 * doc.fit_tol)]
    else:
        feats = doc.features
    res = detect_holes(doc.model, feats, doc.fit_tol)
    consumed = set()
    out = []
    for r in res:
        p, notes = snap_hole(r["params"], axis_dirs=np.eye(3),
                             meas=min(doc.fit_tol, max(2.0 * float(r.get("rms", doc.fit_tol)), 0.005)))
        f = Feature("hole", p, "sub", r["face_ids"], auto=True)
        f.name = tr("Bohrung", "Hole") + f" {f.id}"
        doc.features.append(f)
        consumed |= set(r["consumed"])
        out.append(f"{f.name}: {hole_summary(p)}" + (f"  [{'; '.join(notes + r['notes'])}]" if notes or r["notes"] else ""))
    doc.features = [f for f in doc.features if f.id not in consumed]
    if not res:
        out.append(tr("Keine Bohrungen gefunden.", "No holes found."))
    unseen = sum(1 for f in doc.features if f.kind == "hole" and not f.params.get("bottom_seen", True)
                 and f.params.get("bottom") != "through")
    if unseen:
        out.append(tr(f"⚠ {unseen} Bohrung(en) ohne erfassten Boden: Tiefe/Boden im Bohrungen-Menü festlegen",
                      f"⚠ {unseen} hole(s) without scanned bottom: set depth/bottom in the holes menu"))
    return out


def _step_ports(doc, o, prog):
    from .ports import analyze_ports, apply_ports
    if doc.tol.get("rot_axis") is not None:
        return [tr("Drehteil – übersprungen.", "Turned part – skipped.")]
    res = analyze_ports(doc.model, doc.features, doc.fit_tol, labels=doc.assigned())
    apply_ports(doc, res)
    return res["report"] or [tr("Keine Drehkonturen an Bohrungen gefunden.", "No turned contours at holes found.")]


def _step_cleanup(doc, o, prog):
    from .blends import cleanup
    if doc.tol.get("kind") == "cad" or doc.model.exact:
        return [tr("CAD-Netz – übersprungen.", "CAD mesh – skipped.")]
    return cleanup(doc, r_max=o["blend_r_max"])


def _step_symmetry(doc, o, prog):
    from .symmetry import detect_all, sym_label
    keep = [s for s in doc.symmetries if not s.get("auto")]
    found = detect_all(doc.model, doc.features, doc.fit_tol, existing=keep)
    for s in found:
        s["auto"] = True
    doc.symmetries = keep + found
    out = [f"{s['name']}: {sym_label(s)} ({100 * s['score']:.0f}%)" + (f" – {s['info']}" if s.get("info") else "")
           for s in found]
    return out or [tr("Keine Symmetrien gefunden.", "No symmetries found.")]


def _step_freeform(doc, o, prog):
    from .autopatch import auto_freeform
    m = doc.model
    free = doc.assigned() == 0
    free &= ~m.synthetic
    total = float(m.areas.sum())
    el = m.mean_edge_length()
    cand = []
    for comp in m.components_of(np.nonzero(free)[0]):
        A = float(m.areas[comp].sum())
        if A < o["freeform_min_frac"] * total:
            continue
        loops = m.boundary_loops(comp)
        U = sum(float(np.linalg.norm(np.diff(m.V[np.r_[l, l[:1]]], axis=0), axis=1).sum()) for l in loops)
        if U > 0 and 2 * A / U < 3 * el:
            continue                     # schmaler Streifen an einer Kante
        cand.append(comp)
    if not cand:
        return [tr("Keine größeren freien Bereiche – nichts zu tun (Kantenstreifen übernimmt die Scan-Topologie).",
                   "No larger free areas – nothing to do (edge strips are handled by the scan topology).")]
    ids = np.concatenate(cand)
    feats, rest = auto_freeform(m, ids, tol=doc.fit_tol, progress=lambda k, n: prog(f"{k}/{n}"))
    doc.features += feats
    return [tr(f"{len(feats)} Freiform-Flächen für {len(cand)} freie Bereiche", f"{len(feats)} freeform surfaces for {len(cand)} free areas")]


def _step_preview(doc, o, prog):
    from .export import build_scan_topology
    res = build_scan_topology(doc.model, doc.features, tol=doc.fit_tol, progress=prog)
    doc.preview = res
    return list(res["report"])
