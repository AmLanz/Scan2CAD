"""Geschlossenes B-Rep aus der Scan-Topologie ("Exact Surfacing").

Idee (Várady/Benkő-Pipeline: Segmentierung -> Flächenfit -> B-Rep-Aufbau): Die Nachbarschaft
der Flächen wird NICHT durch Verschneiden der Flächen geraten, sondern aus dem (geschlossenen)
Scan-Netz abgelesen. Jedes Dreieck gehört genau einer Trägerfläche; daraus folgt, welche
Flächen sich wo berühren. Weil jede Kante von genau zwei Flächen GEMEINSAM benutzt wird, ist die
Hülle per Konstruktion dicht.

  1. Regionen   Dreiecke -> Trägerfläche (gefittete Features). Freie Dreiecke und Dreiecke von
                Körper-Features (Bohrungen, Zapfen …) werden dem nächsten Träger zugeschlagen
                (geodätisches Voronoi); Grenzen werden per ICM nach Abstand+Normale geglättet,
                Inseln entfernt. Rundum-Regionen periodischer Flächen werden an einer Naht geteilt.
  2. Topologie  Regionsgrenzen -> Ketten zwischen Ecken (Punkte mit >= 3 Regionen). Sehr kurze
                Ketten werden zu einer Ecke zusammengezogen (Scan-Rauschen an Ecken).
  3. Geometrie  Ecken: Gauß-Newton auf alle angrenzenden Träger. Kanten: exakte Schnittkurve
                (GeomAPI_IntSS: Gerade/Kreis/Ellipse/B-Spline), sonst Mitte der Projektionen als
                B-Spline (tangentiale Übergänge). Kanten-Toleranz = max. Abstand zu beiden Trägern.
  4. Flächen    Träger + Drähte -> Fläche -> Schale -> Volumenkörper; danach Körper-Features per
                Boolean (Zapfen +, Bohrungen −).
"""
from __future__ import annotations

import math

import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components, dijkstra

from .carriers import PERIODIC_KINDS, Carrier, carrier_of, is_carrier
from .geom_utils import angular_range, orthonormal_frame, unit
from .i18n import tr


# ============================================================================ Hilfen
def _dual_graph(model, weighted=True):
    pairs, _ = model.adjacency_pairs
    nF = model.n_faces
    if weighted:
        w = np.linalg.norm(model.centroids[pairs[:, 0]] - model.centroids[pairs[:, 1]], axis=1) + 1e-9
    else:
        w = np.ones(len(pairs))
    G = sparse.csr_matrix((np.r_[w, w], (np.r_[pairs[:, 0], pairs[:, 1]], np.r_[pairs[:, 1], pairs[:, 0]])),
                          shape=(nF, nF))
    return G


def _dual_neighbors(model):
    """Bis zu 3 Nachbardreiecke je Dreieck (fehlende = eigener Index)."""
    pairs, _ = model.adjacency_pairs
    nF = model.n_faces
    nb = np.tile(np.arange(nF)[:, None], (1, 3))
    cnt = np.zeros(nF, np.int64)
    for a, b in ((pairs[:, 0], pairs[:, 1]), (pairs[:, 1], pairs[:, 0])):
        order = np.argsort(a, kind="stable")
        a, b = a[order], b[order]
        for x, y in zip(a, b):
            if cnt[x] < 3:
                nb[x, cnt[x]] = y
                cnt[x] += 1
    return nb


def _nearest_label(model, labels, G=None, allowed=None):
    """Unbeschriftete Dreiecke (-1) bekommen das Label des geodätisch nächsten beschrifteten."""
    G = _dual_graph(model) if G is None else G
    src = np.nonzero(labels >= 0)[0]
    if len(src) == 0:
        return labels
    if allowed is not None:
        keep = allowed | (labels >= 0)
        idx = np.nonzero(keep)[0]
    _, _, sources = dijkstra(G, directed=False, indices=src, min_only=True, return_predecessors=True)
    out = labels.copy()
    m = (out < 0) & (sources >= 0)
    out[m] = labels[sources[m]]
    return out


class _Fast:
    """Schneller Abstand für Freiform-Träger über eine Tessellierung (VTK)."""

    def __init__(self, car):
        self.car = car
        self.pd = None

    def distance(self, P):
        if self.car.kind not in ("freeform", "ellipsoid"):
            return np.abs(self.car.distance(P))
        if self.pd is None:
            from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeFace
            from . import occ_build as ob
            from .preprocess import vf_to_pd
            srf = self.car.geom()
            u0, u1, v0, v1 = srf.Bounds()
            f = BRepBuilderAPI_MakeFace(srf, u0, u1, v0, v1, 1e-7).Face()
            V, F = ob.tessellate(f, 0.02, 0.2)
            self.pd = vf_to_pd(V, F)
        import pyvista as pv
        d = pv.PolyData(np.asarray(P, float)).compute_implicit_distance(self.pd)
        return np.abs(np.asarray(d.point_data["implicit_distance"]))


# ============================================================================ 1. Regionen
def _prefilter(ca, cb, tol):
    if ca.kind != cb.kind or ca.kind in ("freeform", "ellipsoid"):
        return False
    k, p, q = ca.kind, ca.p, cb.p
    lim = 30 * tol + 0.2
    if k == "plane":
        return abs(float(unit(p["normal"]) @ unit(q["normal"]))) > math.cos(math.radians(6))
    if k == "sphere":
        return (np.linalg.norm(np.asarray(p["center"]) - q["center"]) < lim
                and abs(p["radius"] - q["radius"]) < lim)
    if k in ("cylinder", "cone", "torus"):
        if abs(float(ca.axis @ cb.axis)) < math.cos(math.radians(6)):
            return False
        d = cb.base - ca.base
        if k == "cylinder":
            off = np.linalg.norm(d - (d @ ca.axis) * ca.axis)
            return off < lim and abs(p["radius"] - q["radius"]) < lim
        if k == "cone":
            return (np.linalg.norm(d) < 10 * lim
                    and abs(p["half_angle"] - q["half_angle"]) < math.radians(4))
        return np.linalg.norm(d) < lim and abs(p["R"] - q["R"]) < lim and abs(p["r"] - q["r"]) < lim
    return False


def merge_equivalent(model, carriers, cfeat, tol, n_sample=400):
    """Repräsentant je Träger: zwei Träger gleicher Art werden zusammengelegt, wenn jeder die
    Messpunkte des anderen innerhalb der Toleranz erklärt (gegenseitiger Datentest)."""
    n = len(carriers)
    rep = np.arange(n)
    rng = np.random.default_rng(0)
    samples = []
    for f in cfeat:
        ids = np.asarray(f.face_ids, np.int64)
        ids = ids[(ids >= 0) & (ids < model.n_faces)]
        ids = ids[~model.synthetic[ids]] if (~model.synthetic[ids]).sum() >= 20 else ids
        if len(ids) > n_sample:
            ids = rng.choice(ids, n_sample, replace=False)
        samples.append(model.centroids[ids])

    def explains(a, b):
        P = samples[b]
        if len(P) == 0:
            return False
        d = np.abs(carriers[a].distance(P))
        return float(np.sqrt(np.mean(d ** 2))) < 1.6 * tol and float(np.percentile(d, 90)) < 3.0 * tol

    def find(i):
        while rep[i] != i:
            rep[i] = rep[rep[i]]
            i = rep[i]
        return i

    order = np.argsort([-len(f.face_ids) for f in cfeat])
    for ii, i in enumerate(order):
        for j in order[ii + 1:]:
            ri, rj = find(i), find(j)
            if ri == rj or not _prefilter(carriers[i], carriers[j], tol):
                continue
            if explains(i, j) and explains(j, i):
                rep[rj] = ri
    return np.array([find(i) for i in range(n)])


def _aux_carriers(model, labels, carriers, G, tol, min_faces, absorbed):
    """Freie Bereiche, die kein benachbarter Träger erklärt (Abstand > 3·tol), bekommen eine
    eigene Hilfsfläche: überwiegend künstliche Füllung -> Ebene durch den echten Rand (z. B. Boden
    eines nicht erfassten Bohrungsgrunds), sonst Ausgleichsebene oder Freiform.
    Rückgabe: Liste (Carrier, Feature, face_ids)."""
    from .features import Feature, trim_primitive
    from .fitting import fit_plane
    free = (labels < 0) & ~absorbed
    if free.sum() < min_faces:
        return []
    nn = _nearest_label(model, labels, G)
    C = model.centroids
    dmin = np.full(model.n_faces, np.inf)
    fids = np.nonzero(free)[0]
    for comp in model.components_of(fids):
        mm = np.zeros(model.n_faces, bool)
        mm[comp] = True
        cand = np.unique(nn[model.grow(mm, 2)])
        for c in cand:
            if c < 0:
                continue
            d = np.abs(_Fast(carriers[c]).distance(C[comp]))
            dmin[comp] = np.minimum(dmin[comp], d)
    unexpl = free & (dmin > 3.0 * tol)
    el = model.mean_edge_length()
    out = []
    car_area = np.bincount(labels[labels >= 0], model.areas[labels >= 0], minlength=len(carriers))
    for ids in model.components_of(np.nonzero(unexpl)[0]):
        # schmale Streifen entlang von Kanten (Scan-Verrundung) sind keine eigenen Flächen:
        # mittlere Breite 2A/U muss deutlich über der Kantenlänge liegen
        A = float(model.areas[ids].sum())
        # kleine Delle/Fehlstelle mitten in einer Ebene (rundum nur diese Ebene): eben schließen
        mm = np.zeros(model.n_faces, bool)
        mm[ids] = True
        ring = model.grow(mm, 2) & ~mm
        nbl = np.unique(labels[ring])
        nbl = nbl[nbl >= 0]
        if len(nbl) == 1 and carriers[nbl[0]].kind == "plane" and A < 0.1 * car_area[nbl[0]]:
            dpl = float(np.abs(_Fast(carriers[nbl[0]]).distance(C[ids])).max())
            if dpl < max(1.5, 0.5 * math.sqrt(A)):
                labels[ids] = nbl[0]
                continue
        if A < 25 * el * el:
            continue
        loops = model.boundary_loops(ids)
        U = sum(float(np.linalg.norm(np.diff(model.V[np.r_[l, l[:1]]], axis=0), axis=1).sum()) for l in loops)
        if U <= 0 or 2 * A / U < 2.0 * el:
            continue
        syn = float(model.synthetic[ids].mean())
        nrm_mean = unit((model.normals[ids] * model.areas[ids, None]).sum(0))
        try:
            fplane = None
            for fl in model.fills:
                if fl.get("plane") is not None and len(fl["faces"]):
                    if np.isin(ids, np.asarray(fl["faces"], np.int64)).mean() > 0.5:
                        fplane = fl["plane"]
                        break
            if fplane is not None:          # Füllung mit festgelegter Ebene (Boden, "als Ebene erzwingen")
                n = unit(fplane[1])
                params = {"origin": np.asarray(fplane[0], float), "normal": n if n @ nrm_mean >= 0 else -n}
            elif syn >= 0.5:
                R = model.V[np.concatenate(loops)] if loops else model.centroids[ids]
                o = R.mean(0)
                _, _, vt = np.linalg.svd(R - o, full_matrices=False)
                n = vt[2] if vt[2] @ nrm_mean >= 0 else -vt[2]
                params = {"origin": o, "normal": n}
            else:
                fr = fit_plane(C[ids], model.fit_normals[ids], model.weights[ids])
                if fr.rms > 2.0 * tol:
                    from .autopatch import auto_freeform
                    feats, _ = auto_freeform(model, ids, tol=tol, min_faces=min_faces)
                    for f in feats:
                        f.name = tr("Hilfsfläche", "Auxiliary surface") + f" {f.id}"
                        f.params["aux"] = True
                        out.append((carrier_of(f), f, np.asarray(f.face_ids, np.int64)))
                    continue
                params = dict(fr.params)
                if unit(params["normal"]) @ nrm_mean < 0:
                    params["normal"] = -unit(params["normal"])
            dpl = np.abs((C[ids] - params["origin"]) @ unit(params["normal"]))
            if fplane is None and float(np.percentile(dpl, 90)) > max(10 * tol, 0.25 * math.sqrt(A)):
                continue            # keine sinnvolle Ebene -> dem Nachbarträger überlassen
            pr = trim_primitive("plane", params, model, ids, {"outline_tol": max(0.05, 2 * tol)})
            pr["aux"] = True
            f = Feature("plane", pr, "surface", ids, name=tr("Hilfsebene", "Auxiliary plane"), auto=True)
            f.name = f"{f.name} {f.id}"
            out.append((carrier_of(f), f, ids))
        except Exception:
            continue
    return out


def _resolve_cups(model, labels, carriers, cfeat, tol, min_faces):
    """Napf-Regionen (Zylinder rundum mit nur einem Rand, z. B. Sackloch, dessen nicht erfasster
    Grund von einer Membran-Füllung geschlossen wurde): ab dem Ende der echten Messdaten wird ein
    ebener Boden senkrecht zur Achse angenommen (neuer Hilfsträger). Ändert labels in place."""
    from .features import Feature, trim_primitive
    n_done = 0
    for c in range(len(carriers)):
        car = carriers[c]
        if car.kind != "cylinder":
            continue
        ids_all = np.nonzero(labels == c)[0]
        for comp in model.components_of(ids_all):
            loops = model.boundary_loops(comp)
            if len(loops) != 1:
                continue
            a = car.axis
            x, y, _ = _split_frame(car)
            q = model.centroids[comp] - car.base
            u0, u1 = angular_range(np.arctan2(q @ y, q @ x), 25.0)
            if (u1 - u0) < 2 * math.pi - 1e-6:
                continue
            h = q @ a
            h_open = float(np.mean((model.V[loops[0]] - car.base) @ a))
            sgn = 1.0 if np.mean(h) > h_open else -1.0       # Richtung zum geschlossenen Ende
            real = ~model.synthetic[comp]
            d = np.abs(car.distance(model.centroids[comp]))
            ok = real & (d < 3 * tol)
            if ok.sum() < 10:
                continue
            h_cut = float(np.max(sgn * h[ok])) * sgn
            bottom = comp[sgn * (h - h_cut) > 0]
            if len(bottom) < 3:
                continue
            n = sgn * a
            params = {"origin": car.base + h_cut * a, "normal": n}
            try:
                pr = trim_primitive("plane", params, model, bottom, {"outline_tol": max(0.05, 2 * tol)})
            except Exception:
                pr = dict(params)
            pr["aux"] = True
            f = Feature("plane", pr, "surface", bottom, name=tr("Boden (angenommen)", "Floor (assumed)"),
                        auto=True)
            f.name = f"{f.name} {f.id}"
            labels[bottom] = len(carriers)
            carriers.append(carrier_of(f))
            cfeat.append(f)
            n_done += 1
    return n_done


def assign_regions(model, features, tol, min_faces=25, icm_sweeps=4, log=None, passive=None):
    """Rückgabe: label je Dreieck (Index in carriers), carriers, absorbed-Maske."""
    log = log or (lambda s: None)
    nF = model.n_faces
    carriers, cfeat = [], []
    for f in features:
        if is_carrier(f):
            carriers.append(carrier_of(f))
            cfeat.append(f)
    if not carriers:
        raise RuntimeError(tr("Keine Flächen-Features (Träger) vorhanden.", "No surface features (carriers)."))
    labels = -np.ones(nF, np.int64)
    for ci, f in enumerate(cfeat):
        ids = np.asarray(f.face_ids, np.int64)
        ids = ids[(ids >= 0) & (ids < nF)]
        ids = ids[labels[ids] < 0]
        labels[ids] = ci
    # gleichwertige Träger (z. B. zwei Hälften einer Bohrung, getrennt erkannt) zusammenlegen
    rep = merge_equivalent(model, carriers, cfeat, tol)
    nmerge = int((rep != np.arange(len(rep))).sum())
    if nmerge:
        m = labels >= 0
        labels[m] = rep[labels[m]]
        log(tr(f"{nmerge} gleichwertige Flächen zusammengelegt", f"{nmerge} equivalent surfaces merged"))
    absorbed = np.zeros(nF, bool)
    if passive is not None and len(passive):
        absorbed[np.asarray(passive, np.int64)] = True      # Platzhalter (z. B. Deckel verstopfter Bohrungen)
    for f in features:
        if not is_carrier(f):
            ids = np.asarray(f.face_ids, np.int64)
            ids = ids[(ids >= 0) & (ids < nF)]
            absorbed[ids] = True
            labels[ids] = -1
    G = _dual_graph(model)
    free0 = int((labels < 0).sum())
    aux = _aux_carriers(model, labels, carriers, G, tol, min_faces, absorbed)
    for car, f, ids in aux:
        labels[ids] = len(carriers)
        carriers.append(car)
        cfeat.append(f)
    if aux:
        log(tr(f"{len(aux)} Hilfsflächen für unerklärte Bereiche (z. B. Füllungen) ergänzt",
               f"{len(aux)} auxiliary surfaces added for unexplained areas (e.g. fills)"))
        rep = merge_equivalent(model, carriers, cfeat, tol)
        m = labels >= 0
        labels[m] = rep[labels[m]]
    labels = _nearest_label(model, labels, G)
    nb = _dual_neighbors(model)
    fast = [_Fast(c) for c in carriers]
    cosn = model.fit_normals
    data = ~absorbed
    for sweep in range(icm_sweeps):
        nl = labels[nb]
        band = np.nonzero((nl != labels[:, None]).any(1))[0]
        if len(band) == 0:
            break
        cand = np.c_[labels[band], nl[band]]
        cost = np.full(cand.shape, np.inf)
        for c in np.unique(cand):
            if c < 0:
                continue
            rows, cols = np.nonzero(cand == c)
            fb = band[rows]
            d = np.zeros(len(fb))
            dm = data[fb]
            if dm.any():
                P = model.centroids[fb[dm]]
                dd = fast[c].distance(P) / max(tol, 1e-6)
                # Normale
                car = carriers[c]
                if car.kind in ("freeform", "ellipsoid"):
                    pen = np.zeros(len(P))
                else:
                    try:
                        nrm = car.normal(car.project(P))
                        pen = np.where(np.abs(np.einsum("ij,ij->i", nrm, cosn[fb[dm]])) < 0.7, 9.0, 0.0)
                    except Exception:
                        pen = np.zeros(len(P))
                d[dm] = np.minimum(dd ** 2, 25.0) + pen
            potts = (labels[nb[fb]] != c).sum(1) * 2.0
            cost[rows, cols] = d + potts
        best = cand[np.arange(len(band)), np.argmin(cost, axis=1)]
        cur = cost[:, 0]
        newc = cost.min(1)
        upd = (best != labels[band]) & (newc < cur - 0.5)
        if not upd.any():
            break
        labels[band[upd]] = best[upd]
    # Inseln entfernen
    for _ in range(6):
        changed = False
        for c in range(len(carriers)):
            ids = np.nonzero(labels == c)[0]
            if len(ids) == 0:
                continue
            for comp in model.components_of(ids):
                if len(comp) < min_faces:
                    labels[comp] = -1
                    changed = True
        if not changed:
            break
        labels = _nearest_label(model, labels, G)
    ncup = _resolve_cups(model, labels, carriers, cfeat, tol, min_faces)
    if ncup:
        log(tr(f"{ncup} Bohrungsgrund/Napf ohne Daten mit ebenem Boden geschlossen (Annahme – im "
               "Bohrungs-Werkzeug anpassen)",
               f"{ncup} unscanned hole/cup bottoms closed with a flat floor (assumption – adjust in the "
               "hole tool)"))
    if free0:
        log(tr(f"{free0} Dreiecke ohne Feature dem nächsten Träger zugeordnet",
               f"{free0} triangles without feature assigned to the nearest carrier"))
    labels, _ = fix_pinches(model, labels)
    return labels, carriers, cfeat, absorbed


def fix_pinches(model, labels, rounds=6):
    """Beseitigt 'Einschnürungen': Ecken, an denen dasselbe Label in mehreren getrennten Sektoren
    des Dreiecksfächers vorkommt (die Regionsgrenze berührt sich selbst). Kleinere Sektoren
    bekommen das häufigste Nachbarlabel im Fächer."""
    F = model.F
    nV = model.n_verts
    labels = labels.copy()
    fl = F.reshape(-1)
    order = np.argsort(fl, kind="stable")
    starts = np.searchsorted(fl[order], np.arange(nV + 1))
    vf = order // 3
    fixed = 0
    for _ in range(rounds):
        lab3 = labels[F]
        mixed = np.zeros(nV, bool)
        for k in range(3):
            diff = (lab3[:, k] != lab3[:, (k + 1) % 3]) | (lab3[:, k] != lab3[:, (k + 2) % 3])
            mixed[F[diff, k]] = True
        changed = 0
        for v in np.nonzero(mixed)[0]:
            fs = vf[starts[v]:starts[v + 1]]
            if len(fs) < 3:
                continue
            labs = labels[fs]
            # Nachbarschaft im Fächer: gemeinsame Kante (v, w)
            other = {}
            for i, f in enumerate(fs):
                for w in F[f]:
                    if w != v:
                        other.setdefault(int(w), []).append(i)
            parent = list(range(len(fs)))

            def find(a):
                while parent[a] != a:
                    parent[a] = parent[parent[a]]
                    a = parent[a]
                return a

            for lst in other.values():
                if len(lst) == 2 and labs[lst[0]] == labs[lst[1]]:
                    parent[find(lst[0])] = find(lst[1])
            comps = {}
            for i in range(len(fs)):
                comps.setdefault(find(i), []).append(i)
            by_lab = {}
            for root, mem in comps.items():
                by_lab.setdefault(int(labs[mem[0]]), []).append(mem)
            for lab, secs in by_lab.items():
                if len(secs) < 2:
                    continue
                secs.sort(key=lambda s: -len(s))
                for sec in secs[1:]:
                    # häufigstes anderes Label, das an den Sektor grenzt
                    cnt = {}
                    for i in sec:
                        for w in F[fs[i]]:
                            for j in other.get(int(w), []):
                                if labs[j] != lab:
                                    cnt[int(labs[j])] = cnt.get(int(labs[j]), 0) + 1
                    if not cnt:
                        continue
                    new = max(cnt, key=cnt.get)
                    labels[fs[sec]] = new
                    labs[sec] = new
                    changed += 1
        fixed += changed
        if not changed:
            break
    return labels, fixed


def _absorb_small_regions(model, reg, regions, min_n):
    """Splitter-Regionen (z. B. nach dem Teilen periodischer Flächen) der Nachbarregion mit der
    längsten gemeinsamen Grenze zuschlagen (bevorzugt gleicher Träger). Ändert reg in place."""
    pairs, _ = model.adjacency_pairs
    n_done = 0
    for _ in range(3):
        cnt = np.bincount(reg[reg >= 0], minlength=len(regions))
        small = np.nonzero((cnt > 0) & (cnt < min_n))[0]
        if len(small) == 0:
            break
        a, b = reg[pairs[:, 0]], reg[pairs[:, 1]]
        for r in small:
            nb = np.r_[b[a == r], a[b == r]]
            nb = nb[(nb != r) & (nb >= 0)]
            if len(nb) == 0:
                continue
            vals, c = np.unique(nb, return_counts=True)
            same = np.array([regions[v]["carrier"] == regions[r]["carrier"] for v in vals])
            score = c + 1000 * same
            reg[reg == r] = vals[int(np.argmax(score))]
            n_done += 1
    return n_done


def _region_loops_count(model, faces):
    return len(model.boundary_loops(faces))


def _split_frame(car):
    """Kanonischer Rahmen um die Achse: koaxiale Flächen werden in derselben Ebene geteilt."""
    a = car.axis
    hint = np.eye(3)[int(np.argmin(np.abs(a)))]
    x, y, z = orthonormal_frame(a, hint)
    return x, y, z


def build_regions(model, labels, carriers, log=None):
    """Zusammenhängende Regionen; Rundum-Regionen periodischer Träger an einer Naht teilen.
    Rückgabe: reg (Region je Dreieck), regions = [dict(carrier, faces, split)]"""
    log = log or (lambda s: None)
    nF = model.n_faces
    reg = -np.ones(nF, np.int64)
    regions = []
    for c, car in enumerate(carriers):
        ids = np.nonzero(labels == c)[0]
        if len(ids) == 0:
            continue
        for comp in model.components_of(ids):
            info = {"carrier": c, "faces": comp, "split": None}
            if car.kind == "sphere":
                c0 = np.asarray(car.p["center"], float)
                dirs = unit(model.centroids[comp] - c0)
                mdir = unit(dirs.mean(0))
                if float(np.min(dirs @ mdir)) < math.cos(math.radians(80)):
                    sn = orthonormal_frame(mdir)[0]
                    side = (model.centroids[comp] - c0) @ sn
                    for part in (comp[side >= 0], comp[side < 0]):
                        for sub in model.components_of(part):
                            regions.append({"carrier": c, "faces": sub, "split": (c0, sn)})
                    continue
            elif car.kind in PERIODIC_KINDS:
                nloops = _region_loops_count(model, comp)
                x, y, a = _split_frame(car)
                q = model.centroids[comp] - car.base
                ang = np.arctan2(q @ y, q @ x)
                u0, u1 = angular_range(ang, 25.0)
                full = (u1 - u0) >= 2 * math.pi - 1e-6
                if full and nloops != 1:
                    # in zwei Hälften teilen (gemeinsame Teilungsebene für koaxiale Flächen)
                    h1 = comp[(q @ y) >= 0]
                    h2 = comp[(q @ y) < 0]
                    for part in (h1, h2):
                        for sub in model.components_of(part):
                            regions.append({"carrier": c, "faces": sub, "split": (car.base, y)})
                    continue
                if full and nloops == 1:
                    info["cup"] = True
            regions.append(info)
    for i, r in enumerate(regions):
        reg[r["faces"]] = i
    return reg, regions


# ============================================================================ 2. Topologie
def boundary_chains(model, reg):
    """Ketten entlang der Regionsgrenzen.
    Rückgabe: chains (dict r, s, verts, closed), corner-Menge."""
    F = model.F
    nF, nV = len(F), model.n_verts
    a = F.reshape(-1)
    b = np.roll(F, -1, axis=1).reshape(-1)
    hf = np.repeat(np.arange(nF), 3)
    key = np.minimum(a, b) * nV + np.maximum(a, b)
    order = np.argsort(key, kind="stable")
    ks = key[order]
    starts = np.r_[0, np.nonzero(np.diff(ks))[0] + 1]
    cnts = np.diff(np.r_[starts, len(ks)])
    he = {}        # (left_region, right_region, a) -> b   für Halbkanten a->b mit 'left' links
    vregs = [set() for _ in range(nV)]
    for f in range(nF):
        r = reg[f]
        for v in F[f]:
            vregs[v].add(r)
    for st, c in zip(starts, cnts):
        idx = order[st:st + c]
        if c == 1:
            i = idx[0]
            r = reg[hf[i]]
            he.setdefault((r, -1), {})[a[i]] = b[i]
            vregs[a[i]].add(-1)
            vregs[b[i]].add(-1)
        elif c == 2:
            i, j = idx
            r, s = reg[hf[i]], reg[hf[j]]
            if r != s:
                he.setdefault((r, s), {})[a[i]] = b[i]
                he.setdefault((s, r), {})[a[j]] = b[j]
    corner = np.array([len(s) >= 3 for s in vregs])
    # Grad-Test: Punkte, an denen dieselbe Grenze mehrfach vorbeiläuft
    for (r, s), mp in he.items():
        cnt = {}
        for x, y in mp.items():
            cnt[x] = cnt.get(x, 0) + 1
            cnt[y] = cnt.get(y, 0) + 1
        for v, k in cnt.items():
            if k > 2:
                corner[v] = True
    chains = []
    for (r, s), mp in he.items():
        if s != -1 and r > s:
            continue        # jede Grenze nur einmal (Orientierung der kleineren Region)
        if r == -1:
            continue
        used = set()
        # zuerst von Ecken aus
        for x0 in list(mp.keys()):
            if not corner[x0] or x0 in used:
                continue
            x = x0
            path = [x]
            while True:
                used.add(x)
                y = mp.get(x)
                if y is None:
                    break
                path.append(y)
                if corner[y] or y == x0:
                    break
                x = y
            if len(path) >= 2:
                chains.append({"r": r, "s": s, "verts": np.asarray(path, np.int64), "closed": False})
        # geschlossene Schleifen ohne Ecke
        for x0 in list(mp.keys()):
            if x0 in used:
                continue
            path = [x0]
            x = x0
            ok = True
            while True:
                used.add(x)
                y = mp.get(x)
                if y is None:
                    ok = False
                    break
                if y == x0:
                    break
                if y in used:
                    ok = False
                    break
                path.append(y)
                x = y
            if len(path) >= 3:
                chains.append({"r": r, "s": s, "verts": np.asarray(path, np.int64), "closed": ok})
    return chains, corner


class _UF:
    def __init__(self):
        self.p = {}

    def find(self, x):
        self.p.setdefault(x, x)
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[rb] = ra


def chain_length(model, ch):
    P = model.V[ch["verts"]]
    L = float(np.linalg.norm(np.diff(P, axis=0), axis=1).sum())
    if ch["closed"]:
        L += float(np.linalg.norm(P[-1] - P[0]))
    return L


# ============================================================================ 3. Geometrie
def _surf_for_pair(regions, carriers, r, s):
    ca = carriers[regions[r]["carrier"]]
    if s < 0:
        return ca, None
    cb = carriers[regions[s]["carrier"]]
    if regions[r]["carrier"] == regions[s]["carrier"]:
        sp = regions[r].get("split") or regions[s].get("split")
        if sp is None:
            return ca, None
        cb = Carrier("plane", {"origin": sp[0], "normal": sp[1]})
    return ca, cb


def solve_vertex(x0, surfs, iters=12, reg=1e-3):
    """Punkt nahe x0, der möglichst auf allen Flächen liegt (Gauß-Newton, regularisiert)."""
    x = np.asarray(x0, float).copy()
    for _ in range(iters):
        rows, rhs = [], []
        for s in surfs:
            q = s.project(x[None])[0]
            n = s.normal(q[None])[0]
            d = (x - q) @ n
            rows.append(n)
            rhs.append(-d)
        J = np.asarray(rows)
        r = np.asarray(rhs)
        A = J.T @ J + reg * np.eye(3)
        dx = np.linalg.solve(A, J.T @ r)
        x = x + dx
        if np.linalg.norm(dx) < 1e-9:
            break
    err = max(float(np.linalg.norm(x - s.project(x[None])[0])) for s in surfs)
    return x, err


def _mid_projection(P, ca, cb, iters=25):
    X = np.asarray(P, float).copy()
    if cb is None:
        return ca.project(X)
    for _ in range(iters):
        X = 0.5 * (ca.project(X) + cb.project(X))
    return X


def _intersect_points(P, ca, cb, iters=12, lam=1e-4, max_move=5.0):
    """Punkte auf der Schnittkurve zweier Träger nahe P (Newton mit minimaler Schrittweite:
    dx = -J^T (J J^T + λI)^-1 d, J = [n_a; n_b]). Bei (fast) tangentialen Flächen ohne echten
    Schnitt wird die Mitte der beiden Projektionen genommen."""
    P = np.asarray(P, float)
    if cb is None:
        return ca.project(P)
    X = P.copy()
    for _ in range(iters):
        Qa, Qb = ca.project(X), cb.project(X)
        na, nb = ca.normal(Qa), cb.normal(Qb)
        da = np.einsum("ij,ij->i", X - Qa, na)
        db = np.einsum("ij,ij->i", X - Qb, nb)
        g = np.einsum("ij,ij->i", na, nb)
        det = (1 + lam) ** 2 - g ** 2
        a = ((1 + lam) * da - g * db) / det
        b = ((1 + lam) * db - g * da) / det
        dx = -(a[:, None] * na + b[:, None] * nb)
        lim = 2.0 * np.maximum(np.abs(da), np.abs(db)) + 1e-9
        ln = np.linalg.norm(dx, axis=1)
        dx *= np.minimum(1.0, lim / np.maximum(ln, 1e-300))[:, None]
        X = X + dx
        if float(np.max(np.abs(da)) + np.max(np.abs(db))) < 1e-10:
            break
    ea = np.linalg.norm(X - ca.project(X), axis=1)
    eb = np.linalg.norm(X - cb.project(X), axis=1)
    bad = (np.maximum(ea, eb) > 1e-6) | (np.linalg.norm(X - P, axis=1) > max_move)
    if bad.any():
        X[bad] = _mid_projection(P[bad], ca, cb)
    return X


def _as_line_or_circle(M, eps=2e-4):
    """Erkennt exakte Geraden/Kreise in den Schnittpunkten. Rückgabe ('line', p, d) /
    ('circle', c, n, x, r) oder None."""
    M = np.asarray(M, float)
    if len(M) < 4:
        return None
    c = M.mean(0)
    X = M - c
    _, sv, vt = np.linalg.svd(X, full_matrices=False)
    d = vt[0]
    if np.max(np.linalg.norm(X - np.outer(X @ d, d), axis=1)) < eps:
        return ("line", c, d)
    n = vt[2]
    if np.max(np.abs(X @ n)) > eps:
        return None
    x, y = vt[0], vt[1]
    Q = np.c_[X @ x, X @ y]
    A = np.c_[2 * Q, np.ones(len(Q))]
    sol, *_ = np.linalg.lstsq(A, (Q ** 2).sum(1), rcond=None)
    cc = sol[:2]
    r = float(np.sqrt(max(sol[2] + cc @ cc, 0.0)))
    if r <= 0 or r > 1e5:
        return None
    if np.max(np.abs(np.linalg.norm(Q - cc, axis=1) - r)) > eps:
        return None
    center = c + cc[0] * x + cc[1] * y
    return ("circle", center, n, x, r)


def _curve_from_shape(shp, M, X0, X1, closed):
    """Gerade/Kreis (aus _as_line_or_circle), getrimmt zwischen den Ecken.
    Rückgabe (Geom_Curve, u0, u1, reversed) oder None."""
    from OCP.Geom import Geom_Circle, Geom_Line
    from OCP.gp import gp_Ax2, gp_Circ, gp_Lin
    from . import occ_build as ob
    if shp[0] == "line":
        if closed:
            return None
        _, p, d = shp
        # Gerade exakt durch beide Ecken legen, wenn diese (fast) auf der Schnittgeraden liegen:
        # schneiden sich drei Flächen nicht exakt in einem Punkt (Rundung, Achsen gerade gestellt),
        # liefe die Schnittgerade sonst knapp an der Ecke vorbei und überkreuzte die Nachbarkante
        dv = X1 - X0
        L = float(np.linalg.norm(dv))
        if L > 1e-9 and abs(float(unit(dv) @ unit(d))) > math.cos(math.radians(2.0)):
            dd = unit(dv)
            crv = Geom_Line(gp_Lin(ob.P(X0), ob.D(dd)))
            return crv, 0.0, L, False
        crv = Geom_Line(gp_Lin(ob.P(p), ob.D(d)))
        u0, u1 = float((X0 - p) @ d), float((X1 - p) @ d)
        if abs(u1 - u0) < 1e-9:
            return None
        return (crv, u0, u1, False) if u1 > u0 else (crv, u1, u0, True)
    _, cen, n, x, r = shp
    y = np.cross(n, x)
    crv = Geom_Circle(gp_Circ(gp_Ax2(ob.P(cen), ob.D(n), ob.D(x)), r))
    two = 2 * math.pi

    def par(q):
        v = q - cen
        return math.atan2(float(v @ y), float(v @ x)) % two

    u0 = par(X0)
    k = len(M)
    um = par(M[k // 2])
    if closed:
        # Laufrichtung: steigt der Winkel entlang der Kette?
        du = (par(M[min(2, k - 1)]) - par(M[0]) + math.pi) % two - math.pi
        return crv, u0, u0 + two, du < 0
    u1 = par(X1)
    a = (u1 - u0) % two
    b = (um - u0) % two
    if a < 1e-9:
        return None
    if b <= a:
        return crv, u0, u0 + a, False
    return crv, u1, u1 + (two - a), True


def _tangent_points(S, ca, cb, W, closed, rounds=3, ns=25):
    """Berührkurve zweier (fast) tangentialer Flächen (Rundungsübergang): je Kettenpunkt quer
    zur Kante die Stelle suchen, an der der Spalt zwischen den Flächen extremal ist
    (Scheitel einer Parabel durch den Spaltverlauf), dann auf beide Flächen mitteln."""
    X = _mid_projection(S, ca, cb, iters=5)
    n = len(X)
    if n < 2:
        return X
    T = np.gradient(X, axis=0) if not closed else (np.roll(X, -1, 0) - np.roll(X, 1, 0))
    T = unit(T)
    na = ca.normal(ca.project(X))
    nb = cb.normal(cb.project(X))
    sgn = np.sign(np.einsum("ij,ij->i", na, nb))
    sgn[sgn == 0] = 1
    t = unit(np.cross(na, T))
    off = np.zeros(n)
    grid = np.linspace(-1.0, 1.0, ns)
    w = float(W)
    for _ in range(rounds):
        sg = grid * w
        G = np.empty((ns, n))
        for j, sj in enumerate(sg):
            Y = X + (off + sj)[:, None] * t
            G[j] = ca.distance(Y) - sgn * cb.distance(Y)
        c2, c1, c0 = np.polyfit(sg, G, 2)
        ok = np.abs(c2) > 1e-9
        sv = np.where(ok, -c1 / np.where(ok, 2 * c2, 1.0), 0.0)
        bad = ~ok | (np.abs(sv) > w)
        if bad.any():
            sv[bad] = sg[np.argmin(np.abs(G[:, bad]), axis=0)]
        off += sv
        w *= 0.4
    X = X + off[:, None] * t
    return _mid_projection(X, ca, cb, iters=5)


def _point_on_shape_meeting(shape, x0, surfs, iters=20):
    """Punkt auf Gerade/Kreis, der die übrigen Flächen der Ecke trifft (1D-Gauß-Newton)."""
    if shape[0] == "line":
        _, p, d = shape

        def C(u):
            return p + u * d
        u = float((x0 - p) @ d)
        h = 1e-4
    else:
        _, cen, nrm, xx, r = shape
        yy = np.cross(nrm, xx)

        def C(u):
            return cen + r * (math.cos(u) * xx + math.sin(u) * yy)
        v = x0 - cen
        u = math.atan2(float(v @ yy), float(v @ xx))
        h = 1e-5
    for _ in range(iters):
        x = C(u)
        f = np.array([float(sf.distance(x[None])[0]) for sf in surfs])
        df = (np.array([float(sf.distance(C(u + h)[None])[0]) for sf in surfs]) - f) / h
        den = float(df @ df)
        if den < 1e-14:
            return None
        du = -float(df @ f) / den
        u += du
        if abs(du) < 1e-10:
            break
    x = C(u)
    if max(abs(float(sf.distance(x[None])[0])) for sf in surfs) > 1e-3:
        return None
    return x


def _curve_chain_dev(edge, P, n=16):
    """Größter Abstand von Punkten auf der Kante zur Scan-Grenze (Polylinie P)."""
    from OCP.BRepAdaptor import BRepAdaptor_Curve
    from .geom_utils import dist_points_to_polyline
    ad = BRepAdaptor_Curve(edge)
    us = np.linspace(ad.FirstParameter(), ad.LastParameter(), n)
    Q = np.array([[p.X(), p.Y(), p.Z()] for p in (ad.Value(u) for u in us)])
    Lc = float(np.linalg.norm(np.diff(Q, axis=0), axis=1).sum())
    return float(dist_points_to_polyline(Q, np.asarray(P, float)).max()), Lc


def _bspline_through(points, closed):
    from OCP.GeomAPI import GeomAPI_Interpolate
    from OCP.gp import gp_Pnt
    from .occ_compat import HArray1Pnt
    P = np.asarray(points, float)
    keep = [0]
    for i in range(1, len(P)):
        if np.linalg.norm(P[i] - P[keep[-1]]) > 1e-6:
            keep.append(i)
    P = P[keep]
    if closed and np.linalg.norm(P[0] - P[-1]) < 1e-6:
        P = P[:-1]
    if len(P) < 2 or (closed and len(P) < 3):
        raise RuntimeError("zu wenige Punkte")
    arr = HArray1Pnt(1, len(P))
    for i, q in enumerate(P):
        arr.SetValue(i + 1, gp_Pnt(*map(float, q)))
    it = GeomAPI_Interpolate(arr, bool(closed), 1e-7)
    it.Perform()
    if not it.IsDone():
        raise RuntimeError("Interpolation fehlgeschlagen")
    return it.Curve()


def _resample(P, n, closed):
    P = np.asarray(P, float)
    if closed:
        P = np.vstack([P, P[:1]])
    seg = np.linalg.norm(np.diff(P, axis=0), axis=1)
    s = np.r_[0, np.cumsum(seg)]
    if s[-1] <= 0:
        return P[:1]
    t = np.linspace(0, s[-1], n + 1 if closed else n)
    if closed:
        t = t[:-1]
    return np.column_stack([np.interp(t, s, P[:, d]) for d in range(3)])


def _edge_tolerance(curve, u0, u1, surfs, n=12):
    us = np.linspace(u0, u1, n)
    pts = []
    for u in us:
        p = curve.Value(float(u))
        pts.append((p.X(), p.Y(), p.Z()))
    pts = np.asarray(pts)
    t = 0.0
    for s in surfs:
        if s is None:
            continue
        t = max(t, float(np.max(np.linalg.norm(pts - s.project(pts), axis=1))))
    return t


# ============================================================================ Flächen je Region
def _occ_normals(srf, P):
    from OCP.GeomLProp import GeomLProp_SLProps
    from OCP.gp import gp_Pnt
    from OCP.ShapeAnalysis import ShapeAnalysis_Surface
    sas = ShapeAnalysis_Surface(srf)
    out = np.zeros((len(P), 3))
    for i, q in enumerate(P):
        uv = sas.ValueOfUV(gp_Pnt(*map(float, q)), 1e-6)
        pr = GeomLProp_SLProps(srf, uv.X(), uv.Y(), 1, 1e-7)
        if pr.IsNormalDefined():
            n = pr.Normal()
            out[i] = (n.X(), n.Y(), n.Z())
    return out


def _circ_mean(a):
    return math.atan2(float(np.sin(a).mean()), float(np.cos(a).mean()))


def _region_surface(car, C, Cs, Ns):
    """Trägerfläche für eine Region: Naht periodischer Flächen gegenüber der Region legen (sonst
    kreuzen die Parameterkurven die Naht), Kugelpole aus der Region heraus drehen, Normale nach
    außen (wie das Netz) ausrichten."""
    from OCP.Geom import Geom_Circle, Geom_SurfaceOfRevolution
    from OCP.gp import gp_Ax1, gp_Ax2, gp_Circ
    from . import occ_build as ob
    k = car.kind
    srf = None
    if k in ("cylinder", "cone", "torus"):
        a = car.axis
        x, y, _ = orthonormal_frame(a, car.p.get("xdir"))
        q = C - car.base
        seam = _circ_mean(np.arctan2(q @ y, q @ x)) + math.pi
        xd = math.cos(seam) * x + math.sin(seam) * y
        p = dict(car.p)
        p["xdir"] = xd
        if k == "torus":
            h = q @ a
            e = q - np.outer(h, a)
            rho = np.linalg.norm(e, axis=1)
            v = np.arctan2(h, rho - p["R"])
            v0, v1 = angular_range(v, 25.0)
            crosses = (v1 - v0) >= 2 * math.pi - 1e-6 or (v0 < 0 < v1) or (v0 < 2 * math.pi < v1)
            if crosses:
                vs = _circ_mean(v) + math.pi
                tc = car.base + p["R"] * xd
                nc = np.cross(xd, a)            # Kreisebene = Meridianebene (xd, a)
                cx = math.cos(vs) * xd + math.sin(vs) * a
                circ = Geom_Circle(gp_Circ(gp_Ax2(ob.P(tc), ob.D(nc), ob.D(cx)), float(p["r"])))
                srf = Geom_SurfaceOfRevolution(circ, gp_Ax1(ob.P(car.base), ob.D(a)))
        if srf is None:
            srf = Carrier(k, p).geom()
    elif k == "sphere":
        c = np.asarray(car.p["center"], float)
        m = unit(unit(C - c).mean(0))
        z, _, _ = orthonormal_frame(m)
        p = dict(car.p)
        p["zdir"], p["xdir"] = z, -m
        srf = Carrier(k, p).geom()
    else:
        srf = car.geom()
    try:
        nn = _occ_normals(srf, Cs)
        if float(np.einsum("ij,ij->i", nn, Ns).sum()) < 0:
            srf = srf.Reversed()
    except Exception:
        pass
    return srf


def _face_area(face):
    from OCP.BRepGProp import BRepGProp
    from OCP.GProp import GProp_GProps
    g = GProp_GProps()
    BRepGProp.SurfaceProperties_s(face, g)
    return float(g.Mass())


def _make_face(srf, wires, area_expected):
    """Fläche aus Träger + Drähten. Auf geschlossenen Flächen (Kugel, Torus) ist 'innen/außen'
    nur über die Drahtrichtung bestimmt; ShapeFix kann das umdrehen -> Varianten probieren und
    die nehmen, deren Flächeninhalt zur Netzregion passt."""
    from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeFace
    from OCP.ShapeFix import ShapeFix_Face
    from .occ_compat import to_wire

    def build(ws, fix_orient):
        mf = BRepBuilderAPI_MakeFace(srf, ws[0], True)
        for w in ws[1:]:
            mf.Add(w)
        if not mf.IsDone():
            raise RuntimeError("MakeFace")
        fx = ShapeFix_Face(mf.Face())
        fx.FixAddNaturalBoundMode = 0
        fx.FixSplitFaceMode = 0
        if not fix_orient:
            fx.FixOrientationMode = 0
        fx.Perform()
        return fx.Face()

    rev = [to_wire(w.Reversed()) for w in wires]
    best = None
    for ws, fo in ((wires, True), (wires, False), (rev, False), (rev, True)):
        try:
            f = build(ws, fo)
            a = _face_area(f)
        except Exception:
            continue
        ratio = a / max(area_expected, 1e-9)
        score = abs(math.log(max(ratio, 1e-9)))
        if best is None or score < best[0]:
            best = (score, f)
        if 0.8 < ratio < 1.25:
            break
    if best is None:
        raise RuntimeError("MakeFace")
    return best[1]


# ============================================================================ Gesamtaufbau
def build_closed_brep(model, features, tol=0.05, min_faces=25, log=None, progress=None, passive=None, rescue=True,
                      strips=None):
    """Rückgabe dict: shape (Solid/Schale), solid (bool), report (Liste), open_edges (Polylinien),
    labels (Region je Dreieck), regions, stats.

    strips: Listen von Dreiecks-Indizes, die als eigene Übergangsflächen (Freiform zwischen zwei
    Feature-Flächen, z. B. Verrundung) gebaut werden – die Feature-Flächen daneben bleiben exakt.
    Ohne Angabe ermittelt die Rettung solche Streifen selbst, wenn die Hülle nicht schließt."""
    from OCP.BRep import BRep_Builder
    from OCP.BRepBuilderAPI import (BRepBuilderAPI_MakeEdge, BRepBuilderAPI_MakeFace,
                                    BRepBuilderAPI_MakeVertex, BRepBuilderAPI_MakeWire)
    from OCP.Geom import Geom_TrimmedCurve
    from OCP.gp import gp_Pnt
    from OCP.ShapeFix import ShapeFix_Face
    from OCP.TopoDS import TopoDS_Shell
    from . import occ_build as ob
    from .occ_compat import to_edge, to_face

    report = []
    log_ = (lambda s: (report.append(s), log and log(s)))
    prog = progress or (lambda s: None)
    edge_len = model.mean_edge_length()
    if model.boundary_edge_count:
        log_(tr(f"Netz hat {model.boundary_edge_count} offene Randkanten – Hülle wird dort offen bleiben "
                "(Löcher/Boden vorher schließen)",
                f"mesh has {model.boundary_edge_count} open boundary edges – the shell will stay open there "
                "(close holes/base first)"))
    prog(tr("Regionen zuordnen…", "Assigning regions…"))
    labels, carriers, cfeat, absorbed = assign_regions(model, features, tol, min_faces, log=log_,
                                                       passive=passive)
    if strips:
        labels = labels.copy()
        for ids in strips:
            ids = np.asarray(ids, np.int64)
            if len(ids) < 3:
                continue
            f = _transition_feature(model, ids)
            car = carrier_of(f)
            car.transition = True
            labels[ids] = len(carriers)
            carriers.append(car)
            cfeat.append(f)
    reg, regions = build_regions(model, labels, carriers, log_)
    reg, npinch = fix_pinches(model, reg)
    nsmall = _absorb_small_regions(model, reg, regions, max(4, min_faces // 3))
    if npinch or nsmall:
        for i, rg in enumerate(regions):
            rg["faces"] = np.nonzero(reg == i)[0]
        regions_ok = [i for i, rg in enumerate(regions) if len(rg["faces"])]
        if len(regions_ok) < len(regions):
            remap = -np.ones(len(regions), np.int64)
            remap[regions_ok] = np.arange(len(regions_ok))
            regions = [regions[i] for i in regions_ok]
            reg = remap[reg]
    cups = [i for i, r in enumerate(regions) if r.get("cup")]
    if cups:
        names = ", ".join(sorted({carriers[regions[i]['carrier']].name for i in cups}))
        log_(tr(f"⚠ Napfförmige Bereiche ohne Boden ({names}) – dafür besser eine Bohrung definieren",
                f"⚠ cup-shaped regions without a bottom ({names}) – better define a hole"))
    prog(tr("Kanten und Ecken bestimmen…", "Computing edges and vertices…"))
    chains, corner = boundary_chains(model, reg)
    F = model.F
    vfaces = {}
    cv = np.nonzero(corner)[0]
    if len(cv):
        cmask = np.zeros(model.n_verts, bool)
        cmask[cv] = True
        for f in np.nonzero(cmask[F].any(1))[0]:
            for v in F[f]:
                if cmask[v]:
                    vfaces.setdefault(int(v), []).append(int(f))

    def regs_of(members):
        rr = set()
        for v in members:
            for f in vfaces.get(v, []):
                rr.add(int(reg[f]))
        rr.discard(-1)
        return rr

    def surfs_for(regs):
        """Träger der Regionen an einer Ecke (+ Nahtebenen geteilter periodischer Regionen)."""
        surfs = []
        seen_c = {}
        for r in sorted(regs):
            c = regions[r]["carrier"]
            if c in seen_c:
                sp = regions[r].get("split") or regions[seen_c[c]].get("split")
                if sp is not None:
                    sn_ = unit(sp[1])
                    dup = any(sf.kind == "plane" and abs(float(unit(sf.p["normal"]) @ sn_)) > math.cos(math.radians(2))
                              and abs(float((np.asarray(sp[0]) - sf.p["origin"]) @ sn_)) < 3 * tol for sf in surfs)
                    if not dup:
                        surfs.append(Carrier("plane", {"origin": sp[0], "normal": sp[1]}))
                continue
            seen_c[c] = r
            surfs.append(carriers[c])
        return surfs

    # kurze Ketten zusammenziehen – aber nur, wenn die zusammengelegte Ecke geometrisch möglich ist
    # (alle beteiligten Flächen haben dort einen gemeinsamen Punkt). Sonst würden z. B. die beiden
    # Seiten einer dünnen Platte in einer Ecke landen.
    el_med = float(np.median(np.linalg.norm(model.V[F[:, 1]] - model.V[F[:, 0]], axis=1)))
    lmin = max(2.5 * min(edge_len, el_med), 3 * tol)
    uf = _UF()
    for ch in chains:
        if not ch["closed"]:
            uf.find(int(ch["verts"][0]))
            uf.find(int(ch["verts"][-1]))
    members_of = {}
    for ch in chains:
        if not ch["closed"]:
            for v in (int(ch["verts"][0]), int(ch["verts"][-1])):
                members_of.setdefault(uf.find(v), set()).add(v)
    collapsed = 0
    for ch in sorted((c for c in chains if not c["closed"]), key=lambda c: chain_length(model, c)):
        if chain_length(model, ch) >= lmin:
            break
        a, b = uf.find(int(ch["verts"][0])), uf.find(int(ch["verts"][-1]))
        if a == b:
            ch["collapsed"] = True
            collapsed += 1
            continue
        mem = members_of.get(a, {a}) | members_of.get(b, {b})
        surfs = surfs_for(regs_of(mem))
        if surfs:
            x0 = model.V[list(mem)].mean(0)
            x, e = solve_vertex(x0, surfs)
            if e > 3 * tol or np.linalg.norm(x - x0) > max(20 * tol, 4 * edge_len):
                continue
        uf.union(a, b)
        root = uf.find(a)
        members_of[root] = mem
        ch["collapsed"] = True
        collapsed += 1
    live = [ch for ch in chains if not ch.get("collapsed")]
    # Cluster -> Regionen und Lage
    clusters = {}
    for ch in chains:
        if ch["closed"]:
            continue
        for v in (int(ch["verts"][0]), int(ch["verts"][-1])):
            c = uf.find(v)
            clusters.setdefault(c, set()).add(v)
    for ch in live:
        if ch["closed"]:
            v = int(ch["verts"][0])
            clusters.setdefault(("loop", id(ch)), {v})
    vregs_of = {key: regs_of(members) for key, members in clusters.items()}
    ff0 = frozenset(i for i, rg in enumerate(regions) if getattr(carriers[rg["carrier"]], "transition", False))

    def attempt(ff):
        """Ecken, Kanten, Flächen, Schale. ff: Regionen, die als Freiform durch ihre Randkanten
        aufgebaut werden (Träger passt nicht zur Topologie)."""
        vpos = {}
        vsurfs = {}
        verr = 0.0
        bad_vertices = []
        bad_vkeys = []
        for key, members in clusters.items():
            if isinstance(key, tuple):
                continue
            rr_all = vregs_of[key]
            surfs = surfs_for(rr_all - ff) if ff else surfs_for(rr_all)
            x0 = model.V[list(members)].mean(0)
            if len(surfs) >= 1:
                x, e = solve_vertex(x0, surfs)
                if np.linalg.norm(x - x0) > max(20 * tol, 4 * edge_len):
                    x, e = x0, float(np.linalg.norm(x - x0))
                verr = max(verr, e)
                if e > 3 * tol:
                    bad_vertices.append((x0, e, [sf.name or sf.kind for sf in surfs]))
                    bad_vkeys.append((key, x0, e))
            else:
                x = x0
            vpos[key] = x
            vsurfs[key] = surfs
        occ_vert = {}

        def vertex_of(key):
            if key not in occ_vert:
                occ_vert[key] = BRepBuilderAPI_MakeVertex(ob.P(vpos[key])).Vertex()
            return occ_vert[key]

        prog(tr("Kanten konstruieren…", "Building edges…"))
        # Pass 1: Kurvenpunkte je Kette (Schnitt- bzw. Berührkurve), Form erkennen
        geo = []
        for ch in live:
            r, s_ = ch["r"], ch["s"]
            ca, cb = _surf_for_pair(regions, carriers, r, s_)
            P = model.V[ch["verts"]]
            closed = ch["closed"]
            if closed:
                k0 = k1 = ("loop", id(ch))
            else:
                k0 = uf.find(int(ch["verts"][0]))
                k1 = uf.find(int(ch["verts"][-1]))
            L = chain_length(model, ch)
            n = int(np.clip(L / max(2 * edge_len, 1e-3), 8, 80))
            S = _resample(P, n, closed)
            tangent = False
            guided = (r in ff) or (s_ >= 0 and s_ in ff)
            if guided:
                # Kante entlang der (geglätteten) Scan-Grenze, auf die gültige Nachbarfläche gelegt
                Ss = _smooth_polyline(S, closed)
                good = [x for x in (r, s_) if x >= 0 and x not in ff]
                tcar = [carriers[regions[good[0]]["carrier"]]] if good else []
                M = tcar[0].project(Ss) if tcar else Ss
                shape = None
            elif cb is not None:
                tcar = [ca, cb]
                M = _intersect_points(S, ca, cb, max_move=max(2.0, 10 * tol))
                try:
                    g = np.abs(np.einsum("ij,ij->i", ca.normal(ca.project(M)), cb.normal(cb.project(M))))
                    tangent = float(np.median(g)) > math.cos(math.radians(12))
                except Exception:
                    tangent = False
                if tangent:
                    M = _tangent_points(S, ca, cb, max(2.0, 6 * edge_len), closed)
                shape = _as_line_or_circle(M, 0.25 * tol if tangent else 2e-4)
            else:
                tcar = [ca]
                M = ca.project(S)
                shape = None
            if closed:
                vpos[k0] = M[0]
            geo.append({"ch": ch, "ca": ca, "cb": cb, "M": M, "shape": shape, "tangent": tangent,
                        "closed": closed, "k0": k0, "k1": k1, "P": P, "S": S, "guided": guided, "tcar": tcar})
        # Ecken an Berührkanten (Rundungen) auf die Berührgerade/-kreis setzen
        for gd in geo:
            if not gd["tangent"] or gd["shape"] is None or gd["closed"]:
                continue
            for key in (gd["k0"], gd["k1"]):
                others = [sf for sf in vsurfs.get(key, []) if sf is not gd["ca"] and sf is not gd["cb"]]
                if not others:
                    continue
                x = _point_on_shape_meeting(gd["shape"], vpos[key], others)
                if x is not None and np.linalg.norm(x - vpos[key]) < max(10 * tol, 3 * edge_len):
                    vpos[key] = x
        occ_vert.clear()
        # Pass 2: OCC-Kanten
        builder = BRep_Builder()
        vtol = {}
        edges = []         # (chain, TopoDS_Edge (in Kettenrichtung), k0, k1, tol)
        n_exact = n_fit = n_guided = 0
        max_etol = 0.0
        bad_edges = []

        def etol_of(crv, u0, u1, tc):
            return _edge_tolerance(crv, u0, u1, tc) if tc else 1e-4

        for gd in geo:
            ch, ca, cb, M, closed = gd["ch"], gd["ca"], gd["cb"], gd["M"].copy(), gd["closed"]
            tc = gd["tcar"]
            k0, k1 = gd["k0"], gd["k1"]
            X0, X1 = vpos[k0], vpos[k1]
            e = None
            res = _curve_from_shape(gd["shape"], M, X0, X1, closed) if gd["shape"] is not None else None
            if res is not None:
                crv, u0, u1, rev = res
                try:
                    V0, V1 = vertex_of(k0), vertex_of(k1)
                    p_a, p_b = crv.Value(u0), crv.Value(u1)
                    pa = np.array([p_a.X(), p_a.Y(), p_a.Z()])
                    pb = np.array([p_b.X(), p_b.Y(), p_b.Z()])
                    vs, ve = (V1, V0) if rev else (V0, V1)
                    xs, xe = (X1, X0) if rev else (X0, X1)
                    ks, ke = (k1, k0) if rev else (k0, k1)
                    # Toleranz einer Ecke = größter Abstand zu den Enden aller ihrer Kanten (nie verkleinern)
                    vtol[ks] = max(vtol.get(ks, 1e-6), float(np.linalg.norm(pa - xs)) * 1.5)
                    vtol[ke] = max(vtol.get(ke, 1e-6), float(np.linalg.norm(pb - xe)) * 1.5)
                    builder.UpdateVertex(vs, vtol[ks])
                    builder.UpdateVertex(ve, vtol[ke])
                    me = BRepBuilderAPI_MakeEdge(crv, vs, ve, u0, u1)
                    if me.IsDone():
                        e = me.Edge()
                        if rev:
                            e = to_edge(e.Reversed())
                        et = etol_of(crv, u0, u1, tc)
                        n_exact += 1
                except Exception:
                    e = None
            if e is None:
                try:
                    if not closed:
                        M = np.vstack([X0, M[1:-1], X1]) if len(M) > 2 else np.vstack([X0, X1])
                    else:
                        M[0] = vpos[k0]
                    crv = _bspline_through(M, closed)
                    u0, u1 = crv.FirstParameter(), crv.LastParameter()
                    V0, V1 = vertex_of(k0), vertex_of(k1)
                    me = BRepBuilderAPI_MakeEdge(crv, V0, V1, u0, u1)
                    if not me.IsDone():
                        raise RuntimeError("MakeEdge")
                    e = me.Edge()
                    et = etol_of(crv, u0, u1, tc)
                    if gd["guided"]:
                        n_guided += 1
                    else:
                        n_fit += 1
                except Exception:
                    bad_edges.append(gd["P"])
                    continue
            # Plausibilität: die Kurve muss in der Nähe der Scan-Grenze verlaufen (sonst z. B. der falsche
            # Bogen eines riesigen Kreises oder ein Schnitt weit außerhalb) → Kante entlang der Grenze
            # (scharfe Kanten liegen bei verrundetem Scan legitim bis einige mm neben der Grenze)
            lim = max(4.0, 40 * tol, 6 * edge_len)
            try:
                dev, Lc = (0.0, 0.0) if gd["guided"] else _curve_chain_dev(e, gd["P"])
            except Exception:
                dev, Lc = 0.0, 0.0
            Lch = chain_length(model, ch)
            if dev > lim or Lc > 3.0 * Lch + 2 * lim:
                try:
                    Sg = _mid_projection(gd["S"], ca, cb)
                    if np.median(np.linalg.norm(Sg - gd["S"], axis=1)) > lim:
                        Sg = gd["S"].copy()
                    if not closed:
                        Sg = np.vstack([X0, Sg[1:-1], X1]) if len(Sg) > 2 else np.vstack([X0, X1])
                    else:
                        Sg[0] = vpos[k0]
                    crv = _bspline_through(Sg, closed)
                    u0, u1 = crv.FirstParameter(), crv.LastParameter()
                    me = BRepBuilderAPI_MakeEdge(crv, vertex_of(k0), vertex_of(k1), u0, u1)
                    if me.IsDone():
                        e = me.Edge()
                        et = etol_of(crv, u0, u1, tc)
                        res = None
                        n_guided += 1
                except Exception:
                    pass
            et = max(et, 1e-7)
            builder.UpdateEdge(e, et * 1.2)
            max_etol = max(max_etol, et)
            edges.append({"ch": ch, "edge": e, "k0": k0, "k1": k1, "tol": et, "exact": res is not None,
                          "tangent": gd["tangent"], "guided": gd["guided"]})
        # Flächen
        prog(tr("Flächen bauen…", "Building faces…"))
        by_region = {}
        for ed in edges:
            ch = ed["ch"]
            by_region.setdefault(ch["r"], []).append((ed, True))
            if ch["s"] >= 0:
                by_region.setdefault(ch["s"], []).append((ed, False))
        faces = []
        face_fail = []
        invalid_faces = []
        face_diag = {}
        ff_fallback = []
        ff_ok = 0
        from OCP.BRepCheck import BRepCheck_Analyzer
        for ri, rg in enumerate(regions):
            hes = by_region.get(ri, [])
            if not hes:
                face_fail.append(ri)
                continue
            car = carriers[rg["carrier"]]
            # Halbkanten zu Schleifen
            items = []
            for ed, fwd in hes:
                ch = ed["ch"]
                if fwd:
                    items.append({"e": ed["edge"], "a": ed["k0"], "b": ed["k1"],
                                  "va": int(ch["verts"][0]), "vb": int(ch["verts"][-1]), "len": chain_length(model, ch)})
                else:
                    items.append({"e": to_edge(ed["edge"].Reversed()), "a": ed["k1"], "b": ed["k0"],
                                  "va": int(ch["verts"][-1]), "vb": int(ch["verts"][0]), "len": chain_length(model, ch)})
            used = [False] * len(items)
            loops = []
            for i0 in range(len(items)):
                if used[i0]:
                    continue
                loop = [i0]
                used[i0] = True
                cur = items[i0]
                guard = 0
                while cur["b"] != items[i0]["a"] and guard < len(items) + 2:
                    guard += 1
                    cands = [j for j, it in enumerate(items) if not used[j] and it["a"] == cur["b"]]
                    if not cands:
                        break
                    if len(cands) > 1:
                        cands.sort(key=lambda j: np.linalg.norm(model.V[items[j]["va"]] - model.V[cur["vb"]]))
                    j = cands[0]
                    used[j] = True
                    loop.append(j)
                    cur = items[j]
                loops.append(loop)
            wires = []
            for loop in loops:
                mw = BRepBuilderAPI_MakeWire()
                for j in loop:
                    mw.Add(items[j]["e"])
                if mw.IsDone():
                    L = sum(items[j]["len"] for j in loop)
                    wires.append((L, mw.Wire()))
            if not wires:
                face_fail.append(ri)
                continue
            wires.sort(key=lambda t: -t[0])
            try:
                fids = rg["faces"]
                sample = fids[np.linspace(0, len(fids) - 1, min(len(fids), 40)).astype(int)]
                srf = None
                if ri in ff:
                    srf = _constrained_bspline(model, fids, [ed["edge"] for ed, _ in hes], tol, edge_len, car)
                    if srf is not None:
                        try:
                            nn = _occ_normals(srf, model.centroids[sample])
                            if float(np.einsum("ij,ij->i", nn, model.normals[sample]).sum()) < 0:
                                srf = srf.Reversed()
                        except Exception:
                            pass
                        ff_ok += 1
                if srf is None:
                    if ri in ff:
                        ff_fallback.append(ri)
                    srf = _region_surface(car, model.centroids[fids], model.centroids[sample], model.normals[sample])
                face = _make_face(srf, [w for _, w in wires], float(model.areas[fids].sum()))
                faces.append((ri, face))
                face_diag[ri] = (len(wires), len(items), _face_area(face) / max(float(model.areas[fids].sum()), 1e-9),
                                 ri in ff, srf.__class__.__name__)
                if not BRepCheck_Analyzer(face).IsValid():
                    invalid_faces.append(ri)
            except Exception:
                face_fail.append(ri)
        # Schale
        prog(tr("Schale schließen…", "Closing shell…"))
        from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeSolid
        from OCP.ShapeFix import ShapeFix_Shell, ShapeFix_Solid
        shell = TopoDS_Shell()
        builder.MakeShell(shell)
        for _, f in faces:
            builder.Add(shell, f)
        shape = shell
        solid_ok = False
        free = _free_edges(shell)
        if free > 0:
            try:
                fs = ShapeFix_Shell(shell)
                fs.Perform()
                shell = fs.Shell()
                free = _free_edges(shell)
            except Exception:
                pass
        if free > 0 and faces:
            # Rückfall: Nähen (gleiche Kurven -> trivial, repariert nur fehlende gemeinsame Nutzung)
            from OCP.TopAbs import TopAbs_SHELL
            from OCP.TopExp import TopExp_Explorer
            from .occ_compat import to_shell
            for src in ([f for _, f in faces], [shell]):
                try:
                    sewn = ob.sew(src, min(max(max_etol * 2, tol, 1e-3), 2.0))
                    ex = TopExp_Explorer(sewn, TopAbs_SHELL)
                    shells = []
                    while ex.More():
                        shells.append(to_shell(ex.Current()))
                        ex.Next()
                    if len(shells) == 1:
                        f2 = _free_edges(shells[0])
                        if f2 < free:
                            shell, free = shells[0], f2
                except Exception:
                    pass
                if free == 0:
                    break
        try:
            fs = ShapeFix_Shell(shell)
            fs.Perform()
            shell = fs.Shell()
        except Exception:
            pass
        free = _free_edges(shell)
        if free == 0:
            try:
                sol = BRepBuilderAPI_MakeSolid(shell).Solid()
                fsol = ShapeFix_Solid(sol)
                fsol.Perform()
                sol = fsol.Solid()
                sol = ob.fix_shape(sol)
                if ob.is_valid(sol) and _volume(sol) > 0:
                    shape = sol
                    solid_ok = True
                else:
                    shape = shell
            except Exception:
                shape = shell
        else:
            shape = shell
        return {"shape": shape, "solid": solid_ok, "free": free, "faces": faces, "edges": edges,
                "face_fail": face_fail, "invalid_faces": invalid_faces, "bad_vertices": bad_vertices,
                "bad_vkeys": bad_vkeys, "bad_edges": bad_edges, "n_exact": n_exact, "n_fit": n_fit,
                "n_guided": n_guided, "max_etol": max_etol, "verr": verr, "vpos": vpos, "ff": ff, "ff_ok": ff_ok,
                "face_diag": face_diag, "ff_fallback": ff_fallback}

    A = attempt(ff0)
    if not A["solid"] and rescue and strips is None and A["faces"]:
        # Übergangsstreifen: zwischen zwei Feature-Flächen, deren Schnittkante nicht zum Scan passt
        # (Verrundung, Fase, Rundung, die kein Feature ist), einen eigenen Freiform-Übergang
        # einsetzen – die Feature-Flächen bleiben exakt
        try:
            S = _transition_strips(model, reg, regions, carriers, A, vregs_of, tol, edge_len, min_faces)
        except Exception as ex:  # noqa: BLE001
            S = []
            log_(tr(f"Übergangsstreifen nicht bestimmbar: {ex}", f"transition strips failed: {ex}"))
        if S:
            log_(tr(f"{len(S)} Übergangsstreifen zwischen Flächen, deren Schnittkante nicht zum Scan passt "
                    "(Verrundungen) – als Freiform-Übergang, die Flächen daneben bleiben exakt",
                    f"{len(S)} transition strips between faces whose intersection does not match the scan "
                    "(blends) – built as freeform transitions, the faces beside stay exact"))
            try:
                B = build_closed_brep(model, features, tol, min_faces, log=log_, progress=progress,
                                      passive=passive, rescue=True, strips=S)
            except Exception as ex:  # noqa: BLE001
                B = None
                log_(tr(f"Übergangsstreifen fehlgeschlagen: {ex}", f"transition strips failed: {ex}"))
            if B is not None and B["solid"]:
                B["report"] = report
                return B
            log_(tr("Mit Übergangsstreifen nicht geschlossen – Rettung über Freiform/Facetten der Flächen",
                    "not closed with transition strips – rescue via freeform/facets of the faces"))
    if not A["solid"] and rescue and A["faces"]:
        # Rettung: Bereiche, deren Träger die Topologie nicht trägt (ungültige Flächen, Kanten weit
        # weg von der Scan-Grenze, Ecken ohne gemeinsamen Punkt), als Freiform durch ihre Randkanten
        ff = ff0
        for rnd in range(3):
            bad = _blame(model, regions, carriers, A, vregs_of, tol) - ff
            if not bad:
                break
            ff = ff | bad
            log_(tr(f"Rettung {rnd + 1}: {len(ff)} Bereiche als Freiform durch ihre Randkanten",
                    f"rescue {rnd + 1}: {len(ff)} areas as freeform through their boundary edges"))
            try:
                B = attempt(ff)
            except Exception as ex:  # noqa: BLE001
                log_(tr(f"Rettung fehlgeschlagen: {ex}", f"rescue failed: {ex}"))
                break
            def score(X):
                return X["free"] + 3 * len(X["invalid_faces"]) + 5 * len(X["face_fail"])
            if B["solid"] or score(B) < score(A):
                A = B
            if A["solid"]:
                break
    def try_facets(A, extra, need_solid):
        # Facettieren; bleiben Lücken, die Bereiche an den offenen Kanten dazunehmen (im Grenzfall
        # wird alles zum Scan-Netz – das ist geschlossen)
        from scipy.spatial import cKDTree as _KD
        tree_c = None
        extra = set(int(x) for x in extra)
        Cf = None
        for it in range(4):
            try:
                Ci = _facet_fallback(model, reg, regions, A, uf, live, chains, tol, extra=extra)
            except Exception:  # noqa: BLE001
                Ci = None
            if Ci is None:
                break
            if Cf is None or Ci["solid"] or Ci["free"] < Cf["free"]:
                Cf = Ci
            if Ci["solid"] or Ci["free"] == 0:
                break
            try:
                if tree_c is None:
                    tree_c = _KD(model.centroids)
                pls = _free_edge_polylines(Ci["shape"])
                add = set()
                for pl in pls:
                    pl = np.asarray(pl)
                    _, idx = tree_c.query(pl[len(pl) // 2], k=6)
                    add.update(int(reg[i]) for i in np.atleast_1d(idx) if reg[i] >= 0)
                add -= extra | set(Ci["faceted"])
                if not add:
                    break
                extra |= add | set(Ci["faceted"])
            except Exception:  # noqa: BLE001
                break
        if Cf is None:
            return A, False
        if Cf["solid"] or (not need_solid and Cf["free"] < A["free"]):
            log_(tr(f"{len(Cf['faceted'])} Bereiche facettiert (Dreiecke, Rand auf den Nachbarkanten)",
                    f"{len(Cf['faceted'])} areas faceted (triangles, boundary on the neighbouring edges)"))
            if Cf["solid"] and not Cf.get("valid", True):
                log_(tr("⚠ Körper geschlossen, die Formprüfung (BRepCheck) meldet aber Fehler – im CAD ggf. reparieren",
                        "⚠ solid closed, but the shape check (BRepCheck) reports errors – repair in CAD if needed"))
            return dict(A, shape=Cf["shape"], solid=Cf["solid"], free=Cf["free"], vol=Cf.get("vol"),
                        faceted=sorted(set(A.get("faceted", [])) | set(Cf["faceted"]))), True
        return A, False

    if not A["solid"] and rescue and A["faces"]:
        # letzte Stufe: Flächen mit offenen Kanten bzw. ungültige als Facetten, deren Rand exakt auf
        # den Nachbarkanten liegt, und vernähen
        A, _ = try_facets(A, (), need_solid=False)
    if rescue and A["ff"] and A["solid"]:
        # Qualität der Freiformflächen: Ausbeulungen weit weg vom Scan → zusätzlich facettieren,
        # wenn der Körper dabei geschlossen bleibt
        try:
            spiky = _deviating_faces(model, A["faces"], A["ff"], max(1.0, 10 * tol))
        except Exception:
            spiky = []
        if spiky:
            A2, ok2 = try_facets(A, spiky, need_solid=True)
            if ok2:
                A = A2
    shape, solid_ok, free, faces, edges = A["shape"], A["solid"], A["free"], A["faces"], A["edges"]
    face_fail, invalid_faces, bad_vertices, bad_edges = A["face_fail"], A["invalid_faces"], A["bad_vertices"], A["bad_edges"]
    n_exact, n_fit, max_etol, verr, vpos = A["n_exact"], A["n_fit"], A["max_etol"], A["verr"], A["vpos"]
    if A["ff"]:
        log_(tr(f"{A['ff_ok']} Bereiche als Freiform (Kanten entlang der Scan-Grenze)",
                f"{A['ff_ok']} areas as freeform (edges along the scan boundary)"))
    open_polys = _free_edge_polylines(shape) if not solid_ok else []
    stats = {"regions": len(regions), "faces": len(faces), "edges": len(edges), "exact_edges": n_exact,
             "fitted_edges": n_fit, "guided_edges": A["n_guided"], "freeform_regions": len(A["ff"]),
             "collapsed": collapsed, "max_edge_tol": max_etol, "vertex_err": verr,
             "free_edges": free, "failed_faces": len(face_fail), "invalid_faces": invalid_faces, "bad_vertices": bad_vertices, "failed_edges": len(bad_edges),
             "solid": solid_ok, "volume": (A.get("vol") or _volume(shape)) if solid_ok else None,
             "edge_info": [(carriers[regions[ed["ch"]["r"]]["carrier"]].name,
                            carriers[regions[ed["ch"]["s"]]["carrier"]].name if ed["ch"]["s"] >= 0 else None,
                            float(ed["tol"]), bool(ed["exact"]), bool(ed["tangent"])) for ed in edges]}
    log_(tr(f"Scan-Topologie: {len(faces)} Flächen, {len(edges)} Kanten ({n_exact} exakt, {n_fit} angenähert"
            + (f", {A['n_guided']} entlang der Scan-Grenze" if A["n_guided"] else "") + f"), "
            f"max. Kantentoleranz {max_etol:.3f} mm",
            f"scan topology: {len(faces)} faces, {len(edges)} edges ({n_exact} exact, {n_fit} approximated"
            + (f", {A['n_guided']} along the scan boundary" if A["n_guided"] else "") + f"), "
            f"max edge tolerance {max_etol:.3f} mm"))
    if face_fail:
        log_(tr(f"⚠ {len(face_fail)} Flächen konnten nicht gebaut werden", f"⚠ {len(face_fail)} faces failed"))
    if bad_edges:
        log_(tr(f"⚠ {len(bad_edges)} Kanten konnten nicht gebaut werden", f"⚠ {len(bad_edges)} edges failed"))
    if solid_ok:
        log_(tr(f"✔ Geschlossener Volumenkörper, Volumen {stats['volume']:.1f} mm³",
                f"✔ closed solid, volume {stats['volume']:.1f} mm³"))
    else:
        log_(tr(f"✘ Hülle nicht geschlossen ({free} freie Kanten, rot markiert)",
                f"✘ shell not closed ({free} free edges, marked red)"))
    return {"shape": shape, "solid": solid_ok, "report": report, "open_edges": open_polys + bad_edges,
            "labels": reg, "regions": regions, "edges": edges, "face_fail": face_fail, "vpos": vpos, "stats": stats,
            "carriers": carriers, "cfeat": cfeat, "freeform_regions": sorted(A["ff"]), "face_diag": A["face_diag"],
            "transition_regions": sorted(ff0), "faceted_regions": sorted(A.get("faceted", []))}


def _deviating_faces(model, faces, ff, lim):
    """Freiform-Regionen, deren Fläche irgendwo weiter als lim vom Scan abweicht."""
    import pyvista as pv
    from . import occ_build as ob
    from .preprocess import vf_to_pd
    tgt = vf_to_pd(model.V, model.F)
    out = []
    for ri, f in faces:
        if ri not in ff:
            continue
        try:
            V, F = ob.tessellate(f, max(0.1, 0.5 * lim), 0.3)
        except Exception:
            continue
        if len(V) == 0:
            continue
        d = np.abs(np.asarray(pv.PolyData(V).compute_implicit_distance(tgt).point_data["implicit_distance"]))
        if float(d.max()) > lim:
            out.append(int(ri))
    return out


def _facet_fallback(model, reg, regions, A, uf, live, chains, tol, extra=()):
    """Regionen mit offenen Kanten/ungültigen Flächen durch Dreiecke ersetzen (Randpunkte auf die
    Kantenkurven der Nachbarn gezogen) und alles vernähen."""
    from OCP.BRep import BRep_Builder
    from OCP.BRepAdaptor import BRepAdaptor_Curve
    from OCP.TopAbs import TopAbs_EDGE, TopAbs_SHELL
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopoDS import TopoDS_Shell
    from . import occ_build as ob
    from .occ_compat import to_shell
    faces = A["faces"]
    b = BRep_Builder()
    sh = TopoDS_Shell()
    b.MakeShell(sh)
    for _, f in faces:
        b.Add(sh, f)
    mp = _edge_face_map(sh)
    culprit = (set(int(i) for i in A["invalid_faces"]) | set(int(i) for i in A["face_fail"])
               | set(int(i) for i in extra))
    for i in range(1, mp.Extent() + 1):
        if mp.FindFromIndex(i).Size() >= 2:
            continue
        e = mp.FindKey(i)
        for ri, f in faces:
            ex = TopExp_Explorer(f, TopAbs_EDGE)
            hit = False
            while ex.More():
                if ex.Current().IsSame(e):
                    hit = True
                    break
                ex.Next()
            if hit:
                culprit.add(int(ri))
                break
    if not culprit:
        return None
    V = model.V.copy()
    moved = np.zeros(model.n_verts, bool)
    # Ecken
    for ch in chains:
        if ch["closed"]:
            continue
        for v in (int(ch["verts"][0]), int(ch["verts"][-1])):
            key = uf.find(v)
            if key in A["vpos"] and np.linalg.norm(A["vpos"][key] - model.V[v]) < max(10.0, 50 * tol):
                V[v] = A["vpos"][key]
                moved[v] = True
    for ch in chains:
        if ch.get("collapsed"):
            key = uf.find(int(ch["verts"][0]))
            if key in A["vpos"]:
                V[np.asarray(ch["verts"])] = A["vpos"][key]
                moved[np.asarray(ch["verts"])] = True
    # Kettenpunkte auf die Kantenkurven ziehen; zwischen je zwei Kettenpunkten weitere Kurvenpunkte
    # einfügen (grobe Dreiecke: die Sehne läge sonst bis mm-weit neben der gekrümmten Kante)
    ins = {}
    step = max(0.25, 2 * tol)
    snap_lim = max(10.0, 50 * tol)
    for ed in A["edges"]:
        ch = ed["ch"]
        if ch["r"] not in culprit and (ch["s"] < 0 or ch["s"] not in culprit):
            continue
        ad = BRepAdaptor_Curve(ed["edge"])
        u0, u1 = ad.FirstParameter(), ad.LastParameter()
        Q0 = np.array([[p.X(), p.Y(), p.Z()] for p in (ad.Value(u) for u in np.linspace(u0, u1, 50))])
        Lc = float(np.linalg.norm(np.diff(Q0, axis=0), axis=1).sum())
        n = int(np.clip(Lc / (0.25 * step), 50, 4000))
        Q = np.array([[p.X(), p.Y(), p.Z()] for p in (ad.Value(u) for u in np.linspace(u0, u1, n))])
        vs = np.asarray(ch["verts"])
        if not ch["closed"]:
            if np.linalg.norm(Q[0] - model.V[vs[0]]) > np.linalg.norm(Q[-1] - model.V[vs[0]]):
                Q = Q[::-1]
        elif len(vs) >= 4:
            # geschlossene Kette: Umlaufsinn der Kurve an der Kette ausrichten, Start am ersten Punkt
            k0_ = int(np.argmin(np.linalg.norm(Q - model.V[vs[0]], axis=1)))
            Q = np.roll(Q[:-1], -k0_, axis=0)
            Q = np.vstack([Q, Q[:1]])
            k1_ = int(np.argmin(np.linalg.norm(Q - model.V[vs[len(vs) // 4]], axis=1)))
            k3_ = int(np.argmin(np.linalg.norm(Q - model.V[vs[(3 * len(vs)) // 4]], axis=1)))
            if k1_ > k3_:
                Q = Q[::-1]
        idx = []
        for j, v in enumerate(vs):
            if not ch["closed"] and j in (0, len(vs) - 1):
                idx.append(0 if j == 0 else len(Q) - 1)
                if not moved[v]:
                    V[v] = Q[idx[-1]]
                    moved[v] = True
                continue
            x = model.V[v]
            lo = idx[-1] if idx else 0
            k = lo + int(np.argmin(np.linalg.norm(Q[lo:] - x, axis=1)))
            idx.append(k)
            if np.linalg.norm(Q[k] - x) < snap_lim:
                V[v] = Q[k]
                moved[v] = True
        for j in range(len(vs) - 1):
            a, b = int(vs[j]), int(vs[j + 1])
            k0, k1 = idx[j], idx[j + 1]
            if k1 - k0 < 2:
                continue
            seg = Q[k0 + 1:k1]
            arc = np.r_[0, np.cumsum(np.linalg.norm(np.diff(np.vstack([Q[k0], seg]), axis=0), axis=1))]
            keep = []
            last = 0.0
            for t_, p_ in zip(arc[1:], seg):
                if t_ - last >= step:
                    keep.append(p_)
                    last = t_
            if keep:
                ins[(a, b)] = np.asarray(keep)
                ins[(b, a)] = np.asarray(keep)[::-1]
    keep = [(ri, f) for ri, f in faces if ri not in culprit]
    parts = [f for _, f in keep]
    from .mesh_ops import triangulate_polygon
    for ri in sorted(culprit):
        fids = np.nonzero(reg == ri)[0]
        if len(fids) == 0:
            continue
        Vl = [p for p in V]           # Indizes wie im Netz, neue Punkte hinten an
        Fl = []
        for t in model.F[fids]:
            ring = []
            extra = False
            for a, b in ((t[0], t[1]), (t[1], t[2]), (t[2], t[0])):
                ring.append(int(a))
                pts = ins.get((int(a), int(b)))
                if pts is not None:
                    extra = True
                    for p_ in pts:
                        ring.append(len(Vl))
                        Vl.append(p_)
            if not extra:
                Fl.append(tuple(int(x) for x in t))
                continue
            R = np.asarray([Vl[i] for i in ring])
            nrm = np.cross(R[1] - R[0], R[-1] - R[0])
            if np.linalg.norm(nrm) < 1e-12:
                nrm = np.cross(V[t[1]] - V[t[0]], V[t[2]] - V[t[0]])
            x_, y_, _ = orthonormal_frame(unit(nrm))
            Q2 = np.c_[(R - R[0]) @ x_, (R - R[0]) @ y_]
            try:
                tri = triangulate_polygon(Q2)
                Fl += [(ring[i], ring[j], ring[k]) for i, j, k in tri]
            except Exception:
                Fl += [(ring[0], ring[i], ring[i + 1]) for i in range(1, len(ring) - 1)]
        Vl = np.asarray(Vl)
        Fr = np.asarray(Fl, np.int64)
        # zusammenfallende Punkte verschweißen (gezogene Kettenpunkte können aufeinander liegen)
        key = np.round(Vl / 1e-6).astype(np.int64)
        _, first, wmap = np.unique(key, axis=0, return_index=True, return_inverse=True)
        Fr = wmap.reshape(-1)[Fr]
        Fr = Fr[(Fr[:, 0] != Fr[:, 1]) & (Fr[:, 1] != Fr[:, 2]) & (Fr[:, 2] != Fr[:, 0])]
        Vw = Vl[first]
        T = Vw[Fr]
        ar = 0.5 * np.linalg.norm(np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0]), axis=1)
        Fr = Fr[ar > 1e-12]
        if len(Fr) == 0:
            continue
        vid, inv = np.unique(Fr, return_inverse=True)
        parts.append(ob.mesh_to_faceted(Vw[vid], inv.reshape(-1, 3), make_solid=False))
    best = None
    for st in (max(0.05, 3 * tol), max(0.3, 6 * tol), max(0.8, 10 * tol)):
        sewn = ob.sew(parts, st)
        ex = TopExp_Explorer(sewn, TopAbs_SHELL)
        shells = []
        while ex.More():
            shells.append(to_shell(ex.Current()))
            ex.Next()
        free_ = sum(_free_edges(x) for x in shells) if shells else 10 ** 9
        if best is None or (len(shells) == 1 and (best[1] != 1 or free_ < best[2])):
            best = (sewn, len(shells), free_, shells)
        if len(shells) == 1 and free_ == 0:
            break
    sewn, nsh, free_, shells = best
    shell = shells[0] if nsh == 1 else _one_shell(sewn)
    shell = _fix_shell(shell)
    free = _free_edges(shell)
    if 0 < free <= 60:
        # kleine Restlöcher (Splitter, T-Stöße) mit Füllflächen schließen
        try:
            shell2 = _fill_free_bounds(shell, max(0.3, 6 * tol))
            if shell2 is not None and _free_edges(shell2) < free:
                shell = shell2
                free = _free_edges(shell)
        except Exception:  # noqa: BLE001
            pass
    if free == 0:
        try:
            sol, vol = _make_solid(shell)
            # Formprüfung großer facettierter Körper kostet viel – der Export prüft am Ende ohnehin
            valid = ob.is_valid(sol) if _n_faces(sol) <= 3000 else True
            if vol > 0:
                return {"shape": sol, "solid": True, "free": 0, "faceted": sorted(culprit), "valid": valid, "vol": vol}
        except Exception:  # noqa: BLE001
            pass
    return {"shape": shell, "solid": False, "free": free, "faceted": sorted(culprit)}


def _fill_free_bounds(shell, sew_tol):
    """Geschlossene Ketten freier Kanten mit Flächen füllen (eben, sonst Füllfläche) und vernähen."""
    from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeFace
    from OCP.BRepOffsetAPI import BRepOffsetAPI_MakeFilling
    from OCP.GeomAbs import GeomAbs_C0
    from OCP.ShapeAnalysis import ShapeAnalysis_FreeBounds
    from OCP.TopAbs import TopAbs_EDGE, TopAbs_SHELL, TopAbs_WIRE
    from OCP.TopExp import TopExp_Explorer
    from . import occ_build as ob
    from .occ_compat import to_edge, to_shell, to_wire
    wires = None
    for at in (1e-3, 0.05, 0.3):
        fb = ShapeAnalysis_FreeBounds(shell, at, False, False)
        w_ = fb.GetClosedWires()
        ex = TopExp_Explorer(w_, TopAbs_WIRE)
        if ex.More():
            wires = w_
            break
    if wires is None:
        return None
    new = []
    ex = TopExp_Explorer(wires, TopAbs_WIRE)
    while ex.More():
        w = to_wire(ex.Current())
        face = None
        try:
            mf = BRepBuilderAPI_MakeFace(w, True)
            if mf.IsDone():
                face = mf.Face()
        except Exception:
            face = None
        if face is None:
            try:
                fl = BRepOffsetAPI_MakeFilling()
                e2 = TopExp_Explorer(w, TopAbs_EDGE)
                while e2.More():
                    fl.Add(to_edge(e2.Current()), GeomAbs_C0)
                    e2.Next()
                fl.Build()
                if fl.IsDone():
                    face = fl.Shape()
            except Exception:
                face = None
        if face is not None:
            new.append(face)
        ex.Next()
    if not new:
        return None
    sewn = ob.sew([shell] + new, sew_tol)
    return _one_shell(sewn)


def _n_faces(shape):
    from OCP.TopAbs import TopAbs_FACE
    from OCP.TopExp import TopExp_Explorer
    ex = TopExp_Explorer(shape, TopAbs_FACE)
    n = 0
    while ex.More():
        n += 1
        ex.Next()
    return n


def _orient_shell(shell):
    """Flächen einer Schale konsistent ausrichten (jede innere Kante einmal vorwärts, einmal
    rückwärts benutzt) – Breitensuche über die Kanten, linear in der Flächenzahl. Ersatz für
    ShapeFix_Shell bei großen (facettierten) Schalen, das dort quadratisch wird. Rückgabe
    (Schale, Anzahl umgedrehter Flächen) oder (None, 0), wenn die Schale nicht orientierbar ist."""
    from collections import deque
    from OCP.BRep import BRep_Builder
    from OCP.TopAbs import TopAbs_EDGE, TopAbs_FACE, TopAbs_FORWARD, TopAbs_REVERSED
    from OCP.TopExp import TopExp, TopExp_Explorer
    from OCP.TopTools import TopTools_IndexedMapOfShape
    from OCP.TopoDS import TopoDS_Shell
    emap = TopTools_IndexedMapOfShape()
    TopExp.MapShapes_s(shell, TopAbs_EDGE, emap)
    faces = []
    uses = {}
    ex = TopExp_Explorer(shell, TopAbs_FACE)
    while ex.More():
        f = ex.Current()
        i = len(faces)
        faces.append(f)
        seen = {}
        ee = TopExp_Explorer(f, TopAbs_EDGE)
        while ee.More():
            e = ee.Current()
            o = e.Orientation()
            if o in (TopAbs_FORWARD, TopAbs_REVERSED):
                k = emap.FindIndex(e)
                seen.setdefault(k, []).append(1 if o == TopAbs_FORWARD else -1)
            ee.Next()
        for k, os_ in seen.items():
            if len(os_) >= 2 and sum(os_) == 0:
                continue                      # Naht innerhalb der Fläche
            uses.setdefault(k, []).append((i, os_[0]))
        ex.Next()
    adj = [[] for _ in faces]
    for k, us in uses.items():
        if len(us) != 2:
            continue
        (i, oi), (j, oj) = us
        if i == j:
            continue
        need = 1 if oi == oj else 0          # gleiche Richtung -> eine von beiden umdrehen
        adj[i].append((j, need))
        adj[j].append((i, need))
    flip = [-1] * len(faces)
    for s0 in range(len(faces)):
        if flip[s0] >= 0:
            continue
        flip[s0] = 0
        dq = deque([s0])
        while dq:
            i = dq.popleft()
            for j, need in adj[i]:
                want = flip[i] ^ need
                if flip[j] < 0:
                    flip[j] = want
                    dq.append(j)
                elif flip[j] != want:
                    return None, 0           # Möbius-artig / inkonsistent
    nflip = sum(flip)
    if nflip == 0:
        return shell, 0
    b = BRep_Builder()
    sh = TopoDS_Shell()
    b.MakeShell(sh)
    for f, fl in zip(faces, flip):
        b.Add(sh, f.Reversed() if fl else f)
    return sh, nflip


def _fix_shell(shell, big=4000):
    """Ausrichtung reparieren: kleine Schalen mit ShapeFix_Shell, große mit _orient_shell."""
    from OCP.ShapeFix import ShapeFix_Shell
    if _n_faces(shell) > big:
        try:
            sh, _ = _orient_shell(shell)
            if sh is not None:
                return sh
        except Exception:  # noqa: BLE001
            pass
        return shell
    try:
        fs = ShapeFix_Shell(shell)
        fs.Perform()
        return fs.Shell()
    except Exception:  # noqa: BLE001
        return shell


def _make_solid(shell, big=4000):
    """Volumenkörper aus geschlossener Schale, außen orientiert. Große Schalen ohne ShapeFix
    (quadratisch) – Orientierung über BRepLib und das Vorzeichen des Volumens."""
    from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeSolid
    from OCP.BRepLib import BRepLib
    from OCP.ShapeFix import ShapeFix_Solid
    from OCP.TopAbs import TopAbs_SOLID
    from . import occ_build as ob
    from .occ_compat import to_solid
    sol = BRepBuilderAPI_MakeSolid(shell).Solid()
    if _n_faces(shell) <= big:
        fsol = ShapeFix_Solid(sol)
        fsol.Perform()
        sol = ob.fix_shape(fsol.Solid())
    try:
        if sol.ShapeType() == TopAbs_SOLID:
            sol = to_solid(sol)
            BRepLib.OrientClosedSolid_s(sol)
    except Exception:  # noqa: BLE001
        pass
    vol = _volume(sol)
    if vol < 0:
        sol = sol.Reversed()
        vol = -vol
    return sol, vol


def _one_shell(shape):
    """Alle Flächen einer (genähten) Form in eine Schale; None, wenn sie nicht zusammenhängen."""
    from OCP.BRep import BRep_Builder
    from OCP.ShapeFix import ShapeFix_Shell
    from OCP.TopAbs import TopAbs_FACE, TopAbs_SHELL
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopoDS import TopoDS_Shell
    from .occ_compat import to_shell
    ex = TopExp_Explorer(shape, TopAbs_SHELL)
    shells = []
    while ex.More():
        shells.append(to_shell(ex.Current()))
        ex.Next()
    if len(shells) == 1:
        return shells[0]
    b = BRep_Builder()
    sh = TopoDS_Shell()
    b.MakeShell(sh)
    ex = TopExp_Explorer(shape, TopAbs_FACE)
    while ex.More():
        b.Add(sh, ex.Current())
        ex.Next()
    if _n_faces(sh) > 4000:
        try:
            out, _ = _orient_shell(sh)
            return out if out is not None else sh
        except Exception:  # noqa: BLE001
            return sh
    try:
        fs = ShapeFix_Shell(sh)
        fs.Perform()
        out = fs.Shell()
        if fs.NbShells() == 1:
            return out
    except Exception:
        pass
    return sh


def _transition_feature(model, ids):
    """Platzhalter-Träger (Ausgleichsebene) für einen Übergangsstreifen – die Fläche selbst wird als
    Freiform durch die Randkanten gebaut."""
    from .features import Feature
    P = model.centroids[ids]
    o = P.mean(0)
    _, _, vt = np.linalg.svd(P - o, full_matrices=False)
    n = vt[2]
    nm = (model.normals[ids] * model.areas[ids, None]).sum(0)
    if float(n @ nm) < 0:
        n = -n
    f = Feature("plane", {"origin": o, "normal": n, "transition": True}, "surface", ids, auto=True,
                name=tr("Übergang", "Transition"))
    f.name = f"{f.name} {f.id}"
    return f


def _transition_strips(model, reg, regions, carriers, A, vregs_of, tol, edge_len, min_faces,
                       w_max=6.0, k_res=3.0):
    """Dreiecke entlang problematischer Kanten zwischen zwei analytischen Flächen, die keiner der
    beiden Träger erklärt (Abstand > k_res·tol): die Verrundung bzw. der Übergang, den der Scan
    zeigt. Je Kante ein Streifen. Rückgabe: Liste von Index-Arrays."""
    from scipy.spatial import cKDTree
    analytic = ("plane", "cylinder", "cone", "sphere", "torus")
    T_e = max(0.3, 10 * tol)
    bad_ch = {}
    for ed in A["edges"]:
        ch = ed["ch"]
        r, s_ = ch["r"], ch["s"]
        if s_ < 0 or ed["tol"] <= T_e:
            continue
        bad_ch[id(ch)] = ch
    # Kanten an Ecken ohne gemeinsamen Punkt
    badkeys = {key for key, x0, e in A["bad_vkeys"] if e > max(0.3, 10 * tol)}
    if badkeys:
        for ed in A["edges"]:
            ch = ed["ch"]
            if ch["s"] < 0 or ch["closed"]:
                continue
            if ed.get("k0") in badkeys or ed.get("k1") in badkeys:
                bad_ch[id(ch)] = ch
    chains = [ch for ch in bad_ch.values()
              if all(carriers[regions[x]["carrier"]].kind in analytic
                     and not getattr(carriers[regions[x]["carrier"]], "transition", False)
                     for x in (ch["r"], ch["s"]))]
    if not chains:
        return []
    # Restabstand jedes Dreiecks zu seinem eigenen Träger
    C = model.centroids
    res = np.zeros(model.n_faces)
    rr = set()
    for ch in chains:
        rr.update((ch["r"], ch["s"]))
    for ri in rr:
        fids = np.nonzero(reg == ri)[0]
        res[fids] = np.abs(carriers[regions[ri]["carrier"]].distance(C[fids]))
    # Kettenpunkte mit Kettennummer
    pts, own = [], []
    step = max(0.5 * edge_len, 0.1)
    for k, ch in enumerate(chains):
        P = model.V[ch["verts"]]
        L = float(np.linalg.norm(np.diff(P, axis=0), axis=1).sum())
        n = int(np.clip(L / step, 2, 4000))
        Q = _resample(P, n, ch["closed"])
        pts.append(Q)
        own += [k] * len(Q)
    pts = np.vstack(pts)
    own = np.asarray(own)
    tree = cKDTree(pts)
    cand_f = np.nonzero(np.isin(reg, list(rr)))[0]
    d, j = tree.query(C[cand_f], distance_upper_bound=w_max)
    ok = np.isfinite(d)
    cand_f, j = cand_f[ok], j[ok]
    chain_of = own[j]
    # nur Dreiecke der beiden Regionen der jeweiligen Kette, die keiner der Träger erklärt
    rs = np.array([(chains[k]["r"], chains[k]["s"]) for k in range(len(chains))])
    mine = (reg[cand_f] == rs[chain_of, 0]) | (reg[cand_f] == rs[chain_of, 1])
    poor = res[cand_f] > k_res * tol
    sel = mine & poor
    cand_f, chain_of = cand_f[sel], chain_of[sel]
    out = []
    used = np.zeros(model.n_faces, bool)
    for k, ch in enumerate(chains):
        ids = cand_f[chain_of == k]
        ids = ids[~used[ids]]
        if len(ids) < 3:
            continue
        # zusammenhängend mit der Kette (Dreiecke an den Kettenpunkten)
        vset = np.zeros(model.n_verts, bool)
        vset[np.asarray(ch["verts"])] = True
        seed = ids[vset[model.F[ids]].any(1)]
        if len(seed) == 0:
            continue
        comps = model.components_of(ids)
        keep = [c for c in comps if np.isin(c, seed).any()]
        if not keep:
            continue
        ids = np.concatenate(keep)
        # einen Ring dazu (innerhalb der beiden Regionen), glättet den Rand
        mm = np.zeros(model.n_faces, bool)
        mm[ids] = True
        g = model.grow(mm, 1) & np.isin(reg, (ch["r"], ch["s"])) & ~used
        ids = np.nonzero(g)[0]
        if len(ids) < max(8, min_faces // 3):
            continue
        # die Feature-Fläche darf nicht im Streifen aufgehen
        if any(np.isin(np.nonzero(reg == x)[0], ids).mean() > 0.6 for x in (ch["r"], ch["s"])):
            continue
        used[ids] = True
        out.append(ids)
    return out

def _blame(model, regions, carriers, A, vregs_of, tol):
    """Regionen, deren Träger die Topologie nicht trägt."""
    bad = set(int(i) for i in A["face_fail"]) | set(int(i) for i in A["invalid_faces"])

    def dist(ri, P):
        car = carriers[regions[ri]["carrier"]]
        return float(np.median(np.abs(_Fast(car).distance(np.atleast_2d(P)))))

    T_e = max(0.3, 10 * tol)
    for ed in A["edges"]:
        if ed["tol"] <= T_e:
            continue
        ch = ed["ch"]
        r, s_ = ch["r"], ch["s"]
        if s_ < 0:
            bad.add(r)
            continue
        P = model.V[ch["verts"]]
        bad.add(r if dist(r, P) >= dist(s_, P) else s_)
    for key, x0, e in A["bad_vkeys"]:
        if e <= max(0.3, 10 * tol):
            continue
        rr = list(vregs_of.get(key, ()))
        if rr:
            bad.add(max(rr, key=lambda ri: dist(ri, x0)))
    # periodische (geteilte) Bereiche lassen sich nicht eben parametrisieren
    return {b for b in bad if 0 <= b < len(regions) and regions[b].get("split") is None}


def _smooth_polyline(S, closed, iters=4, lam=0.5):
    X = np.asarray(S, float).copy()
    if len(X) < 4:
        return X
    for _ in range(iters):
        if closed:
            Y = 0.5 * (np.roll(X, 1, 0) + np.roll(X, -1, 0))
            X = X + lam * (Y - X)
        else:
            Y = X.copy()
            Y[1:-1] = 0.5 * (X[:-2] + X[2:])
            X = X + lam * (Y - X)
    return X


def _carrier_uv(car, P, margin=0.04):
    """(u, v) aus der Parametrisierung des Trägers (Zylinder/Kegel/Torus/Kugel), um die Mitte der
    Punkte abgewickelt und auf [0, 1] normiert. None, wenn nicht möglich."""
    k = car.kind
    if k not in ("cylinder", "cone", "torus", "sphere"):
        return None
    if k == "sphere":
        c = np.asarray(car.p["center"], float)
        d = unit(P - c)
        m = unit(d.mean(0))
        z, _, _ = orthonormal_frame(m)
        y = np.cross(z, m)
        U = np.arctan2(d @ y, d @ m)
        V = np.arcsin(np.clip(d @ z, -1, 1))
    else:
        a = car.axis
        x, y, _ = orthonormal_frame(a)
        q = P - car.base
        ph = np.arctan2(q @ y, q @ x)
        ph0 = _circ_mean(ph)
        U = (ph - ph0 + math.pi) % (2 * math.pi) - math.pi
        h = q @ a
        if k == "torus":
            rho = np.linalg.norm(q - np.outer(h, a), axis=1)
            th = np.arctan2(h, rho - float(car.p["R"]))
            th0 = _circ_mean(th)
            V = (th - th0 + math.pi) % (2 * math.pi) - math.pi
        else:
            V = h
    if np.ptp(U) > 1.8 * math.pi:
        return None                                # fast ganz herum: nicht eindeutig
    X = np.c_[U, V]
    mn, mx = X.min(0), X.max(0)
    span = np.maximum(mx - mn, 1e-9)
    mn, mx = mn - margin * span, mx + margin * span
    return (X - mn) / (mx - mn)


def _constrained_bspline(model, fids, edges, tol, edge_len, car=None):
    """B-Spline-Fläche durch die Netzregion, die ihre Randkanten (fast) exakt enthält.
    Ebene Parametrisierung (Höhenfeld), sonst die des Trägers (abgewickelt); None, wenn beides
    nicht geht."""
    from OCP.BRepAdaptor import BRepAdaptor_Curve
    from . import occ_build as ob
    from .freeform import fit_bspline_surface, planar_params
    fids = np.asarray(fids, np.int64)
    F = model.F[fids]
    vid = np.unique(F)
    # Stützpunkte: Schwerpunkte + Ecken (grobe, vereinfachte Netze haben sonst zu wenige)
    Pin = np.vstack([model.centroids[fids], model.V[vid]])
    a3 = model.areas[fids] / 3.0
    va = np.bincount(F.ravel(), np.repeat(a3, 3), minlength=model.n_verts)[vid]
    win = np.r_[model.areas[fids], va]
    Nin = model.normals[fids]
    Pb = []
    for e in edges:
        ad = BRepAdaptor_Curve(e)
        u0, u1 = ad.FirstParameter(), ad.LastParameter()
        p0, p1 = ad.Value(u0), ad.Value(u1)
        Le = math.dist((p0.X(), p0.Y(), p0.Z()), (p1.X(), p1.Y(), p1.Z()))
        n = int(np.clip(Le / max(0.5 * edge_len, 0.2), 8, 60))
        for u in np.linspace(u0, u1, n):
            p = ad.Value(u)
            Pb.append((p.X(), p.Y(), p.Z()))
    Pb = np.asarray(Pb, float)
    P = np.vstack([Pin, Pb])
    uv, frame = planar_params(P, None, margin=0.04)
    c, ax_u, ax_v, ax_n, mn, mx = frame
    span = mx - mn
    # Höhenfeld? projizierte Dreiecke dürfen nicht umklappen
    T = model.V[F] - c
    t2 = np.stack([T @ ax_u, T @ ax_v], -1)
    sa = 0.5 * ((t2[:, 1, 0] - t2[:, 0, 0]) * (t2[:, 2, 1] - t2[:, 0, 1])
                - (t2[:, 2, 0] - t2[:, 0, 0]) * (t2[:, 1, 1] - t2[:, 0, 1]))
    tot = max(float(np.abs(sa).sum()), 1e-12)
    flip = min(float(np.abs(sa[sa < 0]).sum()), float(sa[sa > 0].sum())) / tot
    if flip > 0.005:
        uv = _carrier_uv(car, P) if car is not None else None
        if uv is None:
            return None
        ext = np.ptp(P, axis=0)
        span = np.array([float(np.linalg.norm(ext)), float(np.linalg.norm(ext))]) * 0.7
    e_reg = float(np.mean(np.linalg.norm(model.V[F[:, 1]] - model.V[F[:, 0]], axis=1)))
    spacing = max(0.8 * e_reg, float(span.max()) / 40.0, 0.25, 3 * tol)
    nu = int(np.clip(round(span[0] / spacing) + 3, 4, 45))
    nv = int(np.clip(round(span[1] / spacing) + 3, 4, 45))
    wb = np.full(len(Pb), 10.0 * float(win.mean()))
    w = np.r_[win, wb]
    C, tu, tv, uvf, res = fit_bspline_surface(P, uv, w, nu, nv, smooth=0.3, param_iters=2)
    return ob.bspline_surface({"poles": C, "knots_u": tu, "knots_v": tv, "degree": 3})


def _edge_face_map(shape):
    from OCP.TopAbs import TopAbs_EDGE, TopAbs_FACE
    from OCP.TopExp import TopExp
    from .occ_compat import ShapeListMap
    mp = ShapeListMap()
    TopExp.MapShapesAndAncestors_s(shape, TopAbs_EDGE, TopAbs_FACE, mp)
    return mp


def _free_edges(shape):
    from OCP.BRep import BRep_Tool
    from .occ_compat import to_edge
    mp = _edge_face_map(shape)
    n = 0
    for i in range(1, mp.Extent() + 1):
        e = to_edge(mp.FindKey(i))
        if BRep_Tool.Degenerated_s(e):
            continue
        if mp.FindFromIndex(i).Size() < 2:
            n += 1
    return n


def _free_edge_polylines(shape):
    from . import occ_build as ob
    from .occ_compat import to_edge
    mp = _edge_face_map(shape)
    out = []
    for i in range(1, mp.Extent() + 1):
        if mp.FindFromIndex(i).Size() < 2:
            out += ob.edge_polylines(to_edge(mp.FindKey(i)), 0.05)
    return out


def _volume(shape):
    from OCP.BRepGProp import BRepGProp
    from OCP.GProp import GProp_GProps
    g = GProp_GProps()
    try:
        BRepGProp.VolumeProperties_s(shape, g)
        return float(g.Mass())
    except Exception:
        return 0.0
