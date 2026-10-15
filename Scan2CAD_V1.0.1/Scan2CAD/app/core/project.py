"""Projektdatei (.s2c = ZIP mit Netz, Features, Flächenzuordnung, Füllungen, Symmetrien,
Bedingungen)."""
from __future__ import annotations

import io
import json
import zipfile

import numpy as np

from .features import Feature, to_jsonable
from .mesh_model import MeshModel

VERSION = 3


def save_project(path, model, features, selection=None, meta=None, symmetries=(), constraints=()):
    from .symmetry import sym_to_json
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as z:
        buf = io.BytesIO()
        arrays = {"V": model.V, "F": model.F, "synthetic": np.nonzero(model.synthetic)[0]}
        for f in features:
            arrays[f"faces_{f.id}"] = np.asarray(f.face_ids, np.int64)
        fills = []
        for i, fl in enumerate(model.fills):
            arrays[f"fill_faces_{i}"] = np.asarray(fl["faces"], np.int64)
            rims = fl.get("rim")
            rims = [] if rims is None else [np.asarray(r, float) for r in rims]
            for j, rim in enumerate(rims):
                arrays[f"fill_rim_{i}_{j}"] = rim
            d = {k: v for k, v in fl.items() if k not in ("faces", "rim")}
            d["n_rims"] = len(rims)
            fills.append(d)
        if selection is not None:
            arrays["selection"] = np.nonzero(selection)[0]
        np.savez_compressed(buf, **arrays)
        z.writestr("mesh.npz", buf.getvalue())
        z.writestr("project.json", json.dumps(to_jsonable({
            "version": VERSION, "features": [f.to_dict() for f in features], "meta": meta or {},
            "fills": fills, "symmetries": [sym_to_json(s) for s in symmetries],
            "constraints": list(constraints)}), indent=1, ensure_ascii=False))


def load_project(path):
    """Rückgabe dict(model, features, selection, meta, symmetries, constraints)."""
    from .symmetry import sym_from_json
    with zipfile.ZipFile(path, "r") as z:
        data = json.loads(z.read("project.json").decode("utf-8"))
        arr = np.load(io.BytesIO(z.read("mesh.npz")))
        nF = len(arr["F"])
        syn = np.zeros(nF, bool)
        if "synthetic" in arr.files:
            syn[arr["synthetic"]] = True
        fills = []
        for i, d in enumerate(data.get("fills", [])):
            fl = dict(d)
            fl["faces"] = arr[f"fill_faces_{i}"] if f"fill_faces_{i}" in arr.files else np.zeros(0, np.int64)
            fl["rim"] = [arr[f"fill_rim_{i}_{j}"] for j in range(int(d.get("n_rims", 0)))
                         if f"fill_rim_{i}_{j}" in arr.files]
            if fl.get("plane") is not None:
                fl["plane"] = (np.asarray(fl["plane"][0], float), np.asarray(fl["plane"][1], float))
            fl.pop("n_rims", None)
            fills.append(fl)
        model = MeshModel(arr["V"], arr["F"], syn, fills)
        feats = []
        for d in data["features"]:
            key = f"faces_{d.get('id', 0)}"
            ids = arr[key] if key in arr.files else np.zeros(0, np.int64)
            feats.append(Feature.from_dict(d, ids))
        sel = np.zeros(model.n_faces, bool)
        if "selection" in arr.files:
            sel[arr["selection"]] = True
    return {"model": model, "features": feats, "selection": sel, "meta": data.get("meta", {}),
            "symmetries": [sym_from_json(s) for s in data.get("symmetries", [])],
            "constraints": data.get("constraints", [])}
