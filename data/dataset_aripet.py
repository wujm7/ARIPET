"""Patient-partitioned, paired SUV volumes; axial network samples are 200x200."""
import csv
from functools import lru_cache
from pathlib import Path
import numpy as np
import scipy.io as sio
import torch
from torch.utils.data import Dataset
from utils.pet_scaling import normalize_pair


def canonical_domain(scanner, radiotracer):
    scanners = {"siemens": "Siemens", "sinounion": "SinoUnion", "uih": "UIH"}
    tracers = {"FDG", "DOTATATE", "MFBG", "FAPI", "CXCR4", "DOPA", "PSMA"}
    scanner = scanners.get(scanner.strip().casefold())
    tracer = radiotracer.strip().upper()
    if scanner is None or tracer not in tracers:
        raise ValueError("Use canonical scanner and radiotracer metadata; unknown domains are not inferred")
    return scanner, tracer


def read_manifest(path):
    path = Path(path).resolve()
    with path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError("Empty manifest")
    required = {"patient_id", "study_id", "scanner", "radiotracer", "split", "nasc"}
    if not required.issubset(rows[0]):
        raise ValueError("Manifest requires: " + ", ".join(sorted(required)))
    patient_splits, studies, input_paths = {}, set(), set()
    for row in rows:
        if any(not row[k].strip() for k in required):
            raise ValueError("Manifest contains an empty required field")
        for key in required:
            row[key] = row[key].strip()
        row["scanner"], row["radiotracer"] = canonical_domain(row["scanner"], row["radiotracer"])
        if row["split"] not in {"train", "val", "test", "development", "excluded"}:
            raise ValueError("Unknown split: " + row["split"])
        pid = row["patient_id"]
        if pid in patient_splits and patient_splits[pid] != row["split"]:
            raise ValueError("All examinations of a patient must remain in one split")
        patient_splits[pid] = row["split"]
        if row["study_id"] in studies:
            raise ValueError("study_id must be unique per volume")
        studies.add(row["study_id"])
        for key in ("nasc", "asc"):
            if row.get(key):
                file = Path(row[key]).expanduser()
                row[key] = str((path.parent / file).resolve() if not file.is_absolute() else file.resolve())
        if row["nasc"] in input_paths:
            raise ValueError("A NASC volume cannot be repeated under multiple rows")
        input_paths.add(row["nasc"])
        if row.get("asc") == row["nasc"]:
            raise ValueError("NASC input and ASC reference cannot be the same file")
        row["domain"] = row["scanner"] + "__" + row["radiotracer"]
    return rows


@lru_cache(maxsize=2)
def load_volume(path):
    value = sio.loadmat(path, variable_names=["output"]).get("output")
    if value is None or value.ndim != 3 or not np.isfinite(value).all():
        raise ValueError("MAT output must be a finite H x W x S SUV volume: " + str(path))
    return np.asarray(value, dtype=np.float32)


class DatasetARIPET(Dataset):
    def __init__(self, opt):
        self.opt = opt
        self.phase = opt.get("phase", "train")
        all_rows = read_manifest(opt["manifest"])
        training_domains = sorted({r["domain"] for r in all_rows if r["split"] == "train"})
        supplied = opt.get("domain_map")
        self.domain_map = dict(supplied) if supplied is not None else {
            name: i for i, name in enumerate(training_domains)}
        if sorted(self.domain_map.values()) != list(range(len(self.domain_map))):
            raise ValueError("Training domain indices must be contiguous from zero")
        if self.phase in {"train", "val"} and not self.domain_map:
            raise ValueError("A source-training domain map is required")
        self.rows = [r for r in all_rows if r["split"] == self.phase]
        if not self.rows:
            raise ValueError("No rows for split " + self.phase)
        self.need_H = opt.get("need_H", True)
        self.target_size = (200, 200)
        self.num_domains = len(self.domain_map)
        # Compatibility names denote joint scanner--radiotracer classes.
        self.tracer2idx, self.num_tracers = self.domain_map, self.num_domains
        self.slice_info = []
        for index, row in enumerate(self.rows):
            if self.phase in {"train", "val"} and row["domain"] not in self.domain_map:
                raise ValueError("Development contains a domain absent from source training")
            low = load_volume(row["nasc"])
            if self.need_H:
                if not row.get("asc"):
                    raise ValueError("Paired training/validation requires an ASC path")
                high = load_volume(row["asc"])
                if low.shape != high.shape:
                    raise ValueError("NASC and ASC must be aligned before loading; shapes differ")
            for z in range(low.shape[2]):
                self.slice_info.append((index, z))

    def __len__(self):
        return len(self.slice_info)

    def __getitem__(self, index):
        row_index, z = self.slice_info[index]
        row = self.rows[row_index]
        low = load_volume(row["nasc"])[:, :, z]
        high = load_volume(row["asc"])[:, :, z] if self.need_H else None
        low, high, scale = normalize_pair(low, high, self.target_size)
        label = self.domain_map.get(row["domain"], -1)
        result = {"L": torch.from_numpy(low[None].copy()), "scale": scale,
                  "domain_label": torch.tensor(label), "tracer_label": torch.tensor(label),
                  "patient_id": row["patient_id"], "study_id": row["study_id"],
                  "slice_index": z, "L_path": row["nasc"]}
        if high is not None:
            result["H"] = torch.from_numpy(high[None].copy())
            result["H_path"] = row["asc"]
        return result
