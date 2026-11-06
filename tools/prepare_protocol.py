"""Prepare source-only train/validation and unchanged target-test partitions."""
import argparse
import csv
import json
from pathlib import Path
import random
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.dataset_unipet import read_manifest, canonical_domain


def is_target(row, setting):
    if setting == "UIH":
        return row["scanner"] == "UIH"
    if setting == "MFBG":
        return row["radiotracer"] == "MFBG"
    if setting == "SinoUnion-FDG":
        return row["scanner"] == "SinoUnion" and row["radiotracer"] == "FDG"
    if setting == "full":
        return False
    raise ValueError(setting)


def partition(rows, setting, seed=42):
    rows = [dict(r) for r in rows]
    for row in rows:
        row["scanner"], row["radiotracer"] = canonical_domain(row["scanner"], row["radiotracer"])
        row["domain"] = row["scanner"] + "__" + row["radiotracer"]
    target_patients = {r["patient_id"] for r in rows if is_target(r, setting)}
    development = {r["patient_id"] for r in rows if r["split"] in {"train", "val", "development"}
                   and r["patient_id"] not in target_patients}
    test_patients = {r["patient_id"] for r in rows if r["split"] == "test"}
    if development & test_patients:
        raise ValueError("Patient overlap between development and test")
    rng = random.Random(seed)
    candidates = sorted(development)
    rng.shuffle(candidates)
    # Reserve at least one source-training patient for every domain. Selection
    # uses metadata only, never performance or outcome values.
    domain_patients = {}
    for row in rows:
        if row["patient_id"] in development:
            domain_patients.setdefault(row["domain"], set()).add(row["patient_id"])
    training = set()
    for domain in sorted(domain_patients, key=lambda d: (len(domain_patients[d]), d)):
        choices = domain_patients[domain]
        if not training & choices:
            training.add(next(pid for pid in candidates if pid in choices))
    count = 7 * len(development) // 8
    if len(training) > count or count == len(development):
        raise ValueError("Too few source patients for the 7:1 split and all training domains")
    training.update([pid for pid in candidates if pid not in training][:count - len(training)])
    for row in rows:
        pid = row["patient_id"]
        if pid in test_patients:
            row["split"] = "test" if setting == "full" or is_target(row, setting) else "excluded"
        elif pid in development:
            row["split"] = "train" if pid in training else "val"
        else:
            row["split"] = "excluded"
    # A patient with heterogeneous repeated exams is excluded as a whole if
    # only some of its test exams belong to the target, keeping one split/ID.
    for pid in test_patients:
        matching = [r for r in rows if r["patient_id"] == pid]
        if any(r["split"] == "test" for r in matching):
            for row in matching:
                if not is_target(row, setting) and setting != "full":
                    raise ValueError("A test patient spans target/source domains; define a patient-level target rule")
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--setting", choices=["full", "UIH", "MFBG", "SinoUnion-FDG"], required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    rows = partition(read_manifest(args.manifest), args.setting, args.seed)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    counts = {split: len({r["patient_id"] for r in rows if r["split"] == split})
              for split in ("train", "val", "test", "excluded")}
    counts["num_domains"] = len({r["domain"] for r in rows if r["split"] == "train"})
    print(json.dumps(counts, indent=2))


if __name__ == "__main__":
    main()
