"""Reconstruct complete SUV volumes from NASC alone using saved EMA weights."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import scipy.io as sio
from data.dataset_unipet import read_manifest, load_volume
from utils.pet_inference import load_ema_checkpoint, reconstruct_volume
from utils.pet_metrics import volume_metrics, patient_means


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--metrics", action="store_true", help="Optional ASC evaluation AFTER SUV reconstruction")
    args = parser.parse_args()
    network, state = load_ema_checkpoint(args.checkpoint, args.device)
    rows = [r for r in read_manifest(args.manifest) if r["split"] == "test"]
    if not rows:
        raise ValueError("No formal test rows")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    checkpoint_sha = hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest()
    scans = []
    for row in rows:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", row["study_id"]):
            raise ValueError("study_id must be a safe de-identified filename component")
        prediction, scales = reconstruct_volume(network, load_volume(row["nasc"]), args.device)
        path = output / (row["study_id"] + ".mat")
        with path.open("xb") as stream:
            sio.savemat(stream, {"output": prediction, "nasc_scales": scales,
                                "axis_order": "HWS", "units": "SUV"})
        # Read back persisted reconstruction before recording completion.
        restored = sio.loadmat(path)["output"]
        if restored.shape != prediction.shape or not (restored == prediction).all():
            raise IOError("Reconstruction readback failed")
        record = {"study_id": row["study_id"], "patient_id": row["patient_id"],
                  "output": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                  "nasc_sha256": hashlib.sha256(Path(row["nasc"]).read_bytes()).hexdigest(),
                  "checkpoint_sha256": checkpoint_sha, "weights": "params_ema",
                  "normalization": state["normalization"], "shape": list(prediction.shape),
                  "axis_order": "HWS", "units": "SUV"}
        with (output / "reconstructions.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record) + "\n")
        if args.metrics:
            if not row.get("asc"):
                raise ValueError("--metrics needs ASC references; reconstruction itself does not")
            scans.append({"patient_id": row["patient_id"], "study_id": row["study_id"],
                          **volume_metrics(restored, load_volume(row["asc"]))})
    if args.metrics:
        (output / "metrics.json").write_text(
            json.dumps({"scans": scans, "patients": patient_means(scans)}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
