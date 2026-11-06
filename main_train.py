"""Train ARI-PET on a patient manifest; select EMA checkpoints on source validation."""
import argparse
import hashlib
import json
import logging
from pathlib import Path
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from data.dataset_aripet import DatasetARIPET, load_volume
from models.model_aripet import ARIPETModel
from utils.pet_inference import reconstruct_volume
from utils.pet_metrics import volume_metrics, patient_means
from utils.pet_scaling import NORMALIZATION

VARIANTS = ("full", "noce", "no_sd", "no_spatial_prompt", "no_fourier_prompt")


def configure_variant(opt, variant):
    if variant not in VARIANTS:
        raise ValueError(variant)
    opt["variant"] = variant
    opt["train"]["lambda_cls"] = 0.0 if variant in ("noce", "no_sd") else 0.1
    opt["train"]["lambda_orth"] = 0.0 if variant == "no_sd" else 0.01
    opt["netG"]["spatial_prompt"] = variant != "no_spatial_prompt"
    opt["netG"]["fourier_prompt"] = variant != "no_fourier_prompt"


def validate(model, rows):
    network = model.netE
    if network is None:
        raise ValueError("The published configuration selects EMA weights")
    scans = []
    for row in rows:
        prediction, _ = reconstruct_volume(network, load_volume(row["nasc"]), model.device)
        scans.append({"patient_id": row["patient_id"], "study_id": row["study_id"],
                      **volume_metrics(prediction, load_volume(row["asc"]))})
    patients = patient_means(scans)
    return {key: float(np.mean([r[key] for r in patients]))
            for key in ("psnr", "ssim", "nmae_percent", "nmse")}


def _main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--opt", default="train_aripet_G1D30.json")
    parser.add_argument("--manifest", help="Patient-level CSV (overrides both train/val paths)")
    parser.add_argument("--output", help="New run directory; existing directories are never overwritten")
    parser.add_argument("--variant", choices=VARIANTS, default="full")
    parser.add_argument("--device", help="cpu or cuda:N")
    parser.add_argument("--resume", help="Resume a versioned checkpoint into a NEW output directory")
    parser.add_argument("--max-steps", type=int, help="Optional shorter run for execution checks")
    args = parser.parse_args()
    opt = json.loads(Path(args.opt).read_text(encoding="utf-8-sig"))
    maximum = args.max_steps if args.max_steps is not None else opt["train"]["total_iterations"]
    if maximum < 1:
        raise ValueError("The requested step count must be positive")
    opt["run_max_steps"] = maximum
    if set(opt["datasets"]) != {"train", "val"}:
        raise ValueError("Training accepts train/val only; the formal test set is not a development input")
    if opt.get("dist"):
        raise ValueError("This entry point uses one GPU. Launch separate runs for separate protocols.")
    opt["is_train"] = True
    opt["device"] = args.device or ("cuda:" + str(opt["gpu_ids"][0]) if opt.get("gpu_ids") else "cpu")
    use_cuda = torch.device(opt["device"]).type == "cuda"
    configure_variant(opt, args.variant)
    if opt["train"].get("G_lossfn_type") != "charbonnier" or opt["train"].get("G_lossfn_weight", 1) != 1:
        raise ValueError("Use the published Charbonnier objective without an extra overall multiplier")
    if opt["train"].get("E_decay") != 0.999:
        raise ValueError("The published EMA decay is 0.999")
    if args.manifest:
        for settings in opt["datasets"].values():
            settings["manifest"] = args.manifest
    manifests = {str(Path(d["manifest"]).resolve()) for d in opt["datasets"].values()}
    if len(manifests) != 1:
        raise ValueError("Use one audited manifest to enforce disjoint train/validation/test patients")
    manifest = next(iter(manifests))
    seed = int(opt.get("manual_seed", 42))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if use_cuda:
        torch.cuda.manual_seed_all(seed)
    train_settings = dict(opt["datasets"]["train"], phase="train")
    train_set = DatasetARIPET(train_settings)
    val_settings = dict(opt["datasets"]["val"], phase="val", domain_map=train_set.domain_map)
    val_set = DatasetARIPET(val_settings)
    opt["netG"]["num_domains"] = train_set.num_domains
    opt["domain_map"] = train_set.domain_map
    opt["manifest_sha256"] = hashlib.sha256(Path(manifest).read_bytes()).hexdigest()
    # Save hashes of executed source files without copying patient information into source control.
    root = Path(__file__).resolve().parent
    opt["code_sha256"] = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                           for directory in ("data", "models", "utils")
                           for p in sorted((root / directory).glob("*.py"))}
    opt["code_sha256"]["main_train.py"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    output = Path(args.output or (Path(opt["path"]["root"]) / opt["task"]))
    output.mkdir(parents=True, exist_ok=False)
    opt["path"]["models"] = str(output / "models")
    opt["path"]["pretrained_netG"] = None
    opt["path"]["pretrained_netE"] = None
    logger = logging.getLogger("aripet")
    for handler in list(logger.handlers):
        handler.close()
        logger.removeHandler(handler)
    logger.setLevel(logging.INFO)
    logger.addHandler(logging.StreamHandler())
    logger.addHandler(logging.FileHandler(output / "train.log", encoding="utf-8", mode="x"))
    model = ARIPETModel(opt)
    model.init_train()
    logger.info(model.info_params())
    step, epoch, batch_cursor, best = 0, 0, 0, -float("inf")
    if args.resume:
        state = torch.load(args.resume, map_location="cpu", weights_only=False)
        if state.get("normalization") != NORMALIZATION or state.get("frequency_grid") != "unshifted_fftfreq_nyquist_v1":
            raise ValueError("Resume requires matching normalization/frequency-grid metadata")
        prior = state["config"]
        for key in ("domain_map", "netG", "train", "manifest_sha256", "variant", "manual_seed"):
            if prior.get(key) != opt.get(key):
                raise ValueError("Resume mismatch: " + key)
        model.netG.load_state_dict(state["params"], strict=True)
        model.netE.load_state_dict(state["params_ema"], strict=True)
        model.G_optimizer.load_state_dict(state["optimizer"])
        model.schedulers[0].load_state_dict(state["scheduler"])
        meta = state["metadata"]
        step, epoch, batch_cursor, best = meta["step"], meta["epoch"], meta["batch_cursor"], meta["best_val_psnr"]
        random.setstate(meta["python_rng"])
        np.random.set_state(meta["numpy_rng"])
        torch.set_rng_state(meta["torch_rng"])
        if use_cuda and meta.get("cuda_rng"):
            torch.cuda.set_rng_state_all(meta["cuda_rng"])
        opt["resume_checkpoint"] = str(Path(args.resume).resolve())
    (output / "config.json").write_text(json.dumps(opt, indent=2), encoding="utf-8")
    if maximum < 1 or step >= maximum:
        raise ValueError("The final step must exceed the current checkpoint step")
    batch_size = int(train_settings["dataloader_batch_size"])
    if batch_size != 2:
        raise ValueError("The manuscript configuration uses batch size 2")
    if len(train_set) < batch_size:
        raise ValueError("Training dataset is smaller than a batch")
    while step < maximum:
        generator = torch.Generator().manual_seed(seed + epoch)
        indices = torch.randperm(len(train_set), generator=generator).tolist()
        batches = [indices[i:i + batch_size] for i in range(0, len(indices) - batch_size + 1, batch_size)]
        loader = DataLoader(train_set, batch_sampler=batches[batch_cursor:],
                            num_workers=int(train_settings.get("dataloader_num_workers", 0)),
                            generator=torch.Generator().manual_seed(seed + epoch))
        for batch in loader:
            model.feed_data(batch)
            model.optimize_parameters(step + 1)
            model.update_learning_rate()
            step += 1
            batch_cursor += 1
            if step % opt["train"]["checkpoint_print"] == 0:
                logger.info("step=%d lr=%.8g %s", step, model.current_learning_rate(), dict(model.current_log()))
            is_validation = step % opt["train"]["checkpoint_val"] == 0 or step == maximum
            improved = False
            if is_validation:
                metrics = validate(model, val_set.rows)
                improved = metrics["psnr"] > best
                best = max(best, metrics["psnr"])
                with (output / "validation.jsonl").open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps({"step": step, "source_val": metrics, "best": improved}) + "\n")
                logger.info("source validation, step=%d: %s", step, metrics)
            if improved or step % opt["train"]["checkpoint_save"] == 0 or step == maximum:
                meta = {"step": step, "epoch": epoch, "batch_cursor": batch_cursor,
                        "best_val_psnr": best, "python_rng": random.getstate(),
                        "numpy_rng": np.random.get_state(), "torch_rng": torch.get_rng_state(),
                        "cuda_rng": torch.cuda.get_rng_state_all() if use_cuda else None}
                saved = model.save(step, meta)
                with (output / "checkpoints.jsonl").open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps({"step": step, "checkpoint": str(saved),
                                             "selected_on_source_val": improved}) + "\n")
            if step == maximum:
                break
        epoch += 1
        batch_cursor = 0
    logger.info("Finished at step %d. All retained checkpoints are unchanged.", step)
    for handler in list(logger.handlers):
        handler.close()
        logger.removeHandler(handler)


def main():
    try:
        _main()
    finally:
        logger = logging.getLogger("aripet")
        for handler in list(logger.handlers):
            handler.close()
            logger.removeHandler(handler)


if __name__ == "__main__":
    main()
