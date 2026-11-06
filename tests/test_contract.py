import csv
from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import scipy.io as sio
import torch
from data.dataset_aripet import DatasetARIPET, read_manifest, load_volume
from models.pet_objective import compute_total_loss, fourier_channels, charbonnier
from models.restormer_tracer_split_bandprompt import PALHBlock, CoordConcat, Restormer
from models.model_aripet import ARIPETModel
from utils.pet_scaling import normalize_pair, restore_suv
from utils.pet_inference import reconstruct_volume, load_ema_checkpoint
from utils.pet_metrics import volume_metrics, patient_means
from tools.prepare_protocol import partition, is_target
from main_train import configure_variant

torch.set_num_threads(2)
ROOT = Path(__file__).resolve().parents[1]


def small_network(opt, num_tracers=None):
    return Restormer(dim=4, num_blocks=[1, 1, 1, 1], heads=[1, 1, 1, 1],
                     num_refinement_blocks=1, prompt_len=2,
                     num_tracers=num_tracers or opt["netG"]["num_domains"],
                     spatial_prompt=opt["netG"].get("spatial_prompt", True),
                     fourier_prompt=opt["netG"].get("fourier_prompt", True))


def options(directory):
    opt = json.loads((ROOT / "train_aripet_G1D30.json").read_text())
    opt.update(is_train=True, device="cpu")
    opt["netG"]["num_domains"] = 2
    opt["path"]["models"] = str(Path(directory) / "models")
    return opt


def make_manifest(directory):
    root = Path(directory)
    rng = np.random.default_rng(7)
    rows = []
    for i, (split, scanner, tracer) in enumerate([
            ("train", "Siemens", "FDG"), ("train", "UIH", "PSMA"),
            ("val", "UIH", "PSMA"), ("test", "SinoUnion", "MFBG")]):
        low = rng.uniform(1, 4, (9, 11, 3)).astype(np.float32)
        low[:, :, 0] = 0
        for key, value in (("nasc", low), ("asc", low * 2 + 0.1)):
            sio.savemat(root / (str(i) + "_" + key + ".mat"), {"output": value})
        rows.append(dict(patient_id="p" + str(i), study_id="s" + str(i), split=split,
                         scanner=scanner, radiotracer=tracer,
                         nasc=str(i) + "_nasc.mat", asc=str(i) + "_asc.mat"))
    path = root / "patients.csv"
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


class ScalingTests(unittest.TestCase):
    def test_shared_input_only_scaling_and_inverse(self):
        low = np.arange(1, 17, dtype=np.float32).reshape(4, 4)
        high = 5 * low + 7
        l, h, scale = normalize_pair(low, high, size=low.shape)
        _, h2, scale2 = normalize_pair(low, high * 3, size=low.shape)
        self.assertEqual(scale, scale2)
        self.assertGreater(h.max(), 1)
        np.testing.assert_allclose(restore_suv(h, scale), high)
        np.testing.assert_allclose(h2, h * 3, rtol=1e-6)
        np.testing.assert_allclose(restore_suv(l, scale), low)

    def test_zero_and_constant_slices_are_finite(self):
        for value in (0, 4):
            low, _, scale = normalize_pair(np.full((8, 8), value, np.float32))
            self.assertTrue(np.isfinite(low).all())
            self.assertGreater(scale, 0)
            np.testing.assert_allclose(restore_suv(low, scale), value, atol=1e-6)

    def test_complete_volume_without_reference(self):
        class Identity(torch.nn.Module):
            def forward(self, x):
                return x
        volume = np.ones((9, 11, 4), np.float32) * 3
        volume[:, :, 0] = 0
        out, scales = reconstruct_volume(Identity(), volume, "cpu")
        self.assertEqual(out.shape, volume.shape)
        self.assertEqual(len(scales), 4)
        np.testing.assert_allclose(out, volume, atol=2e-6)


class ObjectiveTests(unittest.TestCase):
    def test_full_objective_and_gradient_switches(self):
        for cls_weight, orth_weight in ((0.1, 0.01), (0, 0.01), (0, 0)):
            torch.manual_seed(2)
            prediction = torch.randn(2, 1, 4, 4, requires_grad=True)
            target = torch.randn_like(prediction)
            logits = torch.randn(2, 2, requires_grad=True)
            shared = torch.randn(2, 3, 2, 2, requires_grad=True)
            specific = torch.randn_like(shared, requires_grad=True)
            total, parts = compute_total_loss(prediction, target, logits, torch.tensor([0, 1]),
                                              shared, specific, cls_weight, orth_weight)
            expected = 15 * charbonnier(prediction, target) + charbonnier(
                fourier_channels(prediction), fourier_channels(target))
            expected = expected + cls_weight * parts["loss_cls"] + orth_weight * parts["loss_orth"]
            torch.testing.assert_close(total, expected)
            total.backward()
            self.assertGreater(prediction.grad.abs().sum().item(), 0)
            self.assertEqual(logits.grad is not None, bool(cls_weight))
            self.assertEqual(shared.grad is not None, bool(orth_weight))

    def test_variant_semantics(self):
        for name, expected in (("full", (0.1, 0.01)), ("noce", (0, 0.01)), ("no_sd", (0, 0))):
            opt = options(".")
            configure_variant(opt, name)
            self.assertEqual((opt["train"]["lambda_cls"], opt["train"]["lambda_orth"]), expected)


class FrequencyTests(unittest.TestCase):
    def test_dc_nyquist_symmetry_and_odd_width(self):
        block = PALHBlock(2)
        for height, width in ((16, 16), (25, 25), (24, 25), (25, 24)):
            low, high = block._build_band_masks(height, width, "cpu", torch.float64)
            self.assertEqual(low.shape, (1, 1, height, width // 2 + 1))
            self.assertEqual(low.dtype, torch.float64)
            self.assertEqual(low[0, 0, 0, 0].item(), 1)
            self.assertEqual(high[0, 0, height // 2, 0].item(), 1)
            self.assertEqual(high[0, 0, 0, -1].item(), 1)
            torch.testing.assert_close(low + high, torch.ones_like(low))
            for y in range(1, height):
                torch.testing.assert_close(low[0, 0, y], low[0, 0, -y])
            coords = CoordConcat()(torch.zeros(1, 2, height, width // 2 + 1), width)
            torch.testing.assert_close((coords[:, -1:] <= 0.5).double(), low)

    def test_known_fourier_modes(self):
        low, _ = PALHBlock(1)._build_band_masks(16, 16, "cpu", torch.float32)
        y, x = torch.meshgrid(torch.arange(16), torch.arange(16), indexing="ij")
        for ky, kx, is_low in ((0, 0, True), (1, 1, True), (7, 0, False), (0, 7, False)):
            mode = torch.cos(2 * torch.pi * (ky * y + kx * x) / 16)
            peak = torch.fft.rfft2(mode).abs().reshape(-1).argmax()
            self.assertEqual(bool(low.reshape(-1)[peak]), is_low)

    def test_finite_odd_shape_forward_backward(self):
        module = PALHBlock(2)
        x = torch.randn(2, 2, 25, 25, requires_grad=True)
        prompt = torch.randn_like(x, requires_grad=True)
        out = module(x, prompt)
        self.assertEqual(out.shape, x.shape)
        out.square().mean().backward()
        self.assertTrue(torch.isfinite(x.grad).all())
        self.assertTrue(torch.isfinite(prompt.grad).all())
        self.assertGreater(prompt.grad.abs().sum().item(), 0)


class DatasetTests(unittest.TestCase):
    def test_domain_spelling_cannot_bypass_holdout(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = make_manifest(tmp)
            path.write_text(path.read_text().replace("UIH,PSMA", " UIH , psma "))
            rows = read_manifest(path)
            self.assertEqual(rows[1]["domain"], "UIH__PSMA")
            self.assertTrue(is_target(rows[1], "UIH"))
            path.write_text(path.read_text().replace("MFBG", "18F-MFBG"))
            with self.assertRaisesRegex(ValueError, "canonical"):
                read_manifest(path)

    def test_source_label_map_pairing_and_all_slices(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = make_manifest(tmp)
            train = DatasetARIPET(dict(manifest=path, phase="train"))
            val = DatasetARIPET(dict(manifest=path, phase="val", domain_map=train.domain_map))
            test = DatasetARIPET(dict(manifest=path, phase="test", need_H=False, domain_map=train.domain_map))
            self.assertEqual(len(train), 6)
            self.assertEqual(val[0]["domain_label"].item(), train.domain_map["UIH__PSMA"])
            self.assertEqual(test[0]["domain_label"].item(), -1)
            self.assertNotIn("H", test[0])
            sample = train[1]
            restored_target = sample["H"].numpy()[0] * sample["scale"]
            from utils.pet_scaling import resize_slice
            row = train.rows[0]
            np.testing.assert_allclose(restored_target, resize_slice(load_volume(row["asc"])[:, :, 1]), rtol=1e-6)

    def test_patient_overlap_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = make_manifest(tmp)
            path.write_text(path.read_text().replace("p2,s2", "p0,s2"))
            with self.assertRaisesRegex(ValueError, "one split"):
                read_manifest(path)

    def test_target_reference_not_read_for_inference(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = make_manifest(tmp)
            path.write_text(path.read_text().replace("3_asc.mat", "unavailable_reference.mat"))
            test = DatasetARIPET(dict(manifest=path, phase="test", need_H=False))
            self.assertEqual(len(test), 3)

    def test_target_domain_excluded_from_all_development(self):
        for setting in ("UIH", "MFBG", "SinoUnion-FDG"):
            rows = []
            for domain in (("Siemens", "FDG"), ("UIH", "PSMA"), ("Siemens", "MFBG"), ("SinoUnion", "FDG")):
                for i in range(16):
                    identity = domain[0] + domain[1] + str(i)
                    rows.append(dict(patient_id=identity, scanner=domain[0], radiotracer=domain[1],
                                     domain="__".join(domain), split="test" if i >= 14 else "development"))
            result = partition(rows, setting)
            self.assertFalse(any(is_target(r, setting) and r["split"] in ("train", "val") for r in result))
            n = sum(r["split"] in ("train", "val") for r in result)
            self.assertEqual(sum(r["split"] == "train" for r in result), 7 * n // 8)
            self.assertTrue(all(is_target(r, setting) for r in result if r["split"] == "test"))


class ModelTests(unittest.TestCase):
    def test_ema_default_inference_and_no_reference(self):
        with tempfile.TemporaryDirectory() as tmp, patch("models.model_aripet.define_G", side_effect=small_network):
            model = ARIPETModel(options(tmp))
            model.netG.output.weight.data.zero_()
            model.netE.output.weight.data.fill_(0.01)
            model.feed_data({"L": torch.rand(1, 1, 32, 32)}, need_H=False)
            expected = model.netE(model.L)[0].detach()
            generator_before = model.netG.training
            model.test()
            torch.testing.assert_close(model.E, expected)
            self.assertEqual(model.netG.training, generator_before)

    def test_training_checkpoint_roundtrip_and_exclusive_save(self):
        with tempfile.TemporaryDirectory() as tmp, patch("models.model_aripet.define_G", side_effect=small_network), patch("models.select_network.define_G", side_effect=small_network):
            model = ARIPETModel(options(tmp))
            model.init_train()
            old_ema = deepcopy(model.netE.state_dict())
            model.feed_data({"L": torch.rand(2, 1, 32, 32), "H": torch.rand(2, 1, 32, 32),
                             "domain_label": torch.tensor([0, 1])})
            model.optimize_parameters(1)
            self.assertGreater(model.netG.tracer_classifier.weight.grad.abs().sum().item(), 0)
            for key, ema in model.netE.state_dict().items():
                torch.testing.assert_close(ema, old_ema[key] * 0.999 + model.netG.state_dict()[key] * 0.001)
            path = model.save(1)
            loaded, _ = load_ema_checkpoint(path)
            for key, value in loaded.state_dict().items():
                torch.testing.assert_close(value, model.netE.state_dict()[key])
            with self.assertRaises(FileExistsError):
                model.save(1)

    def test_patient_aggregation(self):
        rows = [{"patient_id": "a", **dict.fromkeys(("psnr", "ssim", "nmae_percent", "nmse"), x)} for x in (1, 3)]
        rows.append({"patient_id": "b", **dict.fromkeys(("psnr", "ssim", "nmae_percent", "nmse"), 5)})
        self.assertEqual([r["psnr"] for r in patient_means(rows)], [2, 5])


class EntrypointTests(unittest.TestCase):
    def test_zero_steps_never_falls_back_to_full_training(self):
        import main_train
        with patch.object(sys, "argv", ["main_train.py", "--opt", str(ROOT / "train_aripet_G1D30.json"),
                                        "--max-steps", "0"]):
            with self.assertRaisesRegex(ValueError, "positive"):
                main_train.main()

    def test_train_resume_ema_native_suv_inference(self):
        import main_train
        import main_test
        with tempfile.TemporaryDirectory() as tmp, patch("models.model_aripet.define_G", side_effect=small_network), patch("models.select_network.define_G", side_effect=small_network):
            root = Path(tmp)
            manifest = make_manifest(root)
            def train(folder, steps, resume=None):
                argv = ["main_train.py", "--opt", str(ROOT / "train_aripet_G1D30.json"),
                        "--manifest", str(manifest), "--output", str(root / folder),
                        "--device", "cpu", "--max-steps", str(steps)]
                if resume:
                    argv += ["--resume", str(resume)]
                with patch.object(sys, "argv", argv):
                    main_train.main()
            train("first", 1)
            checkpoint = root / "first/models/1.pth"
            train("resume", 2, checkpoint)
            train("whole", 2)
            resumed = torch.load(root / "resume/models/2.pth", weights_only=False)
            whole = torch.load(root / "whole/models/2.pth", weights_only=False)
            for key in whole["params"]:
                torch.testing.assert_close(resumed["params"][key], whole["params"][key], rtol=0, atol=0)
            self.assertEqual(resumed["metadata"]["step"], 2)
            with patch.object(sys, "argv", ["main_test.py", "--checkpoint", str(checkpoint),
                                           "--manifest", str(manifest), "--output", str(root / "pred"),
                                           "--device", "cpu", "--metrics"]):
                main_test.main()
            reconstructed = sio.loadmat(root / "pred/s3.mat")["output"]
            self.assertEqual(reconstructed.shape, (9, 11, 3))
            self.assertTrue(np.isfinite(reconstructed).all())
            records = (root / "pred/reconstructions.jsonl").read_text().splitlines()
            self.assertEqual(json.loads(records[0])["weights"], "params_ema")
            results = json.loads((root / "pred/metrics.json").read_text())
            self.assertEqual(len(results["patients"]), 1)


if __name__ == "__main__":
    unittest.main()
