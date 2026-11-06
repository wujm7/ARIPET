"""ARI-PET training wrapper. Compatibility class name: UniPETModel."""
from collections import OrderedDict
from copy import deepcopy
from pathlib import Path
import torch
from models.select_network import define_G
from models.pet_objective import compute_total_loss, compute_orth_loss
from utils.pet_scaling import NORMALIZATION


class UniPETModel:
    def __init__(self, opt):
        self.opt = opt
        self.opt_train = opt["train"]
        self.device = torch.device(opt.get("device", "cuda" if opt.get("gpu_ids") else "cpu"))
        self.netG = define_G(opt).to(self.device)
        if self.opt_train.get("freeze_patch_embedding", False):
            self.netG.patch_embed.requires_grad_(False)
        self.decay = float(self.opt_train.get("E_decay", 0.999))
        self.netE = deepcopy(self.netG).eval().requires_grad_(False) if self.decay > 0 else None
        self.domain_label = None
        self.H = None
        self.log_dict = OrderedDict()
        self.schedulers = []

    def init_train(self):
        self.load()
        self.netG.train()
        self.define_optimizer()
        self.define_scheduler()

    def load(self):
        paths = self.opt.get("path", {})
        if paths.get("pretrained_netG"):
            state = torch.load(paths["pretrained_netG"], map_location="cpu", weights_only=False)
            self.netG.load_state_dict(state.get("params", state), strict=True)
            if self.netE is not None:
                self.netE.load_state_dict(state.get("params_ema", self.netG.state_dict()), strict=True)
        if paths.get("pretrained_netE"):
            if self.netE is None:
                raise ValueError("EMA is disabled")
            state = torch.load(paths["pretrained_netE"], map_location="cpu", weights_only=False)
            self.netE.load_state_dict(state.get("params_ema", state), strict=True)

    def define_optimizer(self):
        if self.opt_train.get("G_optimizer_type", "adam").lower() != "adam":
            raise ValueError("The manuscript configuration uses Adam")
        self.G_optimizer = torch.optim.Adam(
            [p for p in self.netG.parameters() if p.requires_grad],
            lr=self.opt_train.get("G_optimizer_lr", 2e-4), betas=(0.9, 0.999),
            weight_decay=self.opt_train.get("G_optimizer_wd", 0.0))

    def define_scheduler(self):
        self.schedulers = [torch.optim.lr_scheduler.MultiStepLR(
            self.G_optimizer, self.opt_train.get("G_scheduler_milestones", [500000, 1000000]),
            gamma=self.opt_train.get("G_scheduler_gamma", 0.5))]

    def feed_data(self, data, need_H=True):
        self.L = data["L"].to(self.device)
        self.H = data["H"].to(self.device) if need_H else None
        self.domain_label = data.get("domain_label", data.get("tracer_label"))
        if self.domain_label is not None:
            self.domain_label = self.domain_label.to(self.device)
        self.scale = data.get("scale")

    def netG_forward(self):
        self.E, self.tracer_logits, self.z_sh, self.z_sp = self.netG(self.L)

    def total_loss(self):
        if self.H is None:
            raise ValueError("Training requires an ASC reference")
        total, terms = compute_total_loss(
            self.E, self.H, self.tracer_logits, self.domain_label, self.z_sh, self.z_sp,
            alpha=self.opt_train.get("alpha", 15.0),
            lambda_cls=self.opt_train.get("lambda_cls", 0.1),
            lambda_orth=self.opt_train.get("lambda_orth", 0.01),
            eps=self.opt_train.get("G_charbonnier_eps", 1e-9))
        self.log_dict = OrderedDict([("G_loss", total.detach().item())])
        self.log_dict.update(("G_" + k, v.detach().item()) for k, v in terms.items())
        return total

    def optimize_parameters(self, current_step):
        self.netG.train()
        self.G_optimizer.zero_grad(set_to_none=True)
        self.netG_forward()
        loss = self.total_loss()
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite training loss")
        loss.backward()
        clip = self.opt_train.get("G_optimizer_clipgrad")
        if clip:
            torch.nn.utils.clip_grad_norm_(self.netG.parameters(), clip)
        self.G_optimizer.step()
        self.update_E(self.decay)

    @torch.no_grad()
    def update_E(self, decay=0.999):
        if self.netE is None:
            return
        for ema, current in zip(self.netE.parameters(), self.netG.parameters()):
            ema.mul_(decay).add_(current, alpha=1 - decay)
        for ema, current in zip(self.netE.buffers(), self.netG.buffers()):
            ema.copy_(current)
        self.netE.eval()

    @torch.no_grad()
    def test(self):
        network = self.netE if self.netE is not None else self.netG
        prior_training = network.training
        network.eval()
        try:
            self.E, self.tracer_logits, self.z_sh, self.z_sp = network(self.L)
        finally:
            network.train(prior_training)

    def update_learning_rate(self, step=None):
        for scheduler in self.schedulers:
            scheduler.step()

    def current_learning_rate(self):
        return self.G_optimizer.param_groups[0]["lr"]

    def current_log(self):
        return self.log_dict

    def current_results(self, need_H=True, suv=False):
        values = {"L": self.L, "E": self.E}
        if need_H:
            values["H"] = self.H
        if suv:
            if self.scale is None:
                raise ValueError("NASC-derived scale is required for SUV restoration")
            scale = torch.as_tensor(self.scale, device=self.device).reshape(-1, 1, 1, 1)
            values = {key: value * scale for key, value in values.items()}
        return OrderedDict((key, value.detach().float().cpu()) for key, value in values.items())

    def current_visuals(self, need_H=True):
        return OrderedDict((k, v[0]) for k, v in self.current_results(need_H).items())

    def info_network(self):
        return str(self.netG)

    def info_params(self):
        return "Parameters: {:,}".format(sum(p.numel() for p in self.netG.parameters()))

    def save(self, iter_label, metadata=None):
        path = Path(self.opt["path"]["models"]) / (str(iter_label) + ".pth")
        path.parent.mkdir(parents=True, exist_ok=True)
        # Exclusive creation keeps existing training records immutable.
        state = {"format_version": 1, "normalization": NORMALIZATION,
                 "frequency_grid": "unshifted_fftfreq_nyquist_v1",
                 "params": self.netG.state_dict(),
                 "params_ema": self.netE.state_dict() if self.netE is not None else None,
                 "optimizer": self.G_optimizer.state_dict(),
                 "scheduler": self.schedulers[0].state_dict(),
                 "config": self.opt, "metadata": metadata or {}}
        with path.open("xb") as stream:
            torch.save(state, stream)
        return path
