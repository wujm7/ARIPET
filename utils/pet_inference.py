"""Complete-volume inference with NASC-only scale restoration."""
import numpy as np
import torch
from utils.pet_scaling import normalize_pair, restore_suv, NORMALIZATION


@torch.no_grad()
def reconstruct_volume(network, nasc, device=None):
    nasc = np.asarray(nasc, dtype=np.float32)
    if nasc.ndim != 3 or not np.isfinite(nasc).all():
        raise ValueError("Expected a finite H x W x S SUV volume")
    if device is None:
        device = next(network.parameters()).device
    output = np.empty_like(nasc)
    scales = np.empty(nasc.shape[2], dtype=np.float32)
    was_training = network.training
    network.eval()
    try:
        for z in range(nasc.shape[2]):
            low, _, scale = normalize_pair(nasc[:, :, z])
            result = network(torch.from_numpy(low[None, None]).to(device))
            prediction = result[0] if isinstance(result, tuple) else result
            prediction = prediction[0, 0].float().cpu().numpy()
            output[:, :, z] = restore_suv(prediction, scale, nasc.shape[:2])
            scales[z] = scale
    finally:
        network.train(was_training)
    if not np.isfinite(output).all():
        raise FloatingPointError("Non-finite reconstructed SUV")
    return output, scales


def load_ema_checkpoint(path, device="cpu"):
    from models.select_network import define_G
    state = torch.load(path, map_location="cpu", weights_only=False)
    if (state.get("normalization") != NORMALIZATION or
            state.get("frequency_grid") != "unshifted_fftfreq_nyquist_v1"):
        raise ValueError("Checkpoint has no matching normalization/frequency-grid metadata")
    if not isinstance(state.get("params_ema"), dict):
        raise ValueError("Inference requires an explicitly saved params_ema state")
    config = dict(state["config"], is_train=False)
    network = define_G(config).to(device)
    network.load_state_dict(state["params_ema"], strict=True)
    network.eval()
    return network, state
