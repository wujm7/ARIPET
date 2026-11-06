"""The manuscript objective, shared by the public training interfaces."""
import torch
import torch.nn.functional as F


def compute_orth_loss(z_sh, z_sp):
    shared = F.normalize(z_sh.flatten(1), dim=1)
    specific = F.normalize(z_sp.flatten(1), dim=1)
    return ((shared * specific).sum(dim=1) ** 2).mean()


def charbonnier(a, b, eps=1e-9):
    return torch.sqrt((a - b).square() + eps).mean()


def fourier_channels(image):
    # The same two-channel real/imaginary transform for prediction and ASC.
    spectrum = torch.fft.fft2(torch.fft.fftshift(image, dim=(-2, -1)))
    return torch.cat((spectrum.real, spectrum.imag), dim=1)


def compute_total_loss(pred, target, tracer_logits, tracer_label, z_sh, z_sp,
                       lambda_cls=0.1, lambda_orth=0.01, alpha=15.0, eps=1e-9):
    image = charbonnier(pred, target, eps)
    frequency = charbonnier(fourier_channels(pred), fourier_channels(target), eps)
    reconstruction = alpha * image + frequency
    zero = pred.new_zeros(())
    if lambda_cls:
        if tracer_label is None or torch.any(tracer_label < 0):
            raise ValueError("Classification requires source-training domain labels")
        classification = F.cross_entropy(tracer_logits, tracer_label)
    else:
        classification = zero
    orthogonality = compute_orth_loss(z_sh, z_sp) if lambda_orth else zero
    total = reconstruction + lambda_cls * classification + lambda_orth * orthogonality
    return total, {"loss_rec": reconstruction, "loss_image": image,
                   "loss_frequency": frequency, "loss_cls": classification,
                   "loss_orth": orthogonality}
