# ARI-PET

Acquisition- and radiotracer-informed CT-free PET attenuation and scatter
correction. The input is SUV-valued NASC PET and the output is SUV-valued ASC PET.
The model processes 200 x 200 axial slices and assembles complete 3D volumes.

## Installation

Use Python 3.10+ and install a PyTorch build for your CPU/CUDA platform, then:

~~~bash
pip install -r requirements.txt
~~~

The active implementation requires PyTorch 2.1 or later. The manuscript
configuration uses a single NVIDIA GeForce RTX 3090 GPU and batch size 2.
The BasicSR and general-purpose utility modules retained in this repository
are not alternative ARI-PET training entry points.

## Data contract

Convert NASC and paired CT-based ASC images to **SUV** before using this loader.
MAT files contain an aligned, finite array named "output", with axis order
H x W x S. The loader does not infer SUV conversion from image intensity.
No image augmentation is applied. All slices, including constant slices, are
retained. Shape mismatches are rejected instead of silently trimming slices.

A private CSV manifest supplies:

~~~text
patient_id,study_id,scanner,radiotracer,split,nasc,asc
~~~

Use de-identified identifiers. "study_id" is unique per examination and
"patient_id" groups repeated examinations. Paths are absolute or relative to
the manifest. The split is "train", "val", or "test". Every examination from a
patient must use the same split. Inference only requires "nasc"; "asc" can be
empty unless metrics are requested.

Scanner values are Siemens, SinoUnion, or UIH. Radiotracer values are FDG,
DOTATATE, MFBG, FAPI, CXCR4, DOPA, or PSMA. These fields come from acquisition
metadata. They are not inferred from target-image intensities or from arbitrary
positions in a filename. To populate them from DICOM headers:

~~~bash
python tools/domain_metadata.py --input private_dicom_index.csv --output patients.csv
~~~

The input index additionally contains a "dicom_path" column pointing to a NASC
DICOM header. Unrecognized or missing metadata are rejected for explicit review.
Do not commit clinical images, private manifests, or patient identifiers.

## Training and validation

The ARI-PET training configuration is "train_aripet_G1D30.json":

~~~bash
python main_train.py --opt train_aripet_G1D30.json \
    --manifest splits/patients.csv --output runs/full --variant full
~~~

Only the source-training set defines the joint scanner--radiotracer class map.
The validation set reuses that map. The class count is determined from the
training manifest, rather than hard-coded to a tracer count. No scanner/tracer
label is supplied to the network at inference.

Default settings: channel widths 48/96/192/384, blocks 4/6/6/8, heads 1/2/4/8,
four refinement blocks, FFN expansion 2.66, bias-inclusive layer normalization,
10 prompt components with 32 x 32 stored resolution at L4/L3/L2, Adam
(betas 0.9/0.999, weight decay 0), learning rate 0.0002, batch size 2, 2,000,000
iterations, MultiStepLR factor 0.5 at 500,000 and 1,000,000 updates, and EMA 0.999.

Training stops at the configured iteration count. Validation uses the EMA
network, whole 3D SUV volumes and patient-averaged scores. Selection uses
source-validation PSNR only; the formal test set is not evaluated by the
training entry point. The configuration, domain map, manifest/code hashes,
optimizer, scheduler, RNG state, step, and EMA state are saved with checkpoints.
Run directories and checkpoint files are created exclusively; existing runs
are never pruned or overwritten. Resume into a new directory:

~~~bash
python main_train.py --manifest splits/patients.csv --output runs/full_resume \
    --resume runs/full/models/50000.pth --variant full
~~~

Use "--max-steps" only for shorter execution checks. The actual completed step
is recorded in each checkpoint. "checkpoints.jsonl" records which checkpoint
improved the source-validation score.

## Objective and ablations

The SAME objective is used by the training wrapper and standalone loss helper:

~~~text
total = 15 * Charbonnier(pred, ASC)
      + Charbonnier(FFT(pred), FFT(ASC))
      + 0.1 * cross_entropy(domain_logits, source_domain_label)
      + 0.01 * squared_normalized_inner_product(shared, specific)
~~~

Charbonnier is mean(sqrt(error^2 + 1e-9)); the Fourier reconstruction term
compares the real/imaginary channels. All enabled terms participate in
backpropagation.

| --variant | Classification | Orthogonality | Spatial prompt | Fourier prompt |
|---|---:|---:|---|---|
| full | 0.1 | 0.01 | enabled | enabled |
| noce | 0 | 0.01 | enabled | enabled |
| no_sd | 0 | 0 | enabled | enabled |
| no_spatial_prompt | 0.1 | 0.01 | bypassed | enabled |
| no_fourier_prompt | 0.1 | 0.01 | enabled | no prompt gate |

"NoCE" does not remove the orthogonality loss. Removing SD losses retains the
two-stream architecture. Removing the Fourier prompt retains the Fourier
feature branch without prompt-conditioned gating.

## Shared normalization and SUV restoration

For each resized NASC slice L, s = max(abs(L)), with s = 1 for an all-zero
slice. The network input is L/s and the paired training target is H/s.
ASC can exceed one after normalization. Neither ASC nor population statistics
are used to determine s. Predictions are restored by multiplying by that same
NASC-derived s, then resizing to the original in-plane grid. There is no
target-based inverse scaling and no output clipping to a display window.

## Fourier frequency coordinates

PALHBlock uses unshifted rfft2 with orthonormal normalization. Coordinates and
masks use 2*fftfreq(H) and 2*rfftfreq(W), where W is the original spatial width.
The radius is sqrt(fy^2 + fx^2), with low frequencies at radius <= 0.5 and the
complement assigned to the high band. DC (0,0) is low frequency. Vertical
positive and negative frequencies have matching masks; Nyquist is high.
The coordinate channels use the same unshifted frequency ordering as the masks.
The vertical coordinates lie in [-1, 1), while the horizontal half-spectrum
coordinates lie in [0, 1] and contain no negative frequencies. The horizontal
Nyquist endpoint is present only for even spatial widths. Odd spatial widths
are handled explicitly, including the 25 x 25 bottleneck.

Amplitude is normalized per sample/channel. Two coordinate-augmented branches
produce band updates, modulated by sigmoid(abs(rfft2(prompt))). A separate
learnable phase branch updates phase. L4 prompts use the specific latent
stream, and L3/L2 prompts use the corresponding decoder features.

## Multi-source held-out protocols

Starting from a manifest with the formal test patients already frozen, prepare
each source development split independently:

~~~bash
python tools/prepare_protocol.py --manifest splits/patients.csv --setting UIH --output splits/uih.csv
python tools/prepare_protocol.py --manifest splits/patients.csv --setting MFBG --output splits/mfbg.csv
python tools/prepare_protocol.py --manifest splits/patients.csv --setting SinoUnion-FDG --output splits/sino_fdg.csv
~~~

UIH excludes all UIH data; MFBG excludes all MFBG data; SinoUnion-FDG excludes
that entire scanner--radiotracer combination from training and validation.
Target-domain development patients are excluded, not reassigned to testing.
The target test cohort is drawn only from the existing formal test split.
The remaining source development patients are split at floor(7*N/8) for
training, with the remainder used for validation. Every training domain is
represented in source training. Use the SAME output manifest for Full, NoCE
and comparator runs. Patient counts are calculated from the supplied manifest,
not generated from the aggregate manuscript counts.

## EMA inference and evaluation

~~~bash
python main_test.py --checkpoint runs/full/models/50000.pth \
    --manifest splits/patients.csv --output predictions/full --device cuda:0
~~~

The default inference loader requires the explicitly saved "params_ema" state
and matching normalization/frequency-grid metadata. It does not silently
substitute ordinary generator weights for EMA. Each MAT output contains the
complete native-grid SUV volume and NASC-derived scales. A reconstruction
manifest records hashes and successful readback. Existing outputs are preserved.

Add "--metrics" to evaluate saved outputs against ASC. PSNR, SSIM, nMAE (%),
and NMSE are computed on complete 3D SUV volumes; repeated examinations are
averaged within patients. ASC is used as a metric reference, never as a scale
source for reconstruction. Organ/lesion masks and reader-study analyses are
separate downstream evaluations, not part of this inference entry point.

## Verification

~~~bash
python -m unittest discover -s tests -v
~~~

Tests cover objective gradients and ablations, input-only scaling and inverse
scaling, patient/domain partition integrity, DC/Nyquist and odd-size masks,
EMA save/load/inference, full-volume outputs, and source-only model selection.
