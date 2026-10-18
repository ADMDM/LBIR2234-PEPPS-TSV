import os
from pathlib import Path

# Configure caches before importing SMP/HuggingFace, which cache these settings.
_cache = Path(__file__).resolve().parents[1] / ".cache"
os.environ.setdefault("TORCH_HOME", str(_cache / "torch"))
os.environ.setdefault("HF_HOME", str(_cache / "huggingface"))
os.environ.setdefault("HF_HUB_CACHE", str(_cache / "huggingface" / "hub"))
import torch
from torch import nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp


def build_model(classes=2, pretrained=False):
    return smp.DeepLabV3Plus(encoder_name="resnet50", encoder_weights="imagenet" if pretrained else None,
                            encoder_output_stride=16, in_channels=3, classes=classes, activation=None)


def freeze_batchnorm(model):
    # ASPP global pooling has spatial size 1x1; batch=1 needs frozen BN statistics.
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            module.eval()


def dice_ce(logits, target):
    valid = target != 255
    if not valid.any():
        return logits.sum() * 0
    ce = F.cross_entropy(logits, target, ignore_index=255)
    safe = target.masked_fill(~valid, 0)
    truth = F.one_hot(safe, logits.shape[1]).permute(0, 3, 1, 2).float()
    prob = logits.softmax(dim=1) * valid[:, None]
    truth = truth * valid[:, None]
    inter = (prob * truth).sum((0, 2, 3))
    denom = prob.sum((0, 2, 3)) + truth.sum((0, 2, 3))
    # Foreground only; absent classes still penalize false positives.
    dice = (2 * inter[1:] + 1e-6) / (denom[1:] + 1e-6)
    return ce + 1 - dice.mean()


def load_checkpoint(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("format") != "railtsv-deeplab-v1":
        raise ValueError("Checkpoint TSV DeepLabV3+ attendu")
    return checkpoint


def transfer(model, path):
    checkpoint = load_checkpoint(path)
    # New TSV classifier is explicitly reinitialized even if shapes coincide.
    state = {k: v for k, v in checkpoint["state_dict"].items() if not k.startswith("segmentation_head.")}
    result = model.load_state_dict(state, strict=False)
    if result.unexpected_keys or any(not k.startswith("segmentation_head.") for k in result.missing_keys):
        raise ValueError(f"Transfert incompatible : {result}")
    return {"source_task": checkpoint["task"], "reinitialized": result.missing_keys}
