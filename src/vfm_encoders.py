"""Shared loaders for the pathology foundation-model encoders used across the
embedding-generation notebooks (UNI2-h and Virchow2).

Both models are gated on Hugging Face:
  - https://huggingface.co/MahmoodLab/UNI2-h
  - https://huggingface.co/paige-ai/Virchow2
Request access on each model page with the account you'll authenticate as, then
either set HF_TOKEN in the environment or run `huggingface_hub.login()` before
loading a model (the notebooks do this for you in the setup cell).
"""

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
import torch
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from tqdm.auto import tqdm


@dataclass
class Encoder:
    name: str
    model: torch.nn.Module
    transform: Callable
    embed_dim: int
    # maps a raw model forward pass to a single (B, embed_dim) embedding tensor
    pool: Callable[[torch.Tensor], torch.Tensor]


def load_uni2(device: str = "cuda") -> Encoder:
    """UNI2-h (MahmoodLab): ViT-H/14 w/ 8 register tokens, trained on ~200M pathology
    patches. `num_classes=0` gives a bare CLS embedding straight out of the model."""
    timm_kwargs = dict(
        img_size=224,
        patch_size=14,
        depth=24,
        num_heads=24,
        init_values=1e-5,
        embed_dim=1536,
        mlp_ratio=2.66667 * 2,
        num_classes=0,
        no_embed_class=True,
        mlp_layer=timm.layers.SwiGLUPacked,
        act_layer=torch.nn.SiLU,
        reg_tokens=8,
        dynamic_img_size=True,
    )
    model = timm.create_model("hf-hub:MahmoodLab/UNI2-h", pretrained=True, **timm_kwargs)
    model = model.eval().to(device)
    transform = create_transform(**resolve_data_config(model.pretrained_cfg, model=model))

    return Encoder(
        name="uni2-h", model=model, transform=transform, embed_dim=1536, pool=lambda out: out
    )


def load_virchow2(device: str = "cuda") -> Encoder:
    """Virchow2 (Paige): ViT-H/14 w/ 4 register tokens. The model card's recommended
    embedding is [CLS] concatenated with the mean-pooled patch tokens (tokens 5: onward,
    since tokens 1-4 are registers, not patches) -> 1280 + 1280 = 2560-d."""
    model = timm.create_model(
        "hf-hub:paige-ai/Virchow2",
        pretrained=True,
        mlp_layer=timm.layers.SwiGLUPacked,
        act_layer=torch.nn.SiLU,
    )
    model = model.eval().to(device)
    transform = create_transform(**resolve_data_config(model.pretrained_cfg, model=model))

    def pool(out: torch.Tensor) -> torch.Tensor:
        class_token = out[:, 0]
        patch_tokens = out[:, 5:]
        return torch.cat([class_token, patch_tokens.mean(dim=1)], dim=-1)

    return Encoder(name="virchow2", model=model, transform=transform, embed_dim=2560, pool=pool)


@torch.inference_mode()
def embed_images(
    encoder: Encoder,
    images: Sequence,
    device: str = "cuda",
    batch_size: int = 64,
    show_progress: bool = True,
) -> np.ndarray:
    """images: a sequence of RGB PIL.Image patches. Returns (N, encoder.embed_dim) float32."""
    out = np.empty((len(images), encoder.embed_dim), dtype=np.float32)
    steps = range(0, len(images), batch_size)
    if show_progress:
        steps = tqdm(steps, total=len(steps), desc=f"{encoder.name} embeddings")

    for start in steps:
        chunk = images[start : start + batch_size]
        batch = torch.stack([encoder.transform(img) for img in chunk]).to(device)
        embedding = encoder.pool(encoder.model(batch))
        out[start : start + len(chunk)] = embedding.float().cpu().numpy()

    return out
