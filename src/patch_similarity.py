"""Helpers for DINO-style patch-level self-similarity: given a (grid_h, grid_w,
embed_dim) token grid from `vfm_encoders.embed_image_patches`, compare one query
patch against every other patch (within the same image, or across a gallery of
images) by cosine similarity.
"""

import numpy as np


def cosine_similarity_grid(query: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """query: (embed_dim,) patch vector. grid: (H, W, embed_dim) patch tokens.
    Returns (H, W) cosine similarity of every grid patch to `query`, in [-1, 1]."""
    query_n = query / (np.linalg.norm(query) + 1e-8)
    grid_n = grid / (np.linalg.norm(grid, axis=-1, keepdims=True) + 1e-8)
    return grid_n @ query_n


def upsample_heatmap(sim_grid: np.ndarray, patch_size: int) -> np.ndarray:
    """Block-upsamples a (H, W) similarity grid by `patch_size` in both axes,
    for overlay on the resized input image. Each ViT patch token maps to an
    exact `patch_size`-px block, so this is a block repeat, not interpolation."""
    return np.repeat(np.repeat(sim_grid, patch_size, axis=0), patch_size, axis=1)


def patch_crop(image_arr: np.ndarray, row: int, col: int, patch_size: int) -> np.ndarray:
    """Crops the `patch_size`-px block at grid position (row, col) out of a
    (H, W, 3) image array -- the pixel region a single patch token covers."""
    r0, c0 = row * patch_size, col * patch_size
    return image_arr[r0 : r0 + patch_size, c0 : c0 + patch_size]


def get_normalize_stats(transform) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Pulls (mean, std) out of a timm `create_transform` Compose's Normalize
    step, so denormalize() matches each encoder's own preprocessing exactly
    instead of assuming a fixed ImageNet-style normalization."""
    for t in transform.transforms:
        if hasattr(t, "mean") and hasattr(t, "std"):
            return tuple(t.mean), tuple(t.std)
    raise ValueError("transform has no Normalize step")


def denormalize(tensor, mean, std) -> np.ndarray:
    """Undoes the encoder transform's normalization for display: (3, H, W)
    tensor -> (H, W, 3) uint8 array. Get `mean`/`std` from `get_normalize_stats`."""
    arr = tensor.cpu().numpy().transpose(1, 2, 0)
    arr = arr * np.array(std) + np.array(mean)
    return np.clip(arr * 255, 0, 255).astype(np.uint8)
