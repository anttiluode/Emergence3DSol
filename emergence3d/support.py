"""Fixed inert scaffolds and their 26-neighbour contact region.

The rods are environment, not organism genes. They occupy voxels and shade light;
adhesion, twining and drag relief are separate, ablatable laws. No hidden reward.
"""
import math

import torch
import torch.nn.functional as F


def make_scaffold(cfg, device):
    """Sparse, evenly spaced vertical columns, deterministically placed by geometry.

    The columns do not use the world RNG, so a same-seed grip on/off pair has
    exactly the same initial seed cells and random policy weights.
    """
    out = torch.zeros((cfg.D, cfg.H, cfg.W), dtype=torch.bool, device=device)
    n = max(0, int(cfg.scaffold_columns))
    if n == 0:
        return out
    ncol = math.ceil(math.sqrt(n))
    stop = min(cfg.D, max(1, math.ceil(cfg.D * cfg.scaffold_height)))
    for i in range(n):
        gy, gx = divmod(i, ncol)
        y = min(cfg.H - 1, int((gy + 0.5) * cfg.H / ncol))
        x = min(cfg.W - 1, int((gx + 0.5) * cfg.W / ncol))
        out[:stop, y, x] = True
    return out


def contact_shell(scaffold):
    """The 26-neighbour Chebyshev shell of a 3D scaffold.

    This shell makes an axial, six-direction growth lattice able to take
    stepped diagonal turns around a column. Edge voxels are not toroidally
    padded here, whereas x/y cell motion itself is toroidal. Columns are
    deliberately placed away from the side walls.
    """
    near = F.max_pool3d(scaffold[None, None].float(), kernel_size=3, stride=1,
                        padding=1)[0, 0].bool()
    return near & ~scaffold


def drag_loss(height, grip, cfg):
    """One-cell energy loss. Grip can reduce *current stress*, not create energy."""
    exposure = cfg.current_drag * height.clamp(0, 1).pow(cfg.current_power)
    relief = (1.0 - cfg.grip_relief * grip.float()).clamp(min=0)
    return exposure * relief + cfg.grip_cost * grip.float()
