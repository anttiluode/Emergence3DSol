"""Depth profile of a saved world: where the cells are, and the light and nutrient they live in.

    python profile.py runs/night1/ckpt_0500000.pt          # writes <ckpt>_profile.png

Each curve is a mean over the whole layer, normalised to its own maximum, with the floor
at the bottom and the surface at the top. The living layer, if there is one, should sit
where the falling light meets the rising nutrient.
"""
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from emergence3d.config import Config
from emergence3d.world import World


def main():
    path = sys.argv[1]
    d = torch.load(path, map_location="cpu", weights_only=False)
    cfg = Config.from_dict(d["config"])
    cfg.device, cfg.seed_cells = "cpu", 0
    w = World(cfg)
    w.load_state_dict(d)
    if "photo" not in d:          # older checkpoints did not save it: take one step to refill it
        w.step()
    z = torch.arange(cfg.D)
    cells = w.alive.float().mean((1, 2))
    light = w.light.mean((1, 2))
    clear = (w.sky.mean() * w.depth_att[:, 0, 0])            # what the light would be with no life
    nut = w.M.mean((1, 2))
    photo = (w.photo.sum((1, 2)) / w.alive.float().sum((1, 2)).clamp(min=1))
    nrm = lambda v: (v / v.max().clamp(min=1e-9)).numpy()
    fig, ax = plt.subplots(figsize=(5, 6))
    ax.plot(nrm(clear), z, ":", color="gold", label="light, if the water were empty")
    ax.plot(nrm(light), z, "-", color="gold", label="light that gets through")
    ax.plot(nrm(nut), z, "-", color="sienna", label="nutrient in the water")
    ax.fill_betweenx(z.numpy(), 0, nrm(cells), color="tab:green", alpha=0.35, label="living cells")
    ax.plot(nrm(photo), z, "--", color="tab:green", label="photosynthesis per cell")
    ax.set_ylabel("layer (0 = floor)")
    ax.set_xlabel("normalised to each curve's maximum")
    ax.set_title(f"t = {w.t}")
    ax.legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    out = path.replace(".pt", "_profile.png")
    fig.savefig(out, dpi=110)
    print("wrote", out)
    for k in range(cfg.D - 1, -1, -1):
        print(f"z={k:>3}  cells {float(cells[k]):.3f}  light {float(light[k]):.3f}  "
              f"nutrient {float(nut[k]):.3f}  photo/cell {float(photo[k]):.4f}")


if __name__ == "__main__":
    main()
