"""Seeing a 3D world: colour volumes, a ray-marched orbit view, and flat slices.

Everything is computed on the world's device (the GPU on a CUDA machine) and only the
final image is copied to the CPU.

Views (what each voxel looks like):
  life      lineage colour, lit by the light that actually reaches the cell (shaded
            cells are dark, so a canopy reads as a bright roof over a dim interior)
  bodies    one colour per organism
  clades    one colour per founding lineage
  light     the light field (cells drawn faintly)
  nutrient  nutrient in the water (cells drawn faintly)
  photo     energy each cell made by photosynthesis last step
  energy    energy held
  age       age
"""
import math
import torch
import torch.nn.functional as F

VIEWS = ["life", "bodies", "clades", "light", "nutrient", "photo", "energy", "age"]


def _hue(x):
    x = ((x % 2147483647) * 2654435761) % 4294967296
    h = (x.float() / 4294967296.0)[..., None]
    k = torch.tensor([0.0, 2.0 / 3.0, 1.0 / 3.0], device=x.device)
    return (0.5 + 0.5 * torch.cos(6.2832 * (h + k))).clamp(0, 1)


@torch.no_grad()
def volume(world, view="life", cell_alpha=0.55):
    """RGBA volume (4, D, H, W) in [0, 1]."""
    a = world.alive
    af = a.float()
    cfg = world.cfg
    lit = (world.light / (cfg.light_base + cfg.light_sun)).clamp(0, 1)
    if view == "life":
        col = world.color[world.gid.clamp(min=0)]                         # D,H,W,3
        shade = (0.18 + 0.82 * lit.sqrt())[..., None]
        rgb = col * shade
        alpha = af * cell_alpha
    elif view == "bodies":
        rgb = _hue(world.body.clamp(min=0)) * (0.35 + 0.65 * lit.sqrt())[..., None]
        alpha = af * cell_alpha
    elif view == "clades":
        cl = world.pool.clade[world.gid.clamp(min=0)].clamp(min=0)
        rgb = _hue(cl) * (0.35 + 0.65 * lit.sqrt())[..., None]
        alpha = af * cell_alpha
    elif view == "light":
        rgb = torch.stack([lit, lit * 0.9, lit * 0.4 + 0.1], -1)
        alpha = 0.04 * lit + af * 0.25
        rgb = torch.where(a[..., None], torch.tensor([0.3, 0.8, 0.4], device=a.device), rgb)
    elif view == "nutrient":
        m = (world.M / cfg.nutrient_max).clamp(0, 1)
        rgb = torch.stack([0.9 * m, 0.45 * m, 0.15 + 0.2 * m], -1)
        alpha = 0.035 * m + af * 0.25
        rgb = torch.where(a[..., None], torch.tensor([0.3, 0.8, 0.4], device=a.device), rgb)
    elif view == "photo":
        p = (world.photo / (cfg.photo_yield * 0.4 + 1e-9)).clamp(0, 1)
        rgb = torch.stack([p, p, 0.15 + 0.3 * (1 - p)], -1)
        alpha = af * cell_alpha
    elif view == "energy":
        e = (world.energy / cfg.e_max).clamp(0, 1)
        rgb = torch.stack([e, e ** 2, 0.25 * torch.ones_like(e)], -1)
        alpha = af * cell_alpha
    elif view == "age":
        ag = (world.age / (cfg.age_onset + cfg.age_span)).clamp(0, 1)
        rgb = torch.stack([ag, 0.4 * (1 - ag), 1 - ag], -1)
        alpha = af * cell_alpha
    else:
        raise ValueError(view)
    out = torch.cat([rgb.permute(3, 0, 1, 2), alpha[None]], 0)
    if view not in ("light", "nutrient"):
        # the sea floor: sand lit by whatever light gets through, so the life above casts
        # visible shadows on it, tinted by the nutrient seeping out of it
        fl = (world.light[0] / (cfg.light_base + cfg.light_sun) / float(world.depth_att[0, 0, 0])).clamp(0, 1)
        m = (world.M[0] / cfg.nutrient_max).clamp(0, 1)
        sand = torch.stack([0.55 + 0.1 * m, 0.45 + 0 * m, 0.30 - 0.1 * m])
        floor = sand * (0.12 + 0.6 * fl)[None]
        empty0 = ~a[0]
        out[:3, 0] = torch.where(empty0[None], floor, out[:3, 0])
        out[3, 0] = torch.where(empty0, torch.full_like(fl, 0.95), out[3, 0])
    return out.clamp(0, 1)


def _rot(yaw, pitch):
    """Camera basis (right, up, forward) in world axes (x, y, z) with z pointing up."""
    cy, sy, cp, sp = math.cos(yaw), math.sin(yaw), math.cos(pitch), math.sin(pitch)
    fwd = torch.tensor([cy * cp, sy * cp, -sp])          # looking down at the box when pitch > 0
    right = torch.tensor([-sy, cy, 0.0])
    up = torch.cross(fwd, right, dim=0)
    return right, up, fwd


@torch.no_grad()
def raymarch(vol, yaw=0.6, pitch=0.5, zoom=1.0, size=(480, 480), cut=None, steps=None,
             background=(0.05, 0.06, 0.08)):
    """Orthographic ray march through an RGBA volume (4, D, H, W).

    cut = (axis, frac, keep_low): hide everything beyond a plane, e.g. ("y", 0.5, True)
    keeps the half with y < 0.5 so the inside of the box shows.
    """
    dev = vol.device
    _, D, H, W = vol.shape
    if cut is not None:
        ax, frac, low = cut
        n = {"x": W, "y": H, "z": D}[ax]
        i = torch.arange(n, device=dev, dtype=torch.float32)
        keep = (i < frac * n) if low else (i >= frac * n)
        shape = {"x": (1, 1, n), "y": (1, n, 1), "z": (n, 1, 1)}[ax]
        vol = vol.clone()
        vol[3] = vol[3] * keep.float().view(shape)
    ext = torch.tensor([W, H, D], dtype=torch.float32, device=dev)
    radius = 0.5 * float(ext.norm())
    right, up, fwd = (v.to(dev) for v in _rot(yaw, pitch))
    w, h = size
    half = radius / zoom
    us = torch.linspace(-half, half, w, device=dev)
    vs = torch.linspace(half * h / w, -half * h / w, h, device=dev)
    V, U = torch.meshgrid(vs, us, indexing="ij")
    origin = U[..., None] * right + V[..., None] * up - radius * fwd          # (h, w, 3)
    n_steps = steps or int(2 * radius)
    ts = torch.linspace(0, 2 * radius, n_steps, device=dev)
    dt = 2 * radius / n_steps
    inp = vol[None]                                                           # 1,4,D,H,W
    acc = torch.zeros(4 - 1, h, w, device=dev)
    T = torch.ones(h, w, device=dev)
    chunk = 32
    for s0 in range(0, n_steps, chunk):
        t = ts[s0:s0 + chunk]
        pts = origin[None] + t[:, None, None, None] * fwd                     # (c, h, w, 3) in voxels
        grid = (pts / (0.5 * ext)).clamp(-1.5, 1.5)                          # x,y,z -> W,H,D
        smp = F.grid_sample(inp, grid[None], mode="nearest", padding_mode="zeros",
                            align_corners=False)[0]                          # 4, c, h, w
        a = 1 - (1 - smp[3]).clamp(0, 1) ** dt                               # step-size corrected
        for j in range(a.shape[0]):
            wgt = T * a[j]
            acc = acc + wgt[None] * smp[:3, j]
            T = T * (1 - a[j])
        if float(T.max()) < 0.01:
            break
    bg = torch.tensor(background, device=dev)[:, None, None]
    # a faint gradient so the empty water reads as depth
    grad = torch.linspace(1.15, 0.85, h, device=dev)[None, :, None]
    img = acc + T[None] * bg * grad
    return (img.clamp(0, 1) * 255).to(torch.uint8).permute(1, 2, 0)


def box_corners_2d(cfg, yaw, pitch, zoom, size):
    """Pixel positions of the 8 box corners (for drawing the box outline)."""
    right, up, fwd = _rot(yaw, pitch)
    ext = torch.tensor([cfg.W, cfg.H, cfg.D], dtype=torch.float32)
    radius = 0.5 * float(ext.norm())
    half = radius / zoom
    w, h = size
    out = []
    for sx in (-1, 1):
        for sy in (-1, 1):
            for sz in (-1, 1):
                p = 0.5 * ext * torch.tensor([sx, sy, sz], dtype=torch.float32)
                u, v = float(p @ right), float(p @ up)
                out.append(((u / half + 1) * 0.5 * w, (1 - v / (half * h / w)) * 0.5 * h))
    return out


EDGES = [(0, 1), (2, 3), (4, 5), (6, 7), (0, 2), (1, 3), (4, 6), (5, 7), (0, 4), (1, 5), (2, 6), (3, 7)]


@torch.no_grad()
def slice_img(world, view, axis="side", index=None):
    """A flat cut through the volume: 'side' = vertical x-z slice at row y = index
    (surface at the top of the image), 'top' = horizontal slice at layer z = index."""
    vol = volume(world, view, cell_alpha=1.0)
    rgb, a = vol[:3], vol[3:4]
    bg = torch.tensor([0.05, 0.06, 0.08], device=vol.device)[:, None, None, None]
    img = rgb * a.clamp(min=0.0) + bg * (1 - a) if view not in ("light", "nutrient") else rgb + bg * 0.3
    cfg = world.cfg
    if axis == "side":
        i = cfg.H // 2 if index is None else index
        out = img[:, :, i, :].flip(1)          # 3, D, W  with the surface on top
    else:
        i = cfg.D - 1 if index is None else index
        out = img[:, i, :, :]                  # 3, H, W
    return (out.clamp(0, 1) * 255).to(torch.uint8).permute(1, 2, 0).cpu().numpy()


@torch.no_grad()
def render3d(world, view="life", yaw=0.6, pitch=0.5, zoom=1.0, size=(480, 480), cut=None,
             cell_alpha=0.55, steps=None):
    return raymarch(volume(world, view, cell_alpha), yaw, pitch, zoom, size, cut, steps).cpu().numpy()
