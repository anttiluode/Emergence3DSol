"""Emergence3D - run the 3D world.

    python run.py                                   # live viewer (pygame), default 40 x 144 x 144
    python run.py --depth 48 --size 192             # bigger box (depth layers, width)
    python run.py --headless --steps 200000 --out runs/night1 --video
    python run.py --load runs/night1/ckpt_0200000.pt
    python run.py --set regulation=0 --set share_rate=0     # change a law

Viewer keys
    mouse drag     rotate the box (3D view)       mouse wheel   zoom (3D) / brush size (slices)
    arrows         rotate                         O             slow auto-orbit on/off
    TAB            3D view -> side slice -> top slice
    C              cut the box open: none -> front half -> side half -> top layers
    [ ]            move the cut plane (3D) or the slice (slices)
    , .            cells more / less transparent
    G              render quality (internal resolution)
    1..8           view: life, bodies, clades, light, nutrient, photo, energy, age
    SPACE pause    + / -  steps per frame
    slices: left drag wipes a ball of cells, F + drag pours nutrient, right click inspects a genome
    R  500 random new cells     S save     P screenshot     V video on/off     H panel     Q/ESC quit
"""
import argparse
import csv
import math
import os
import time

import numpy as np
import torch

from emergence3d.config import Config
from emergence3d.world import World
from emergence3d.render import VIEWS, render3d, slice_img, box_corners_2d, EDGES


def build(args):
    cfg = Config()
    if args.size:
        cfg.H = cfg.W = args.size
    if args.depth:
        cfg.D = args.depth
    cfg.seed = args.seed
    for kv in args.set or []:
        k, v = kv.split("=", 1)
        old = getattr(cfg, k)
        setattr(cfg, k, type(old)(v) if not isinstance(old, bool) else v.lower() in ("1", "true"))
    if args.load:
        d = torch.load(args.load, map_location="cpu", weights_only=False)
        saved = Config.from_dict(d["config"])
        for kv in args.set or []:
            k, v = kv.split("=", 1)
            setattr(saved, k, type(getattr(saved, k))(v))
        saved.device = cfg.device
        cfg = saved
        keep = cfg.seed_cells
        cfg.seed_cells = 0
        w = World(cfg)
        cfg.seed_cells = keep
        w.load_state_dict(d)
        print(f"loaded {args.load} at t={w.t}")
    else:
        w = World(cfg)
    print(f"device={w.device}  box={cfg.D} deep x {cfg.H} x {cfg.W}  "
          f"cell net {w.n_percept}->{cfg.hidden}->{w.n_actions}  "
          f"translator d={cfg.d_model} x{cfg.n_layers}")
    return w


def fmt(m):
    keys = ["t", "population", "bodies", "body_size_cellweighted", "body_height_cellweighted",
            "tall_fraction", "mean_height", "living_layer", "genomes", "clades", "mean_depth",
            "light_captured", "photo_per_cell", "reseeds"]
    return "  ".join(f"{k}={m[k]:.3g}" if isinstance(m.get(k), float) else f"{k}={m.get(k)}"
                     for k in keys if k in m)


def framed(w, yaw, pitch, size, zoom=1.05, view="life", cut=None):
    """3D render with the box outline drawn on it."""
    from PIL import Image, ImageDraw
    img = Image.fromarray(render3d(w, view, yaw, pitch, zoom, (size, size), cut))
    dr = ImageDraw.Draw(img)
    pts = box_corners_2d(w.cfg, yaw, pitch, zoom, (size, size))
    for i, j in EDGES:
        dr.line([pts[i], pts[j]], fill=(105, 120, 150), width=1)
    return np.array(img)


def snapshot(w, size=360):
    """Four panels: life from two angles, the box cut open, and a side slice."""
    a = framed(w, 0.7, 0.45, size)
    b = framed(w, 0.7 + math.pi, 0.15, size)
    c = framed(w, 0.7, 0.45, size, cut=("y", 0.5, True))
    s = slice_img(w, "life", "side")
    from PIL import Image
    s = np.array(Image.fromarray(s).resize((size, max(1, int(size * s.shape[0] / s.shape[1]))), Image.NEAREST))
    pad = np.zeros((size, size, 3), np.uint8) + 13
    y0 = (size - s.shape[0]) // 2
    pad[y0:y0 + s.shape[0]] = s
    return np.concatenate([a, b, c, pad], 1)


# ---------------------------------------------------------------------- headless
def headless(w, args):
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "config.json"), "w") as f:
        f.write(w.cfg.to_json())
    writer = None
    if args.video:
        import imageio
        writer = imageio.get_writer(os.path.join(args.out, "life3d.mp4"), fps=30, quality=8,
                                    macro_block_size=1)
    fh = open(os.path.join(args.out, "metrics.csv"), "a", newline="")
    cw = None
    t0, s0 = time.time(), w.t
    end = w.t + args.steps
    while w.t < end:
        w.step()
        if w.t % args.log_every == 0:
            m = w.metrics()
            m["steps_per_sec"] = (w.t - s0) / (time.time() - t0)
            if cw is None:
                cw = csv.DictWriter(fh, fieldnames=list(m.keys()), extrasaction="ignore")
                if fh.tell() == 0:
                    cw.writeheader()
            cw.writerow(m)
            fh.flush()
            print(fmt(m) + f"  ({m['steps_per_sec']:.1f} steps/s)", flush=True)
        if writer is not None and w.t % args.video_every == 0:
            yaw = 0.7 + 0.002 * w.t
            writer.append_data(framed(w, yaw, 0.45, 480))
        if w.t % args.snap_every == 0:
            from PIL import Image
            Image.fromarray(snapshot(w)).save(os.path.join(args.out, f"snap_{w.t:07d}.png"))
        if w.t % args.ckpt_every == 0:
            torch.save(w.state_dict(), os.path.join(args.out, f"ckpt_{w.t:07d}.pt"))
    torch.save(w.state_dict(), os.path.join(args.out, f"ckpt_{w.t:07d}.pt"))
    if writer is not None:
        writer.close()
    fh.close()


# ---------------------------------------------------------------------- viewer
LABELS = [("population", "cells"), ("bodies", "organisms"),
          ("body_size_cellweighted", "body size (per cell)"), ("max_body", "largest body"),
          ("body_height_cellweighted", "body height (layers)"), ("tall_fraction", "in tall bodies"),
          ("mean_height", "mean height of life"), ("living_layer", "busiest layer"),
          ("light_captured", "light captured"), ("photo_per_cell", "photosynthesis/cell"),
          ("nutrient_water", "nutrient in water"),
          ("genomes", "genomes alive"), ("clades", "founding clades"),
          ("mean_depth", "mutations (mean)"), ("mean_genome_len", "genome length"),
          ("kin_neighbours", "body-mate neighbours"), ("stranger_neighbours", "stranger neighbours"),
          ("top_genome_share", "top genome share"), ("mean_energy", "energy / cell"),
          ("mean_age", "age / cell"), ("genome_table_used", "genome table used"),
          ("reseeds", "reseeds")]
QUALITY = [256, 384, 512, 640]


def viewer(w, args):
    import pygame
    pygame.init()
    cfg = w.cfg
    info = pygame.display.Info()
    sw, sh = (info.current_w, info.current_h) if info.current_w > 0 else (1920, 1080)
    VS = int(min(sh - 110, sw - 420))                # square main view
    panel = 380
    os.environ.setdefault("SDL_VIDEO_CENTERED", "1")
    screen = pygame.display.set_mode((VS + panel, VS))
    pygame.display.set_caption("Emergence3D")
    font = pygame.font.SysFont("consolas,menlo,monospace", 15)
    small = pygame.font.SysFont("consolas,menlo,monospace", 12)
    clock = pygame.time.Clock()

    view, paused, spf, show_panel = "life", False, 1, True
    mode = "3d"                                       # 3d | side | top
    yaw, pitch, zoom, orbit = 0.7, 0.45, 1.05, False
    cuts = [None, ("y", 0.5, True), ("x", 0.5, True), ("z", 0.75, True)]
    cut_i, cut_frac = 0, 0.5
    alpha, q = 0.55, 1
    side_i, top_i, brush = cfg.H // 2, cfg.D - 1, 6
    feeding, inspect, rec = False, None, None
    dragging, last_mouse = False, (0, 0)
    hist = {k: [] for k in ["population", "genomes", "body_height_cellweighted", "mean_height",
                            "light_captured"]}
    last_m, t_rate, s_rate, rate, t_metrics = w.metrics(), time.time(), w.t, 0.0, 0.0
    os.makedirs(args.out, exist_ok=True)

    def slice_to_world(pos):
        """Mouse position in a slice view -> (z, y, x) of the voxel."""
        rows = cfg.D if mode == "side" else cfg.H
        off = (VS - VS * rows / cfg.W) / 2                 # the slice is centred vertically
        x = int(pos[0] * cfg.W / VS)
        r = int((pos[1] - off) * cfg.W / VS)
        if mode == "side":
            return cfg.D - 1 - r, side_i, x
        return top_i, r, x

    running = True
    while running:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                running = False
            elif ev.type == pygame.KEYDOWN:
                k = ev.key
                if k in (pygame.K_ESCAPE, pygame.K_q):
                    running = False
                elif pygame.K_1 <= k <= pygame.K_8:
                    view = VIEWS[k - pygame.K_1]
                elif k == pygame.K_SPACE:
                    paused = not paused
                elif k == pygame.K_TAB:
                    mode = {"3d": "side", "side": "top", "top": "3d"}[mode]
                elif k == pygame.K_c:
                    cut_i = (cut_i + 1) % len(cuts)
                    cut_frac = cuts[cut_i][1] if cuts[cut_i] else 0.5
                elif k in (pygame.K_LEFTBRACKET, pygame.K_RIGHTBRACKET):
                    s = -1 if k == pygame.K_LEFTBRACKET else 1
                    if mode == "3d":
                        cut_frac = float(np.clip(cut_frac + 0.05 * s, 0.05, 1.0))
                    elif mode == "side":
                        side_i = int(np.clip(side_i + 2 * s, 0, cfg.H - 1))
                    else:
                        top_i = int(np.clip(top_i + s, 0, cfg.D - 1))
                elif k == pygame.K_COMMA:
                    alpha = min(1.0, alpha + 0.1)
                elif k == pygame.K_PERIOD:
                    alpha = max(0.05, alpha - 0.1)
                elif k == pygame.K_g:
                    q = (q + 1) % len(QUALITY)
                elif k == pygame.K_o:
                    orbit = not orbit
                elif k == pygame.K_LEFT:
                    yaw -= 0.15
                elif k == pygame.K_RIGHT:
                    yaw += 0.15
                elif k == pygame.K_UP:
                    pitch = min(1.5, pitch + 0.1)
                elif k == pygame.K_DOWN:
                    pitch = max(-1.5, pitch - 0.1)
                elif k in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS):
                    spf = min(64, spf * 2)
                elif k in (pygame.K_MINUS, pygame.K_KP_MINUS):
                    spf = max(1, spf // 2)
                elif k == pygame.K_r:
                    w.seed_life(500)
                elif k == pygame.K_s:
                    p = os.path.join(args.out, f"ckpt_{w.t:07d}.pt")
                    torch.save(w.state_dict(), p); print("saved", p)
                elif k == pygame.K_p:
                    p = os.path.join(args.out, f"shot_{w.t:07d}.png")
                    pygame.image.save(screen, p); print("saved", p)
                elif k == pygame.K_f:
                    feeding = not feeding
                elif k == pygame.K_h:
                    show_panel = not show_panel
                elif k == pygame.K_v:
                    if rec is None:
                        import imageio
                        p = os.path.join(args.out, f"video_{w.t:07d}.mp4")
                        rec = imageio.get_writer(p, fps=30, quality=8, macro_block_size=1)
                        print("recording", p)
                    else:
                        rec.close(); rec = None; print("recording stopped")
            elif ev.type == pygame.MOUSEWHEEL:
                if mode == "3d":
                    zoom = float(np.clip(zoom * (1.1 ** ev.y), 0.5, 8))
                else:
                    brush = int(np.clip(brush + ev.y, 1, 40))
            elif ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1 and mode == "3d":
                dragging, last_mouse = True, ev.pos
            elif ev.type == pygame.MOUSEBUTTONUP and ev.button == 1:
                dragging = False
            elif ev.type == pygame.MOUSEMOTION and dragging and mode == "3d":
                dx, dy = ev.pos[0] - last_mouse[0], ev.pos[1] - last_mouse[1]
                yaw -= dx * 0.01
                pitch = float(np.clip(pitch + dy * 0.01, -1.5, 1.5))
                last_mouse = ev.pos
            elif ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 3 and mode != "3d":
                z, y, x = slice_to_world(ev.pos)
                if 0 <= z < cfg.D and 0 <= y < cfg.H and 0 <= x < cfg.W and bool(w.alive[z, y, x]):
                    inspect = int(w.gid[z, y, x])
        if mode != "3d" and pygame.mouse.get_pressed()[0] and pygame.mouse.get_pos()[0] < VS:
            z, y, x = slice_to_world(pygame.mouse.get_pos())
            if 0 <= z < cfg.D and 0 <= y < cfg.H and 0 <= x < cfg.W:
                (w.feed if feeding else w.wipe)(z, y, x, brush)

        if not paused:
            for _ in range(spf):
                w.step()
        if orbit:
            yaw += 0.006
        if time.time() - t_metrics > 0.5:
            t_metrics = time.time()
            last_m = w.metrics()
            for kk in hist:
                hist[kk].append(last_m.get(kk, 0)); hist[kk] = hist[kk][-300:]
        now = time.time()
        if now - t_rate > 1:
            rate = (w.t - s_rate) / (now - t_rate); t_rate, s_rate = now, w.t

        screen.fill((13, 15, 20))
        if mode == "3d":
            res = QUALITY[q]
            cut = None if cuts[cut_i] is None else (cuts[cut_i][0], cut_frac, cuts[cut_i][2])
            img = render3d(w, view, yaw, pitch, zoom, (res, res), cut, alpha)
            surf = pygame.surfarray.make_surface(np.ascontiguousarray(img.transpose(1, 0, 2)))
            surf = pygame.transform.smoothscale(surf, (VS, VS))
            screen.blit(surf, (0, 0))
            pts = [(x * VS / res, y * VS / res) for x, y in box_corners_2d(cfg, yaw, pitch, zoom, (res, res))]
            for i, j in EDGES:
                pygame.draw.line(screen, (70, 80, 100), pts[i], pts[j], 1)
        else:
            img = slice_img(w, view, mode, side_i if mode == "side" else top_i)
            surf = pygame.surfarray.make_surface(np.ascontiguousarray(img.transpose(1, 0, 2)))
            hh = int(VS * img.shape[0] / img.shape[1])
            surf = pygame.transform.scale(surf, (VS, hh))
            screen.blit(surf, (0, (VS - hh) // 2))
            mx, my = pygame.mouse.get_pos()
            if mx < VS:
                pygame.draw.circle(screen, (120, 255, 120) if feeding else (255, 255, 255),
                                   (mx, my), max(2, int(brush * VS / cfg.W)), 1)
        if rec is not None:
            rec.append_data(pygame.surfarray.array3d(screen).transpose(1, 0, 2)[:, :VS])

        if show_panel:
            x0, y = VS + 14, 10
            cuttxt = "none" if cuts[cut_i] is None else f"{cuts[cut_i][0]} at {cut_frac:.2f}"
            lines = [f"EMERGENCE 3D   t={w.t}",
                     f"view [{VIEWS.index(view) + 1}] {view}   mode {mode}",
                     f"{rate:.1f} steps/s   x{spf}/frame" + ("   PAUSED" if paused else ""),
                     (f"cut {cuttxt}   res {QUALITY[q]}   alpha {alpha:.2f}" if mode == "3d" else
                      f"slice {'y=' + str(side_i) if mode == 'side' else 'z=' + str(top_i)}   "
                      f"brush {brush} {'FEED' if feeding else 'wipe'}") + ("  REC" if rec else ""), ""]
            for kk, lab in LABELS:
                v = last_m.get(kk)
                if v is not None:
                    lines.append(f"{lab:<22}{v:.3g}" if isinstance(v, float) else f"{lab:<22}{v}")
            for ln in lines:
                screen.blit(font.render(ln, True, (220, 220, 225)), (x0, y)); y += 17
            y += 6
            for kk, series in hist.items():
                if len(series) > 1:
                    lo, hi = min(series), max(series)
                    pts = [(x0 + i * 340 / 300, y + 22 - 22 * (v - lo) / (hi - lo + 1e-9))
                           for i, v in enumerate(series)]
                    pygame.draw.lines(screen, (120, 200, 255), False, pts, 1)
                    screen.blit(small.render(f"{kk}  {lo:.3g}..{hi:.3g}", True, (150, 150, 160)), (x0, y + 24))
                y += 38
            if inspect is not None and int(w.pool.count[inspect]) > 0:
                p = w.pool
                screen.blit(font.render(f"genome #{inspect}  cells={int(p.count[inspect])}",
                                        True, (255, 230, 150)), (x0, y)); y += 18
                screen.blit(small.render(f"clade {int(p.clade[inspect])}  depth {int(p.depth[inspect])}"
                                         f"  len {int(p.length[inspect])}", True, (200, 200, 200)), (x0, y)); y += 16
                screen.blit(small.render(p.as_text(inspect)[:60], True, (200, 200, 200)), (x0, y)); y += 16
        pygame.display.flip()
        clock.tick(60)
    if rec is not None:
        rec.close()
    pygame.quit()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--size", type=int, default=0, help="width and length of the box (cells)")
    ap.add_argument("--depth", type=int, default=0, help="layers of water")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--set", action="append", help="override a law, e.g. --set regulation=0")
    ap.add_argument("--load", type=str, default="")
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--steps", type=int, default=100000)
    ap.add_argument("--out", type=str, default="runs/default")
    ap.add_argument("--log-every", type=int, default=500)
    ap.add_argument("--snap-every", type=int, default=10000)
    ap.add_argument("--ckpt-every", type=int, default=50000)
    ap.add_argument("--video", action="store_true")
    ap.add_argument("--video-every", type=int, default=20)
    args = ap.parse_args()
    w = build(args)
    if args.headless:
        headless(w, args)
    else:
        viewer(w, args)


if __name__ == "__main__":
    main()
