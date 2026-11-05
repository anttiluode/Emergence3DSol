"""Matched support-economy experiment; writes raw metrics, never invented winners.

Each world has independently reinitialized RNGs and weights with the same supplied
seed. In the first three arms the inert rods, light model and initial state match.
Only grip benefit and phase-guided growth change.
"""
import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path

from emergence3d.config import Config
from emergence3d.world import World

ARMS = (
    ("vine", True, 1.0, True),
    ("grip_no_twine", True, 0.0, True),
    ("no_grip", False, 0.0, True),
    ("no_rods", True, 1.0, False),
)


def write_csv(path, records):
    if not records:
        return
    with open(path, "w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(records[0]))
        wr.writeheader()
        wr.writerows(records)


def run(seed, arm, args):
    name, grip, twine, rods = arm
    cfg = Config(
        D=args.depth, H=args.size, W=args.size, seed=seed, device=args.device,
        seed_cells=args.seed_cells, scaffold_columns=args.rods if rods else 0,
        grip_enabled=grip, twine_strength=twine,
        current_drag=args.drag,
    )
    world = World(cfg)
    records = []
    while world.t < args.steps:
        if args.sever_at >= 0 and world.t == args.sever_at:
            world.sever_scaffolds()
        world.step()
        if world.t % args.log_every == 0 or world.t == args.steps:
            m = world.metrics()
            records.append({"seed": seed, "arm": name, **m})
            print(f"seed={seed} arm={name:14} step={world.t:7} "
                  f"pop={m['population']:6} grip={m['grip_fraction']:.3f} "
                  f"high={m['high_cell_fraction']:.3f}")
    stem = f"{name}_seed{seed}"
    write_csv(args.out / f"{stem}.csv", records)
    if args.checkpoints:
        import torch
        torch.save(world.state_dict(), args.out / f"{stem}.pt")
    return records[-1]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--log-every", type=int, default=200)
    ap.add_argument("--depth", type=int, default=24)
    ap.add_argument("--size", type=int, default=48)
    ap.add_argument("--seed-cells", type=int, default=1200)
    ap.add_argument("--rods", type=int, default=12)
    ap.add_argument("--drag", type=float, default=0.008)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--sever-at", type=int, default=-1)
    ap.add_argument("--out", type=Path, default=Path("runs/support_comparison"))
    ap.add_argument("--checkpoints", action="store_true")
    args = ap.parse_args()
    if args.log_every < 1 or args.steps < 1:
        ap.error("--log-every and --steps must be positive")
    args.out.mkdir(parents=True, exist_ok=True)
    summary = [run(seed, arm, args) for seed in args.seeds for arm in ARMS]
    write_csv(args.out / "summary.csv", summary)
    with open(args.out / "protocol.json", "w", encoding="utf-8") as fh:
        d = vars(args).copy()
        d["out"] = str(args.out)
        json.dump({"settings": d, "arms": [a[0] for a in ARMS]}, fh, indent=2)
    print("Saved", args.out / "summary.csv")
    print("No causal claim follows from a single seed. Compare several seeds and "
          "matched controls for survival, light capture, height and grip occupancy.")


if __name__ == "__main__":
    main()
