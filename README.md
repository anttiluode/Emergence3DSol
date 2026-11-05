# Emergence3DSol — a vine-support ecology

A controlled extension of [anttiluode/Emergence3D](https://github.com/anttiluode/Emergence3D) (original 2026-11-05 version), not a replacement for Claude's parent world.

**Question:** Can inherited cell policies exploit **other structures** to live higher in a world where exposure is costly? Does a rotating exploratory growth bias help more than attachment alone?

The story began with a houseplant winding around a hanging chain. We are *not* asserting biological realism or that the computer evolved literal plants. This repository implements a falsifiable support economy in an existing genetic/ecological world.

## Start

Requires Python, PyTorch 2.1+, NumPy; the interactive viewer also needs Pygame and Pillow. For Windows + NVIDIA, install a CUDA-enabled PyTorch build appropriate for your system.

```bash
pip install -r requirements.txt
python -m pytest -q tests            # pip install pytest if necessary
python run.py --size 64 --depth 24
```

The viewer retains the original controls, **plus K to sever all inert columns**. In the 3-D view and slices, slate columns are inert supports; cells currently gripping a support are highlighted turquoise in the life view. Use the cutaway and slices to examine attachment and height.

A short, headless example:

```bash
python run.py --headless --size 48 --depth 24 --steps 10000 --out runs/vine --set device=auto
python run.py --headless --size 48 --depth 24 --steps 10000 --out runs/control --set grip_enabled=0 --set twine_strength=0
python run.py --headless --size 48 --depth 24 --steps 10000 --out runs/sever --sever-at 5000
```

For matched, automatically seeded controls and a combined CSV:

```bash
python experiment.py --seeds 0 1 2 --depth 24 --size 48 --steps 4000 --out runs/support_comparison
```

This runs four conditions with the same starting seed and preexisting code:

| Arm | Inert rods | Adhesion relief | Phase growth bias |
|---|---|---|---|
| `vine` | yes | yes | yes |
| `grip_no_twine` | yes | yes | no |
| `no_grip` | yes | no | no |
| `no_rods` | no | yes (possible contact with foreign living bodies) | yes |

The first three retain **identical physical rod obstacles and shading**. No-grip is the primary causal control for the benefit of attachment. The no-twine arm asks whether the rotating bias adds value. No-rods is a different environment, not a pure adhesion ablation. Altered reproduction can cause trajectories to diverge despite matched initial seeds.

## What the vine modification actually does

- **Geometry:** slender, fixed, evenly spaced vertical rods are occupied voxels, so a cell cannot divide into one. They shade light. A 26-neighbor shell around each rod is precomputed; cell division still uses the original six face-neighbors.
- **Sensing:** the cellular policy sees proximity to rods and proximity to adjacent *foreign living bodies*. These add two dimensions to the original 45-feature perception vector.
- **Grip:** the lineage-conditioned policy emits an adhesion output. A cell adjacent to a rod or foreign body can attach when that output exceeds 0.5. There is no body-ID merger or energy sharing with a foreign body.
- **Cost:** cells pay a synthetic altitude-dependent current-stress energy cost. A gripping cell receives a configurable discount and pays an attachment cost. Grip never creates energy; it reduces loss.
- **Circumnutation:** a phase carried by each cell evolves over time, with phase velocity influenced by a second genetic-policy output. When attached, the phase biases horizontal growth, slightly favors upward growth, and prefers destinations that remain near the rod. This is **programmed search behavior**; only its expression and ecological success are subject to selection.
- **Intervention:** press K in the viewer or pass `--sever-at N`. Rods disappear and the support benefit disappears wherever foreign neighbors do not provide alternate contact.
- **Persistence:** phase, grip and scaffold state are stored in checkpoints along with the original world's state.

This is an intentionally crude energy model, **not** actual gravity, fluid mechanics, tension, friction, a spring chain, or neural representation learning. Foreign-body contact grants only a local stress discount. It does not transmit structural forces along an extended organism. Old Emergence3D checkpoints are not generally compatible because perception/action dimensions changed.

## What to measure

The original metrics still include population, clades, body height, light capture, resource economy and shadowing. New fields are:

- `grip_fraction` and `gripped_cells`: which fraction of living cells actually chooses contact.
- `near_rod_fraction`: opportunity to grip a rod, independently of the policy decision.
- `high_cell_fraction`: fraction above 65% of the column height.
- `scaffold_cells` and `scaffolds_severed`: verify the environment intervention.

The question is **not whether the viewer shows spirals**. Success means that, over multiple seeds and held-out environmental settings, living organisms exploit support in a measurable way: better persistence, retained light capture, or more productive high-altitude occupancy **at a fair energy cost**, and that the effect falls in the grip-disabled or post-sever controls. Check whether direct sun-seeking and surface mats explain results without vines.

A beautiful winding path with no advantage is not evidence of adaptive vine evolution.

## Scientific boundaries and provenance

The underlying world has a fixed randomly initialized cellular network, while genomes and a fixed transformer-derived expression mechanism influence the network's behavior. Genomes evolve by mutation and differential survival, not by ordinary backpropagation. Features such as shared metabolism and the floor nutrient source are inherited from Emergence3D.

The support mechanism was designed as a controlled experiment. **No long-run selection advantage or biologically evolved climbing strategy is claimed here.** CPU integration tests check conservation-oriented inequalities, geometry, birth exclusion, save/load and support removal; real ecological comparisons require running `experiment.py` and inspecting the results, not merely interpreting screenshots.

The metaphor links to our fig/parent-network and ComputingBeach explorations through **dependency on external structure**, not through literal mathematical equivalence between neural weight edits, RoPE, crystals and climbing plants.
