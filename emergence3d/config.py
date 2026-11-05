"""All the laws of the 3D world in one place.

Everything that is a *law of the world* lives here. Everything that is a *choice of a
lineage* lives in its genome and is decided by evolution.
"""
from dataclasses import dataclass, asdict
import json


@dataclass
class Config:
    # ---- world: a box of water, D layers deep, toroidal sideways -------------------
    D: int = 40                    # layers (z = 0 is the floor, z = D-1 the surface)
    H: int = 144
    W: int = 144
    seed: int = 0
    device: str = "auto"           # "auto" -> cuda if available

    # ---- cell anatomy ---------------------------------------------------------------
    n_state: int = 8               # private internal channels per cell
    n_signal: int = 4              # diffusing chemical channels (shared medium)
    hidden: int = 48               # hidden units of the cell's rule network

    # ---- genomes ------------------------------------------------------------------
    vocab: int = 32                # token alphabet (0 = PAD)
    genome_max: int = 48
    genome_init_len: int = 16
    max_genomes: int = 0           # genome table slots; 0 = auto (scales with volume)
    mutation_rate: float = 0.03

    # ---- translator (the transformer that reads genomes) -----------------------------
    d_model: int = 96
    n_heads: int = 4
    n_layers: int = 3
    regulation: float = 0.6        # strength of the lineage-state token (0 = no feedback)
    express_every: int = 25

    # ---- light: falls from a moving sky, absorbed by water and by every cell it crosses
    light_base: float = 0.25       # sky brightness everywhere
    light_sun: float = 0.75        # extra under the moving suns
    n_suns: int = 5
    sun_radius: float = 0.14       # fraction of the world's width
    sun_speed: float = 0.0004      # fraction of the world's width per step
    water_clarity: float = 0.5     # light at the floor = exp(-1/clarity) of the sky, with no cells
    cell_shade: float = 0.5        # each living cell passes exp(-cell_shade) of the light

    # ---- nutrient: seeps from the floor, diffuses up, is taken up, returns at death ----
    nutrient_floor: float = 1.0    # the floor layer relaxes toward this
    floor_exchange: float = 0.2
    nutrient_diffusion: float = 0.6
    nutrient_max: float = 1.5

    # ---- a cell's economy ----------------------------------------------------------
    e_max: float = 2.0
    n_max: float = 1.0             # nutrient a cell can hold
    photo_yield: float = 0.12      # energy per unit of absorbed light per step
    n_per_e: float = 0.25          # nutrient used per unit of energy made
    uptake_max: float = 0.25       # max fraction of local nutrient taken per step
    e_divide: float = 0.6
    birth_cost: float = 0.05
    metabolism: float = 0.006
    emit_cost: float = 0.004
    attack_rate: float = 0.12
    attack_cost: float = 0.004
    attack_eff: float = 0.7
    share_rate: float = 0.20       # max fraction of energy AND nutrient sent to each body-mate
    decompose: float = 0.6         # fraction of a dead cell's energy returned (as nutrient)

    # ---- ageing ------------------------------------------------------------------
    hazard_base: float = 0.0004
    age_onset: int = 300
    age_span: int = 1500
    hazard_old: float = 0.02

    # ---- signals -------------------------------------------------------------------
    signal_decay: float = 0.92
    signal_diffusion: float = 0.5

    # ---- abiogenesis -----------------------------------------------------------------
    seed_cells: int = 4000
    min_population: int = 200
    reseed_cells: int = 1000

    def to_json(self):
        return json.dumps(asdict(self), indent=2)

    @staticmethod
    def from_dict(d):
        c = Config()
        for k, v in d.items():
            if hasattr(c, k):
                setattr(c, k, v)
        return c
