"""Tests for support mechanics (CPU; no graphics or online model downloads)."""
import math

import pytest
import torch

from emergence3d.config import Config
from emergence3d.support import contact_shell, drag_loss, make_scaffold


def test_rod_occupancy_and_contact_geometry():
    cfg = Config(D=10, H=14, W=14, scaffold_columns=4, scaffold_height=0.8,
                 device="cpu")
    solid = make_scaffold(cfg, torch.device("cpu"))
    near = contact_shell(solid)
    assert solid.dtype == torch.bool
    assert solid.shape == (10, 14, 14)
    assert int(solid.sum()) == 4 * 8
    assert not (near & solid).any()
    assert near.any()
    assert not torch.equal(near, solid)
    # Disabling the geometry gives no phantom contacts.
    cfg.scaffold_columns = 0
    assert not contact_shell(make_scaffold(cfg, torch.device("cpu"))).any()


def test_attached_stress_is_lower_but_not_free():
    cfg = Config(current_drag=0.02, current_power=1.0,
                 grip_relief=0.85, grip_cost=0.001)
    z = torch.tensor([1.0, 1.0, 0.0])
    attached = torch.tensor([True, False, True])
    loss = drag_loss(z, attached, cfg)
    assert float(loss[0]) < float(loss[1])
    assert math.isclose(float(loss[0]), 0.004, rel_tol=1e-5)
    assert math.isclose(float(loss[1]), 0.02, rel_tol=1e-5)
    assert math.isclose(float(loss[2]), cfg.grip_cost, rel_tol=1e-5)


@pytest.fixture(scope="module")
def small_world():
    from emergence3d.world import World
    cfg = Config(D=8, H=12, W=12, seed=4, device="cpu", hidden=16,
                 n_suns=1, seed_cells=32, max_genomes=512,
                 scaffold_columns=4, regulation=0.0,
                 express_every=10000)
    torch.set_num_threads(2)
    return World(cfg)


def test_vine_world_smoke_and_checkpoint(small_world):
    from emergence3d.world import World
    w = small_world
    assert not (w.alive & w.scaffold).any()
    for _ in range(3):
        w.step()
        assert not (w.alive & w.scaffold).any()
        assert torch.isfinite(w.energy).all()
        assert torch.isfinite(w.phase).all()
        assert w.grip.dtype == torch.bool
    m = w.metrics()
    assert "grip_fraction" in m and "near_rod_fraction" in m
    assert 0 <= m["grip_fraction"] <= 1
    state = w.state_dict()
    restored = World(w.cfg)
    restored.load_state_dict(state)
    assert torch.equal(restored.scaffold, w.scaffold)
    assert torch.equal(restored.phase, w.phase)
    assert torch.equal(restored.grip, w.grip)


def test_sever_is_an_intervention(small_world):
    w = small_world
    before = w.light.clone()
    w.sever_scaffolds()
    assert not w.scaffold.any()
    assert not w.support_near.any()
    assert w.support_severed
    assert torch.all(w.light >= before - 1e-7)
    w.step()
    assert w.metrics()["scaffolds_severed"] == 1
    # The absence of rods must survive a save/load cycle.
    from emergence3d.world import World
    restored = World(w.cfg)
    restored.load_state_dict(w.state_dict())
    assert restored.support_severed
    assert not restored.scaffold.any()
