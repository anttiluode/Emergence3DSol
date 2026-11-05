"""The 3D world: a box of water where every occupied voxel is one cell.

Axes are (z, y, x). z = 0 is the floor, z = D-1 the surface. The box wraps sideways
(x and y are toroidal) and is closed at the floor and the surface.

What is new compared with the flat Emergence is where the two things life needs come from:

  LIGHT falls from a moving sky (a few drifting "suns" over a dimmer sky). Water dims it
  with depth, and every living cell it passes through absorbs part of it. A cell sitting
  above another cell shades it. Nothing about this is a rule for organisms; it is just
  how light behaves in a column.

  NUTRIENT seeps out of the floor, diffuses upward, is taken up by cells, and returns to
  the water where a cell dies.

A cell makes energy only from light AND nutrient together (photosynthesis): energy made
per step = photo_yield x light absorbed, limited by the nutrient it holds (n_per_e of
nutrient per unit of energy). So the two resources come from opposite ends of the box.
Body-mates share both energy and nutrient. Whether anything takes advantage of that (a
"plant" that reaches from the dark nutrient-rich deep into the light, a canopy, a living
layer at the depth where the two meet) is not written anywhere. It is only possible.

Everything else is the flat Emergence, moved to six neighbours:
  - one shared cell network (fixed random weights) maps what a cell perceives to what it
    does, modulated per lineage by the phenotype that the fixed random transformer
    expresses from the genome tokens plus the lineage's current STATE;
  - actions: update private state, emit 4 signals, take up nutrient, share with
    body-mates, steal energy from strangers, divide (and in which of 6 directions), and
    whether the child stays attached to the parent's body;
  - energy is conserved except where it is spent; nutrient is conserved except at the
    floor, which exchanges with an unlimited reservoir.
"""
import math
import torch

from .genome import GenomePool
from .translator import Translator

# six face neighbours: (dz, dy, dx)
DIRS = [(0, 0, 1), (0, 0, -1), (0, 1, 0), (0, -1, 0), (1, 0, 0), (-1, 0, 0)]
OPP = [1, 0, 3, 2, 5, 4]
UP, DOWN = 4, 5


def nb(x, d):
    """Value at the neighbour in direction d, for every site (wraps in all axes; callers
    mask the floor/surface wrap with World.valid)."""
    dz, dy, dx = DIRS[d]
    return torch.roll(x, shifts=(-dz, -dy, -dx), dims=(-3, -2, -1))


class World:
    def __init__(self, cfg):
        self.cfg = cfg
        dev = cfg.device
        if dev == "auto":
            dev = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(dev)
        self.gen = torch.Generator(device=self.device).manual_seed(cfg.seed)
        D, H, W, C, Q = cfg.D, cfg.H, cfg.W, cfg.n_state, cfg.n_signal
        if cfg.max_genomes <= 0:
            cfg.max_genomes = max(32768, (D * H * W) // 32)

        self.n_actions = C + Q + 1 + 1 + 1 + 6 + 1 + 1
        self.n_percept = 2 * C + 4 * Q + 13
        self.state_dim = C + 5

        # ---- shared cell network (fixed, seeded) -------------------------------------
        g = torch.Generator().manual_seed(cfg.seed + 99)
        self.W1 = (torch.randn(cfg.hidden, self.n_percept, generator=g) * 1.5
                   / math.sqrt(self.n_percept)).to(self.device)
        self.W2 = (torch.randn(self.n_actions, cfg.hidden, generator=g) * 1.5
                   / math.sqrt(cfg.hidden)).to(self.device)

        # ---- genomes and their translator --------------------------------------------
        self.pool = GenomePool(cfg, self.device, self.gen)
        self.tr = Translator(cfg, self.n_actions, self.state_dim, cfg.seed).to(self.device)
        self.tr.calibrate(2048, self.device, self.gen)
        G = cfg.max_genomes
        self.gamma = torch.zeros(G, cfg.hidden, device=self.device)
        self.beta = torch.zeros(G, cfg.hidden, device=self.device)
        self.obias = torch.zeros(G, self.n_actions, device=self.device)
        self.color = torch.zeros(G, 3, device=self.device)
        self.lin_state = torch.zeros(G, self.state_dim, device=self.device)

        # ---- geometry ----------------------------------------------------------------
        z = torch.arange(D, device=self.device)[:, None, None].expand(D, H, W)
        self.zz = z.float()
        self.valid = [torch.ones(D, H, W, dtype=torch.bool, device=self.device) for _ in DIRS]
        self.valid[UP] = z < D - 1
        self.valid[DOWN] = z > 0
        self.n_valid = sum(v.float() for v in self.valid)
        self.depth_att = torch.exp(-(D - 1 - self.zz) / (cfg.water_clarity * D))
        yy, xx = torch.meshgrid(torch.arange(H, device=self.device, dtype=torch.float32),
                                torch.arange(W, device=self.device, dtype=torch.float32),
                                indexing="ij")
        self.yy, self.xx = yy, xx

        # ---- fields --------------------------------------------------------------------
        zf = lambda *s: torch.zeros(*s, device=self.device)
        self.alive = torch.zeros(D, H, W, dtype=torch.bool, device=self.device)
        self.gid = torch.full((D, H, W), -1, dtype=torch.long, device=self.device)
        self.body = torch.full((D, H, W), -1, dtype=torch.long, device=self.device)
        self.next_body = 0
        self.energy = zf(D, H, W)
        self.nut = zf(D, H, W)            # nutrient held by cells
        self.S = zf(C, D, H, W)
        self.age = zf(D, H, W)
        self.M = zf(D, H, W) + cfg.nutrient_floor * 0.5     # nutrient in the water
        self.sig = zf(Q, D, H, W)
        self.light = zf(D, H, W)
        self.photo = zf(D, H, W)          # energy made last step (for views / metrics)
        self.sun_pos = torch.rand(cfg.n_suns, 2, generator=self.gen, device=self.device) \
            * torch.tensor([H, W], device=self.device, dtype=torch.float32)
        ang = torch.rand(cfg.n_suns, generator=self.gen, device=self.device) * 6.283
        self.sun_vel = torch.stack([torch.sin(ang), torch.cos(ang)], 1)
        self.sky = zf(H, W)

        self.t = 0
        self.births = 0
        self.deaths = 0
        self.reseeds = 0
        self._update_sun()
        self._update_light()
        self.seed_life(cfg.seed_cells)

    # ------------------------------------------------------------------ utils
    def rand(self, *shape):
        return torch.rand(*shape, generator=self.gen, device=self.device)

    def nbv(self, x, d, fill=0):
        """Neighbour value with the floor/surface treated as a wall (fill)."""
        return torch.where(self.valid[d], nb(x, d), torch.full_like(x, fill))

    def diffuse(self, f, k):
        """f += k * (mean of the existing face neighbours - f); no flux through floor/surface."""
        s = sum(torch.where(self.valid[d], nb(f, d), f) for d in range(6))
        return f + k * (s / 6.0 - f)

    def grads(self, f):
        gx = 0.5 * (nb(f, 0) - nb(f, 1))
        gy = 0.5 * (nb(f, 2) - nb(f, 3))
        gz = 0.5 * (self.nbv(f, UP) - self.nbv(f, DOWN)) * self.valid[UP] * self.valid[DOWN]
        return gx, gy, gz

    def _write_phenotype(self, rows, state):
        if rows.numel() == 0:
            return
        g, b, o, c = self.tr.express(self.pool.tokens[rows], self.pool.length[rows], state)
        self.gamma[rows], self.beta[rows], self.obias[rows], self.color[rows] = g, b, o, c
        self.lin_state[rows] = state

    def seed_life(self, n):
        """Abiogenesis: n random cells with random genomes, anywhere in the water."""
        cfg = self.cfg
        empty = torch.nonzero(~self.alive.flatten()).flatten()
        if empty.numel() == 0:
            return
        pick = empty[torch.randperm(empty.numel(), generator=self.gen, device=self.device)[:n]]
        rows = self.pool.new_random(pick.numel(), self.t)
        pick = pick[:rows.numel()]
        self._write_phenotype(rows, torch.zeros(rows.numel(), self.state_dim, device=self.device))
        self.alive.view(-1)[pick] = True
        self.gid.view(-1)[pick] = rows
        self.body.view(-1)[pick] = torch.arange(self.next_body, self.next_body + pick.numel(),
                                                device=self.device)
        self.next_body += pick.numel()
        self.energy.view(-1)[pick] = 0.8
        self.nut.view(-1)[pick] = 0.3
        self.age.view(-1)[pick] = 0
        S = self.S.view(cfg.n_state, -1)
        S[:, pick] = 0.1 * torch.randn(cfg.n_state, pick.numel(), generator=self.gen, device=self.device)
        self.pool.recount(self.gid[self.alive])

    def _update_sun(self):
        cfg = self.cfg
        H, W = cfg.H, cfg.W
        jitter = (torch.rand(cfg.n_suns, 2, generator=self.gen, device=self.device) - 0.5) * 0.15
        self.sun_vel = self.sun_vel + jitter
        self.sun_vel = self.sun_vel / self.sun_vel.norm(dim=1, keepdim=True)
        size = torch.tensor([H, W], device=self.device, dtype=torch.float32)
        self.sun_pos = (self.sun_pos + self.sun_vel * cfg.sun_speed * size) % size
        r = cfg.sun_radius * min(H, W)
        dy = (self.yy[None] - self.sun_pos[:, 0, None, None]).abs()
        dx = (self.xx[None] - self.sun_pos[:, 1, None, None]).abs()
        dy = torch.minimum(dy, H - dy)
        dx = torch.minimum(dx, W - dx)
        sun = torch.exp(-(dy ** 2 + dx ** 2) / (2 * r * r)).sum(0).clamp(max=1.0)
        self.sky = cfg.light_base + cfg.light_sun * sun

    def _update_light(self):
        """Light at every voxel: sky x water attenuation x shading by the cells above."""
        aF = self.alive.float()
        above = torch.flip(torch.cumsum(torch.flip(aF, [0]), 0), [0]) - aF
        self.light = self.sky[None] * self.depth_att * torch.exp(-self.cfg.cell_shade * above)

    # ------------------------------------------------------------------ step
    @torch.no_grad()
    def step(self):
        cfg = self.cfg
        D, H, W, C, Q = cfg.D, cfg.H, cfg.W, cfg.n_state, cfg.n_signal
        alive = self.alive
        aF = alive.float()
        gid = self.gid
        E, N = self.energy, self.nut

        # ---------------- light (shading by the current cells) ------------------------
        self._update_light()
        L = self.light

        # ---------------- perception ------------------------------------------------------
        nb_alive = torch.stack([self.nbv(alive, d, False) for d in range(6)])
        nb_gid = torch.stack([nb(gid, d) for d in range(6)])
        body = self.body
        nb_body = torch.stack([nb(body, d) for d in range(6)])
        kin = nb_alive & (nb_body == body[None]) & alive[None]
        alien = nb_alive & (nb_body != body[None]) & alive[None]
        twin = alien & (nb_gid == gid[None])
        n_occ = nb_alive.float().sum(0)
        S_nb = sum(self.nbv(self.S, d) for d in range(6)) / n_occ.clamp(min=1)
        mgx, mgy, mgz = self.grads(self.M)
        sgx, sgy, sgz = self.grads(self.sig)
        P = torch.cat([
            self.S,
            (E / cfg.e_max)[None], (N / cfg.n_max)[None],
            torch.tanh(self.age / cfg.age_onset)[None],
            self.M[None], mgx[None] * 10, mgy[None] * 10, mgz[None] * 10,
            L[None],
            self.sig, sgx * 5, sgy * 5, sgz * 5,
            S_nb,
            kin.float().sum(0)[None] / 6, alien.float().sum(0)[None] / 6,
            twin.float().sum(0)[None] / 6,
            (self.n_valid - n_occ)[None] / 6,
            torch.ones_like(E)[None],
        ])

        # ---------------- the cell network (only for living cells) ------------------------
        idx = torch.nonzero(alive.flatten()).flatten()
        A = self.n_actions
        out = torch.zeros(A, D * H * W, device=self.device)
        if idx.numel():
            g = gid.flatten()[idx]
            p = P.view(P.shape[0], -1)[:, idx].T
            h = p @ self.W1.T
            h = torch.tanh(h * (1 + self.gamma[g]) + self.beta[g])
            o = h @ self.W2.T + self.obias[g]
            out[:, idx] = o.T
        out = out.view(A, D, H, W)
        k = 0
        dS = out[k:k + C]; k += C
        emit = torch.sigmoid(out[k:k + Q]) * aF; k += Q
        take = torch.sigmoid(out[k]) * aF; k += 1
        share = torch.sigmoid(out[k]) * aF; k += 1
        divide = torch.sigmoid(out[k]) * aF; k += 1
        dirlog = out[k:k + 6]; k += 6
        attack = torch.sigmoid(out[k]) * aF; k += 1
        detach = torch.sigmoid(out[k]) * aF; k += 1

        # ---------------- internal state ------------------------------------------------
        self.S = (0.7 * self.S + 0.3 * torch.tanh(dS)) * aF

        # ---------------- nutrient uptake -------------------------------------------------
        up = torch.minimum(take * cfg.uptake_max * self.M, (cfg.n_max - N).clamp(min=0))
        self.M = self.M - up
        N = N + up

        # ---------------- photosynthesis: light AND nutrient --------------------------------
        absorbed = L * (1 - math.exp(-cfg.cell_shade)) * aF
        made = torch.minimum(cfg.photo_yield * absorbed, N / cfg.n_per_e)
        made = torch.minimum(made, (cfg.e_max - E).clamp(min=0))
        E = E + made
        N = N - made * cfg.n_per_e
        self.photo = made

        # ---------------- sharing with body-mates (exact, energy and nutrient) -----------------
        kf = kin.float()
        out_e = (share * cfg.share_rate * E)[None] * kf / 6 * 4      # same per-neighbour rate as 2D
        out_n = (share * cfg.share_rate * N)[None] * kf / 6 * 4
        E = E - out_e.sum(0) + sum(nb(out_e[OPP[d]], d) for d in range(6))
        N = N - out_n.sum(0) + sum(nb(out_n[OPP[d]], d) for d in range(6))

        # ---------------- stealing energy from strangers ----------------------------------------
        nbE = torch.stack([nb(E, d) for d in range(6)])
        req = (attack * cfg.attack_rate)[None] * nbE * alien.float()
        req_tot = sum(nb(req[OPP[d]], d) for d in range(6))
        scale = (E / (req_tot + 1e-9)).clamp(max=1.0)
        took = req * torch.stack([nb(scale, d) for d in range(6)])
        lost = sum(nb(took[OPP[d]], d) for d in range(6))
        E = E - lost + cfg.attack_eff * took.sum(0)

        # ---------------- costs -------------------------------------------------------------
        E = E - aF * (cfg.metabolism + cfg.emit_cost * emit.sum(0) + cfg.attack_cost * attack)
        E = E.clamp(max=cfg.e_max) * aF
        N = N.clamp(0, cfg.n_max) * aF

        # ---------------- death: energy and nutrient return to the water as nutrient -------------
        ramp = ((self.age - cfg.age_onset) / cfg.age_span).clamp(0, 1)
        hazard = cfg.hazard_base + (cfg.hazard_old - cfg.hazard_base) * ramp
        die = alive & ((E <= 0) | (self.rand(D, H, W) < hazard))
        back = die.float() * (E.clamp(min=0) * cfg.decompose * cfg.n_per_e + N)
        self.M = (self.M + back).clamp(max=cfg.nutrient_max)
        alive = alive & ~die
        aF = alive.float()
        E = E * aF
        N = N * aF
        self.S = self.S * aF
        gid = torch.where(alive, gid, torch.full_like(gid, -1))
        body = torch.where(alive, body, torch.full_like(body, -1))
        self.deaths += die.sum()

        # ---------------- division into one of six neighbours ----------------------------------
        want = alive & (E > cfg.e_divide) & (self.rand(D, H, W) < divide)
        gum = -torch.log(-torch.log(self.rand(6, D, H, W).clamp(1e-9, 1 - 1e-9)))
        d_choice = (dirlog + gum).argmax(0)
        wd = [want & (d_choice == d) & self.valid[d] for d in range(6)]
        empty = ~alive
        pr = self.rand(D, H, W)
        claims = torch.stack([self.valid[d] & nb(wd[OPP[d]], d) & empty for d in range(6)])
        prio = torch.stack([nb(pr, d) for d in range(6)])
        prio = torch.where(claims, prio, torch.full_like(prio, -1.0))
        win = prio.argmax(0)
        newborn = claims.any(0)
        won = torch.zeros_like(alive)
        for d in range(6):
            won |= wd[d] & nb(newborn, d) & (nb(win, d) == OPP[d])
        sel = win[None]
        pick = lambda x: torch.gather(torch.stack([nb(x, d) for d in range(6)]), 0, sel)[0]
        pE, pN, pG, pB, pD = pick(E), pick(N), pick(gid), pick(body), pick(detach)
        leaves = newborn & (self.rand(D, H, W) < pD)
        fresh = self.next_body + torch.arange(D * H * W, device=self.device).view(D, H, W)
        pB = torch.where(leaves, fresh, pB)
        self.next_body += D * H * W
        pS = torch.stack([nb(self.S, d) for d in range(6)])
        pS = torch.gather(pS, 0, sel[:, None].expand(1, C, D, H, W))[0]
        childE = (0.5 * pE - cfg.birth_cost) * newborn.float()
        E = torch.where(won, 0.5 * E, E)
        N = torch.where(won, 0.5 * N, N)
        E = torch.where(newborn, childE, E)
        N = torch.where(newborn, 0.5 * pN, N)
        gid = torch.where(newborn, pG, gid)
        body = torch.where(newborn, pB, body)
        self.S = torch.where(newborn[None], pS, self.S)
        self.age = torch.where(newborn, torch.zeros_like(self.age), self.age)
        alive = alive | newborn
        self.births += newborn.sum()

        # ---------------- mutation at birth --------------------------------------------------
        mut = newborn & (self.rand(D, H, W) < cfg.mutation_rate)
        midx = torch.nonzero(mut.flatten()).flatten()
        if midx.numel():
            parents = gid.flatten()[midx]
            rows = self.pool.mutate_from(parents, self.t)
            n = rows.numel()
            if n:
                self._write_phenotype(rows, self.lin_state[parents[:n]])
                gid.view(-1)[midx[:n]] = rows

        # ---------------- environment ----------------------------------------------------------
        if self.t % 4 == 0:
            self._update_sun()
        self.M[0] = self.M[0] + cfg.floor_exchange * (cfg.nutrient_floor - self.M[0])
        self.M = self.diffuse(self.M, cfg.nutrient_diffusion).clamp(0, cfg.nutrient_max)
        self.sig = cfg.signal_decay * self.diffuse(self.sig, cfg.signal_diffusion) + 0.5 * emit
        self.sig = self.sig.clamp(0, 3)

        # ---------------- commit ------------------------------------------------------------------
        self.alive, self.gid, self.energy, self.nut, self.body = alive, gid, E, N, body
        self.age = (self.age + 1) * alive.float()
        self.pool.recount(gid[alive])
        self.t += 1

        if self.t % cfg.express_every == 0:
            self.regulate()
        if self.t % 10 == 0 and int(alive.sum()) < cfg.min_population:
            self.reseeds += 1
            self.seed_life(cfg.reseed_cells)

    # ------------------------------------------------------------- regulation
    @torch.no_grad()
    def regulate(self):
        """Feed each lineage's lived state back into the translator and re-express."""
        cfg = self.cfg
        G = cfg.max_genomes
        a = self.alive.flatten()
        if not a.any():
            return
        g = self.gid.flatten()[a]
        nbk = torch.stack([self.nbv(self.alive, d, False) & (nb(self.body, d) == self.body)
                           for d in range(6)])
        feats = torch.cat([
            self.S.view(cfg.n_state, -1)[:, a],
            (self.energy.flatten()[a] / cfg.e_max)[None],
            (self.nut.flatten()[a] / cfg.n_max)[None],
            self.light.flatten()[a][None],
            (nbk.float().sum(0).flatten()[a] / 6)[None],
        ]).T
        sums = torch.zeros(G, feats.shape[1], device=self.device).index_add_(0, g, feats)
        cnt = self.pool.count.float()
        rows = torch.nonzero(cnt > 0).flatten()
        mean = sums[rows] / cnt[rows, None]
        size = torch.log1p(cnt[rows])[:, None] / 6
        self._write_phenotype(rows, torch.cat([mean, size], 1))

    # ------------------------------------------------------------- interaction
    def _ball(self, z, y, x, r):
        dz = self.zz - z
        dy = (self.yy[None] - y).abs(); dx = (self.xx[None] - x).abs()
        dy = torch.minimum(dy, self.cfg.H - dy); dx = torch.minimum(dx, self.cfg.W - dx)
        return dz * dz + dy * dy + dx * dx <= r * r

    @torch.no_grad()
    def wipe(self, z, y, x, r):
        """Kill every cell in a ball (damage / repair experiments)."""
        m = self._ball(z, y, x, r) & self.alive
        self.alive &= ~m
        self.energy *= self.alive.float()
        self.nut *= self.alive.float()
        self.S *= self.alive.float()
        self.gid = torch.where(self.alive, self.gid, torch.full_like(self.gid, -1))
        self.body = torch.where(self.alive, self.body, torch.full_like(self.body, -1))
        self.pool.recount(self.gid[self.alive])

    @torch.no_grad()
    def feed(self, z, y, x, r, amount=1.0):
        """Pour nutrient into a ball of water."""
        self.M = torch.where(self._ball(z, y, x, r), torch.full_like(self.M, amount), self.M)

    # ------------------------------------------------------------- measurement
    @torch.no_grad()
    def metrics(self):
        cfg = self.cfg
        a = self.alive
        pop = int(a.sum())
        cnt = self.pool.count
        live = cnt > 0
        m = {"t": self.t, "population": pop, "genomes": int(live.sum()),
             "births": int(self.births), "deaths": int(self.deaths),
             "mutations": self.pool.total_mutations, "mutations_blocked": self.pool.blocked,
             "genome_table_used": float(live.sum()) / cfg.max_genomes,
             "reseeds": self.reseeds,
             "nutrient_water": float(self.M.mean()),
             "light_reaching_floor": float((self.light[0].sum() / (self.sky.sum() + 1e-9))),
             "light_captured": float(1 - self.light[0].sum() / (self.sky.sum() * float(self.depth_att[0, 0, 0]) + 1e-9))}
        if pop:
            c = cnt[live].float()
            m["clades"] = int(torch.unique(self.pool.clade[live]).numel())
            m["mean_genome_len"] = float((self.pool.length[live].float() * c).sum() / c.sum())
            m["mean_depth"] = float((self.pool.depth[live].float() * c).sum() / c.sum())
            m["max_depth"] = int(self.pool.depth[live].max())
            m["top_genome_share"] = float(c.max() / c.sum())
            pnorm = c / c.sum()
            m["genome_entropy"] = float(-(pnorm * pnorm.log()).sum())
            occ = torch.stack([self.nbv(a, d, False) for d in range(6)])
            kin = torch.stack([self.nbv(a, d, False) & (nb(self.body, d) == self.body) for d in range(6)])
            k = (kin.float().sum(0))[a]
            o = (occ.float().sum(0))[a]
            m["kin_neighbours"] = float(k.mean())
            m["stranger_neighbours"] = float((o - k).mean())
            bid, inv, bsz = torch.unique(self.body[a], return_inverse=True, return_counts=True)
            bsz = bsz.float()
            m["bodies"] = int(bsz.numel())
            m["body_size_cellweighted"] = float((bsz * bsz).sum() / bsz.sum())
            m["max_body"] = int(bsz.max())
            m["multicell_fraction"] = float(bsz[bsz >= 8].sum() / bsz.sum())
            m["mean_energy"] = float(self.energy[a].mean())
            m["mean_nutrient_cell"] = float(self.nut[a].mean())
            m["photo_per_cell"] = float(self.photo[a].mean())
            # where life lives in depth, and how tall bodies are
            zc = self.zz[a]
            m["mean_height"] = float(zc.mean() / max(1, cfg.D - 1))
            per_layer = a.float().sum((1, 2))
            m["living_layer"] = float(per_layer.argmax()) / max(1, cfg.D - 1)
            zmax = torch.full((bid.numel(),), -1.0, device=self.device).scatter_reduce(0, inv, zc, "amax")
            zmin = torch.full((bid.numel(),), 1e9, device=self.device).scatter_reduce(0, inv, zc, "amin")
            span = (zmax - zmin + 1)[inv]                       # each cell: its body's height
            m["body_height_cellweighted"] = float(span.mean())
            m["tall_fraction"] = float((span >= max(4, cfg.D // 4)).float().mean())
        return m

    # ------------------------------------------------------------- persistence
    def state_dict(self):
        keys = ["alive", "gid", "body", "energy", "nut", "S", "age", "M", "sig", "sun_pos",
                "sun_vel", "photo", "gamma", "beta", "obias", "color", "lin_state"]
        d = {k: getattr(self, k).cpu() for k in keys}
        d["pool"] = {k: (v.cpu() if torch.is_tensor(v) else v)
                     for k, v in self.pool.state_dict().items()}
        d["next_body"] = self.next_body
        d["t"], d["births"], d["deaths"], d["reseeds"] = self.t, int(self.births), int(self.deaths), self.reseeds
        d["config"] = self.cfg.__dict__.copy()
        d["gen_state"] = self.gen.get_state()
        return d

    def load_state_dict(self, d):
        for k in ["alive", "gid", "body", "energy", "nut", "S", "age", "M", "sig", "sun_pos", "sun_vel"]:
            setattr(self, k, d[k].to(self.device))
        if "photo" in d:
            self.photo = d["photo"].to(self.device)
        for k in ["gamma", "beta", "obias", "color", "lin_state"]:
            src = d[k].to(self.device)
            getattr(self, k)[:src.shape[0]] = src
        self.pool.load_state_dict({k: (v.to(self.device) if torch.is_tensor(v) else v)
                                   for k, v in d["pool"].items()})
        self.next_body = d.get("next_body", 0)
        self.t, self.births, self.deaths, self.reseeds = d["t"], d["births"], d["deaths"], d["reseeds"]
        try:
            self.gen.set_state(d["gen_state"])
        except Exception:
            pass
        self._update_sun()
        self._update_light()
