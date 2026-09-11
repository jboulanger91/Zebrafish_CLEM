"""
PopulationSlow
=============================

Adds a low-rank contribution to each population's OWN (diagonal-block)
connectivity, so that the timescale of the mode associated with that population
is set by one interpretable number, `gamma_p`.

Two structural facts this cannot address:

* a population with an **empty** diagonal block gets nothing, whatever `gamma`
  is. `sMI <- sMI` is 0.0 in both the old and the new connectome;
* a block whose surviving synapses form a **directed acyclic** subgraph is
  nilpotent -- `rho = 0` exactly -- so no weight assignment creates a
  self-sustaining mode. Measured fraction of samples with no directed cycle:
  0.00 for original iMI (q=0.1125), 0.35 for augmented iMI (q=0.0230), 0.66 
  for augmented cMI (q=0.0123).

Stochastic mode vectors for ANY modes_per_population (including modes_per_population=1):
   - Rather than forcing an identical flat vector `torch.ones(...)` when
     `modes_per_population=1`, mode vectors `u` and `v` are sampled randomly
     from `torch.rand` (or using a dedicated Generator if `seed` is provided).
   - This ensures distinct initial conditions across different random seeds
     or runs, preventing initialization collapse across multiple model instances.

A note on the interpretation around gamma paramgers:

Despite naively the hard floor `gamma_min` is supposedly around 0.90 in 
`gammas() = gamma_min + (gamma_max - gamma_min) * sigmoid(eta)`, this
would impose every population with a cyclic diagonal block to carry a self-mode 
of gain >= 0.90, i.e. to be at least a 1s integrator, with no way for the fit to 
say otherwise. 
Thus, the floor `gamma_min` is set to 0.0, so `gamma_p` is free to be small. 
And `gamma_init` was 0.995: with the block normalised (below), 0.995 means the 
slow blocks alone contribute a gain of 0.995, which leaves no room for `W_fast`, 
the init rescale had to crush `W_fast` to a gain of 0.003 to keep the total stable.
It is now 0.50, and since `eta` is in the optimizer the fit can raise it.
`gamma_p` maps to a timescale through the same relation as the rest of the
network: a mode with gain `g` relaxes with `tau_eff = tau / (1 - g)`. With
`tau = 0.1 s`, `gamma_p = 0.95` is 2 s and `gamma_p = 0.995` is 20 s -- but note
`gamma_p` adds to whatever gain `W_fast` already puts on that block, so the
realised timescale is set by the sum, not by `gamma_p` alone.
"""

import numpy as np
import torch
import torch.nn as nn


class PopulationSlow(nn.Module):
    def __init__(
        self,
        population_indices,
        support,                 # unsigned 0/1 support, derived from the signed mask_W
        slow_populations,
        signs=None,              # (N,) Dale sign per neuron; +1 E, -1 I
        gamma_init=0.50,         # Final desired effective value is 0.995, but this allows for W_fast contribution (see note)
        gamma_min=0.0,           # Final desired effective value is 0.900, but this allows for W_fast contribution (see note)
        gamma_max=0.9995,
        modes_per_population=1,
        randomize_modes=True,    # if True, randomise mode vectors even for modes_per_population=1
        seed=None,
        verbose=False,
    ):
        super().__init__()
        self.population_indices = population_indices
        self.slow_populations = list(slow_populations)
        self.gamma_min = float(gamma_min)
        self.gamma_max = float(gamma_max)
        self.modes_per_population = int(modes_per_population)
        self.randomize_modes = bool(randomize_modes)

        support = torch.as_tensor(np.asarray(support), dtype=torch.float32)
        support = (support != 0).to(torch.float32)
        N = support.shape[0]
        self.n_units = N

        if signs is None:
            signs = torch.ones(N)
        signs = torch.as_tensor(np.asarray(signs), dtype=torch.float32).reshape(-1)
        # FIX (was `self.signs = ...`): a plain attribute is not moved by
        # .to(device) and does not appear in state_dict.
        self.register_buffer("signs", signs)
        self.register_buffer("support", support, persistent=False)

        P = len(self.slow_populations)
        # one gamma per POPULATION, not per mode. The normalisation
        # below is of the population's whole masked block, so a per-mode gamma
        # would no longer have a well-defined gain. Still, `modes_per_population`
        # controls how heterogeneous the mode's shape is within the population.
        g0 = (float(gamma_init) - self.gamma_min) / (self.gamma_max - self.gamma_min + 1e-8)
        g0 = min(1 - 1e-4, max(1e-4, g0))
        eta0 = float(np.log(g0 / (1.0 - g0)))
        self.eta = nn.Parameter(eta0 * torch.ones(P))

        g = torch.Generator().manual_seed(seed) if seed is not None else None

        # One normalised, masked, signed block per population, precomputed once.
        blocks = torch.zeros(P, N, N)
        self.pop_rho = []  # spectral radius of the raw masked block
        self.pop_status = []  # "ok" | "acyclic" | "empty" | "ok (loop with pop X)"
        self.partner_pop = []  # None or int index of partnered population

        # Symmetrized connectome to identify contralateral homologue or reciprocal pairs
        n_all_pops = len(population_indices)

        for k, p in enumerate(self.slow_populations):
            idx_p = torch.as_tensor(population_indices[p], dtype=torch.long)
            B = self._build_intra_block(idx_p, support, N, g)
            rho = self._spectral_radius(B[idx_p][:, idx_p])

            n_syn = int(support[idx_p][:, idx_p].sum().item())
            status = "ok"
            partner = None

            if n_syn == 0:
                status = "empty"
            elif rho < 1e-8:
                status = "acyclic"

            # Fallback to multi-population reciprocal loop if intra-population is degenerate
            if status in ("empty", "acyclic"):
                B_loop, rho_loop, partner_found = self._find_reciprocal_partner_loop(
                    p, idx_p, support, n_all_pops, N, g
                )
                if partner_found is not None and rho_loop >= 1e-8:
                    B = B_loop
                    rho = rho_loop
                    status = f"ok (loop with pop {partner_found})"
                    partner = partner_found

            self.pop_rho.append(float(rho))
            self.pop_status.append(status)
            self.partner_pop.append(partner)

            if "ok" in status and rho >= 1e-8:
                # Normalise so that rho(block) == 1, ensuring rho(gamma_p * block) == gamma_p
                B = B / rho
                # Apply Dale's sign column-wise (post-by-pre orientation: presynaptic column Dale sign)
                B = B * self.signs[None, :]
            else:
                B = torch.zeros_like(B)

            blocks[k] = B

        self.register_buffer("blocks", blocks)

        if verbose:
            self._report()

    def _sample_pattern(self, length, g=None):
        """Samples positive mode weights across neurons in a population."""
        if length == 0:
            return torch.empty(0)
        if not self.randomize_modes and self.modes_per_population == 1:
            return torch.ones(length)
        # Sample heterogeneous strictly positive participation weights
        return 0.1 + 0.9 * torch.rand(length, generator=g).float()

    def _build_intra_block(self, idx, support, N, g):
        """Constructs low-rank outer products within a single population."""
        B = torch.zeros(N, N)
        if len(idx) == 0:
            return B
        for _ in range(self.modes_per_population):
            v = torch.zeros(N)
            u = torch.zeros(N)
            pat_v = self._sample_pattern(len(idx), g)
            pat_u = self._sample_pattern(len(idx), g)
            v[idx] = pat_v
            u[idx] = pat_u
            v = torch.abs(v)
            v = v / (v.norm() + 1e-8)
            u = torch.abs(u)
            u = u / (u.norm() + 1e-8)
            B = B + torch.outer(v, u)
        # Apply sparse connectome mask
        return B * support

    def _find_reciprocal_partner_loop(self, p, idx_p, support, n_all_pops, N, g):
        """
        Searches other populations for surviving reciprocal loops (p <-> q),
        prioritizing contralateral homologues (e.g. p ^ 4 in standard 8-pop layout)
        or populations with the densest bidirectional connectivity.
        """
        # Prioritize contralateral counterpart (if 8 populations: 0<->4, 1<->5, 2<->6, 3<->7)
        candidates = []
        if n_all_pops == 8:
            homologue = (p + 4) % 8
            candidates.append(homologue)
        for other in range(n_all_pops):
            if other != p and other not in candidates:
                candidates.append(other)

        for q in candidates:
            idx_q = torch.as_tensor(self.population_indices[q], dtype=torch.long)
            if len(idx_p) == 0 or len(idx_q) == 0:
                continue

            # Check if reciprocal synapses exist: p -> q and q -> p
            syn_pq = support[idx_q][:, idx_p].sum().item()  # from p into q
            syn_qp = support[idx_p][:, idx_q].sum().item()  # from q into p

            if syn_pq > 0 and syn_qp > 0:
                # Build low-rank loop block
                B = torch.zeros(N, N)
                vp = torch.zeros(N)
                up = torch.zeros(N)
                vq = torch.zeros(N)
                uq = torch.zeros(N)

                pat_vp = self._sample_pattern(len(idx_p), g)
                pat_up = self._sample_pattern(len(idx_p), g)
                pat_vq = self._sample_pattern(len(idx_q), g)
                pat_uq = self._sample_pattern(len(idx_q), g)

                vp[idx_p] = pat_vp / (pat_vp.norm() + 1e-8)
                up[idx_p] = pat_up / (pat_up.norm() + 1e-8)
                vq[idx_q] = pat_vq / (pat_vq.norm() + 1e-8)
                uq[idx_q] = pat_uq / (pat_uq.norm() + 1e-8)

                # Off-diagonal bipartite coupling: p -> q and q -> p
                B = B + torch.outer(vq, up)  # drive from p into q
                B = B + torch.outer(vp, uq)  # drive from q into p

                B = B * support
                idx_both = torch.cat([idx_p, idx_q])
                rho = self._spectral_radius(B[idx_both][:, idx_both])
                if rho >= 1e-8:
                    return B, rho, q

        return torch.zeros(N, N), 0.0, None

    @staticmethod
    def _spectral_radius(A):
        if A.numel() == 0:
            return 0.0
        lam = torch.linalg.eigvals(A.to(torch.float32))
        return float(lam.abs().max().item())

    def _report(self):
        print("[PopulationSlow] one gamma per population, block normalised to "
              "rho = 1 so gamma is the mode gain")
        for k, p in enumerate(self.slow_populations):
            idx = np.asarray(self.population_indices[p])
            n_syn = int(self.support[torch.as_tensor(idx)][:, torch.as_tensor(idx)]
                        .sum().item())
            q = n_syn / max(1, len(idx) ** 2)
            sign = float(self.signs[idx[0]].item()) if len(idx) else 0.0
            status = self.pop_status[k]
            note = ""
            if status == "empty":
                note = "  <-- EMPTY: no intra- or reciprocal multi-pop loop found"
            elif status == "acyclic":
                note = "  <-- ACYCLIC: no cycle found at this density"
            elif "loop with pop" in status:
                note = f"  <-- RESCUED: using multi-pop reciprocal loop ({status})"
            elif sign < 0:
                note = ("  <-- INHIBITORY population: gamma gives a NEGATIVE eigenvalue, "
                        "which shortens the timescale. Consider excluding it.")
            print(f"  pop {p}: n={len(idx):3d}  intra-pop synapses={n_syn:4d} "
                  f"(q={q:.4f})  raw rho={self.pop_rho[k]:.4f}  "
                  f"sign={'E' if sign > 0 else 'I'}{note}")

    def gammas(self):
        s = torch.sigmoid(self.eta)
        return self.gamma_min + (self.gamma_max - self.gamma_min) * s

    def forward(self, device=None):
        """
        `sum_p gamma_p * normalised_signed_block_p`.
        """
        gam = self.gammas()
        W_slow = torch.einsum("p,pij->ij", gam, self.blocks)
        return W_slow if device is None else W_slow.to(device)

    def population_gains(self):
        """`gamma_p` as a dict, plus the timescale it implies for a given tau."""
        return {int(p): float(g) for p, g in zip(self.slow_populations,
                                                 self.gammas().detach())}
