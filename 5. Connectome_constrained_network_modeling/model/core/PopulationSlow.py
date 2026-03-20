import torch
import torch.nn as nn

class PopulationSlow(nn.Module):
    def __init__(
        self,
        population_indices,
        signs,
        mask,
        slow_populations,
        gamma_init=0.995,
        gamma_min=0.90,
        gamma_max=0.9995,
        modes_per_population=1,
        seed=None,
    ):
        super().__init__()
        self.population_indices = population_indices
        self.signs = signs
        self.mask = mask
        self.slow_populations = slow_populations
        self.gamma_min = float(gamma_min)
        self.gamma_max = float(gamma_max)
        self.modes_per_population = int(modes_per_population)

        N = mask.shape[0]
        P = len(slow_populations)
        R = self.modes_per_population
        K = P * R

        # Unconstrained parameters -> bounded gammas
        # Initialize eta so that gamma ~= gamma_init
        # Map gamma_init into [0,1] then invert sigmoid approximately.
        g0 = (float(gamma_init) - self.gamma_min) / (self.gamma_max - self.gamma_min + 1e-8)
        g0 = min(1 - 1e-4, max(1e-4, g0))
        eta0 = torch.log(torch.tensor(g0) / (1 - torch.tensor(g0)))
        self.eta = nn.Parameter(eta0 * torch.ones(K))

        # fixed basis vectors
        self.register_buffer("v_slow", torch.zeros(K, N))
        self.register_buffer("u_slow", torch.zeros(K, N))

        if seed is not None:
            g = torch.Generator().manual_seed(seed)
        else:
            g = None

        k = 0
        for p in slow_populations:
            idx = population_indices[p]
            idx_t = torch.tensor(idx, dtype=torch.long)

            for r in range(R):
                v = torch.zeros(N)
                u = torch.zeros(N)

                if R == 1:
                    # all-ones mode
                    v[idx_t] = 1.0
                    u[idx_t] = 1.0
                else:
                    # heterogeneous within-pop basis: random within ±1
                    pattern = torch.rand(len(idx), generator=g).float()
                    # pattern = torch.randint(0, 2, (len(idx),), generator=g).float() * 2 - 1
                    v[idx_t] = pattern
                    u[idx_t] = pattern

                # Dale sign on outgoing vector
                v = torch.abs(v)
                u = self.signs * torch.abs(u)

                v = v / (v.norm() + 1e-8)
                u = u / (u.norm() + 1e-8)

                self.v_slow[k] = v
                self.u_slow[k] = u
                k += 1

    def gammas(self):
        s = torch.sigmoid(self.eta)
        return self.gamma_min + (self.gamma_max - self.gamma_min) * s

    def forward(self, device):
        N = self.mask.shape[0]
        W_slow = torch.zeros(N, N, device=device)
        gammas = self.gammas().to(device)

        # sum of outer products
        for k in range(len(gammas)):
            W_slow = W_slow + gammas[k] * torch.outer(self.v_slow[k], self.u_slow[k])

        return W_slow
