import numpy as np
import torch
from torch import nn, optim

# Manually add root path for imports to improve interoperability
import sys; sys.path.insert(0, "../..")

from model.core.PopulationSlow import PopulationSlow
from utils.services.ds_service import DSService
from utils.services.rnn_service import RNNService


class RNNFreePop(nn.Module):
    def __init__(
        self,
        nA, nB, nC, nD, nX=10,  # free population (unkown neurons) labelled as X
        input_dim=1,
        tau=0.1, dt=1.0,
        # E/I ratios per population
        E_frac_A=0.7, E_frac_B=0.1, E_frac_C=0.1, E_frac_D=1/3, E_frac_X=0.37,
        sparsity_U=1,
        # intra-hemisphere sparsity
        sparsity_AA=0.1125, sparsity_AB=0.18, sparsity_AC=0.475, sparsity_AD=0.065, sparsity_AX=0.07,
        sparsity_BA=0, sparsity_BB=0, sparsity_BC=0, sparsity_BD=0, sparsity_BX=0.1,
        sparsity_CA=0.08, sparsity_CB=0.45, sparsity_CC=0.09, sparsity_CD=0.04, sparsity_CX=0.13,
        sparsity_DA=0.02, sparsity_DB=0, sparsity_DC=0, sparsity_DD=0, sparsity_DX=0.02,
        sparsity_XA=0.07, sparsity_XB=0.1, sparsity_XC=0.1, sparsity_XD=0.02, sparsity_XX=0.074,  # Free population X is just one for both hemispheres
        # inter-hemispheric sparsities (keep-prob)
        sparsity_LA_RA=0, sparsity_LA_RB=0, sparsity_LA_RC=0, sparsity_LA_RD=0,
        sparsity_LB_RA=0.2, sparsity_LB_RB=0.3, sparsity_LB_RC=0.04, sparsity_LB_RD=0.06,
        sparsity_LC_RA=0.03, sparsity_LC_RB=0.15, sparsity_LC_RC=0, sparsity_LC_RD=0,
        sparsity_LD_RA=0.05, sparsity_LD_RB=0.05, sparsity_LD_RC=0.05, sparsity_LD_RD=0,
        lr=1e-3,
        weight_decay=1e-5,
        fast_spectral_radius_penalty_strength=1e-2,
        slow_antagonism_penalty_strength=5e-4,
        rho_target_fast=0.95,
        activation='softplus',
        seed=None,
        device=None,
        diag_zero_W=True,
        gcamp_tau_rise=0.25,
        gcamp_tau_decay=2.4,
        clamp_weights_min=None,
        clamp_weights_max=None,
        W_symmetric=False,
        stage1_frac=0.3,
        stage2_frac=0.3,
        verbose_every=None,        # if None -> default to 50 prints per run
        n_slow_pops=8
    ):
        super().__init__()

        # device
        if device is None:
            self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)
        self.to(self.device)

        self.loss = None
        self.loss_mse = None
        self.loss_reg = None
        self.clamp_weights_min = clamp_weights_min
        self.clamp_weights_max = clamp_weights_max

        self.nA = nA
        self.nB = nB
        self.nC = nC
        self.nD = nD
        self.nX = nX
        self.n_hemi = 2
        self.n_units_hemi = nA + nB + nC + nD
        self.n_units_anchor = self.n_hemi * self.n_units_hemi
        self.n_units = self.n_hemi * self.n_units_hemi + self.nX
        self.n_out = 8

        self.dt = dt
        self.alpha = dt / tau
        self.gcamp_tau_rise = gcamp_tau_rise
        self.gcamp_tau_decay = gcamp_tau_decay

        self.n_free = int(nX)
        self.stage1_frac = float(stage1_frac)
        self.stage2_frac = float(stage2_frac)
        self.verbose_every = verbose_every

        self.f = RNNService.activation_dict[activation]

        # -------- parameters --------
        if seed is not None:
            torch.manual_seed(seed)

        W_raw = torch.randn(self.n_units, self.n_units)
        # mild hemisphere asymmetry to help pick a basin
        W_raw[:self.n_units_hemi] *= 0.95
        W_raw[self.n_units_hemi:] *= 1.05

        self.W_raw = nn.Parameter(W_raw / np.sqrt(self.n_units))
        self.U_raw = nn.Parameter(torch.randn(self.n_units, input_dim) / np.sqrt(max(1, input_dim)))

        # -------- indices (populations) --------
        # left hemisphere
        idx_LA = torch.arange(0, nA)
        idx_LB = torch.arange(nA, nA + nB)
        idx_LC = torch.arange(nA + nB, nA + nB + nC)
        idx_LD = torch.arange(nA + nB + nC, nA + nB + nC + nD)
        idx_L = torch.arange(0, nA + nB + nC + nD)

        self.register_buffer("idx_LA", idx_LA)
        self.register_buffer("idx_LB", idx_LB)
        self.register_buffer("idx_LC", idx_LC)
        self.register_buffer("idx_LD", idx_LD)
        self.register_buffer("idx_L", idx_L)

        offset = nA + nB + nC + nD

        # right hemisphere
        idx_RA = idx_LA + offset
        idx_RB = idx_LB + offset
        idx_RC = idx_LC + offset
        idx_RD = idx_LD + offset
        idx_R = idx_L + offset

        self.register_buffer("idx_RA", idx_RA)
        self.register_buffer("idx_RB", idx_RB)
        self.register_buffer("idx_RC", idx_RC)
        self.register_buffer("idx_RD", idx_RD)
        self.register_buffer("idx_R", idx_R)

        idx_X = torch.arange(self.n_units_anchor, self.n_units)
        self.register_buffer("idx_X", idx_X)
        self.population_indices = [
            self.idx_LA.tolist(),
            self.idx_LB.tolist(),
            self.idx_LC.tolist(),
            self.idx_LD.tolist(),
            self.idx_RA.tolist(),
            self.idx_RB.tolist(),
            self.idx_RC.tolist(),
            self.idx_RD.tolist(),
            self.idx_X.tolist(),
        ]

        # -------- Dale signs --------
        signs = torch.ones(self.n_units)
        for hemi in [0, 1]:
            off = hemi * self.n_units_hemi
            signs[off + int(E_frac_A * nA): off + nA] *= -1
            signs[off + nA + int(E_frac_B * nB): off + nA + nB] *= -1
            signs[off + nA + nB + int(E_frac_C * nC): off + nA + nB + nC] *= -1
            signs[off + nA + nB + nC + int(E_frac_D * nD): off + nA + nB + nC + nD] *= -1
        signs[self.n_units_anchor + int(E_frac_X * nX) : self.n_units] = -1
        self.register_buffer("signs", signs)

        # -------- sparsity masks --------
        def block_mask(n_rows, n_cols, keep_prob, seed_=None):
            g = None if seed_ is None else torch.Generator().manual_seed(seed_)
            size = n_rows * n_cols
            r = torch.rand(size, generator=g)
            _, order = torch.sort(r)
            k = int(keep_prob * size)
            r[order[:k]] = 1.0
            if k < size:
                r[order[k:]] = 0.0
            return r.reshape(n_rows, n_cols).float()

        def define_mask_W():
            mask_W = torch.zeros((self.n_units, self.n_units), dtype=torch.float32)

            # L -> L blocks
            mask_W[np.ix_(idx_LA, idx_LA)] = block_mask(len(idx_LA), len(idx_LA), sparsity_AA, seed)
            mask_W[np.ix_(idx_LB, idx_LA)] = block_mask(len(idx_LB), len(idx_LA), sparsity_AB, seed)
            mask_W[np.ix_(idx_LC, idx_LA)] = block_mask(len(idx_LC), len(idx_LA), sparsity_AC, seed)
            mask_W[np.ix_(idx_LD, idx_LA)] = block_mask(len(idx_LD), len(idx_LA), sparsity_AD, seed)
            mask_W[np.ix_(idx_X , idx_LA)] = block_mask(len(idx_X) , len(idx_LA), sparsity_AX, seed)

            mask_W[np.ix_(idx_LA, idx_LB)] = block_mask(len(idx_LA), len(idx_LB), sparsity_BA, seed)
            mask_W[np.ix_(idx_LB, idx_LB)] = block_mask(len(idx_LB), len(idx_LB), sparsity_BB, seed)
            mask_W[np.ix_(idx_LC, idx_LB)] = block_mask(len(idx_LC), len(idx_LB), sparsity_BC, seed)
            mask_W[np.ix_(idx_LD, idx_LB)] = block_mask(len(idx_LD), len(idx_LB), sparsity_BD, seed)
            mask_W[np.ix_(idx_X , idx_LB)] = block_mask(len(idx_X) , len(idx_LB), sparsity_BX, seed)

            mask_W[np.ix_(idx_LA, idx_LC)] = block_mask(len(idx_LA), len(idx_LC), sparsity_CA, seed)
            mask_W[np.ix_(idx_LB, idx_LC)] = block_mask(len(idx_LB), len(idx_LC), sparsity_CB, seed)
            mask_W[np.ix_(idx_LC, idx_LC)] = block_mask(len(idx_LC), len(idx_LC), sparsity_CC, seed)
            mask_W[np.ix_(idx_LD, idx_LC)] = block_mask(len(idx_LD), len(idx_LC), sparsity_CD, seed)
            mask_W[np.ix_(idx_X , idx_LC)] = block_mask(len(idx_X) , len(idx_LC), sparsity_CX, seed)

            mask_W[np.ix_(idx_LA, idx_LD)] = block_mask(len(idx_LA), len(idx_LD), sparsity_DA, seed)
            mask_W[np.ix_(idx_LB, idx_LD)] = block_mask(len(idx_LB), len(idx_LD), sparsity_DB, seed)
            mask_W[np.ix_(idx_LC, idx_LD)] = block_mask(len(idx_LC), len(idx_LD), sparsity_DC, seed)
            mask_W[np.ix_(idx_LD, idx_LD)] = block_mask(len(idx_LD), len(idx_LD), sparsity_DD, seed)

            # R -> R blocks
            if W_symmetric:
                mask_W[np.ix_(idx_R, idx_R)] = mask_W[np.ix_(idx_L, idx_L)]
                mask_W[np.ix_(idx_X, idx_R)] = mask_W[np.ix_(idx_X, idx_L)]
            else:
                mask_W[np.ix_(idx_RA, idx_RA)] = block_mask(len(idx_RA), len(idx_RA), sparsity_AA, seed)
                mask_W[np.ix_(idx_RB, idx_RA)] = block_mask(len(idx_RB), len(idx_RA), sparsity_AB, seed)
                mask_W[np.ix_(idx_RC, idx_RA)] = block_mask(len(idx_RC), len(idx_RA), sparsity_AC, seed)
                mask_W[np.ix_(idx_RD, idx_RA)] = block_mask(len(idx_RD), len(idx_RA), sparsity_AD, seed)
                mask_W[np.ix_(idx_X , idx_RA)] = block_mask(len(idx_X) , len(idx_RA), sparsity_AX, seed)

                mask_W[np.ix_(idx_RA, idx_RB)] = block_mask(len(idx_RA), len(idx_RB), sparsity_BA, seed)
                mask_W[np.ix_(idx_RB, idx_RB)] = block_mask(len(idx_RB), len(idx_RB), sparsity_BB, seed)
                mask_W[np.ix_(idx_RC, idx_RB)] = block_mask(len(idx_RC), len(idx_RB), sparsity_BC, seed)
                mask_W[np.ix_(idx_RD, idx_RB)] = block_mask(len(idx_RD), len(idx_RB), sparsity_BD, seed)
                mask_W[np.ix_(idx_X , idx_RB)] = block_mask(len(idx_X) , len(idx_RB), sparsity_BX, seed)

                mask_W[np.ix_(idx_RA, idx_RC)] = block_mask(len(idx_RA), len(idx_RC), sparsity_CA, seed)
                mask_W[np.ix_(idx_RB, idx_RC)] = block_mask(len(idx_RB), len(idx_RC), sparsity_CB, seed)
                mask_W[np.ix_(idx_RC, idx_RC)] = block_mask(len(idx_RC), len(idx_RC), sparsity_CC, seed)
                mask_W[np.ix_(idx_RD, idx_RC)] = block_mask(len(idx_RD), len(idx_RC), sparsity_CD, seed)
                mask_W[np.ix_(idx_X , idx_RC)] = block_mask(len(idx_X) , len(idx_RC), sparsity_CX, seed)

                mask_W[np.ix_(idx_RA, idx_RD)] = block_mask(len(idx_RA), len(idx_RD), sparsity_DA, seed)
                mask_W[np.ix_(idx_RB, idx_RD)] = block_mask(len(idx_RB), len(idx_RD), sparsity_DB, seed)
                mask_W[np.ix_(idx_RC, idx_RD)] = block_mask(len(idx_RC), len(idx_RD), sparsity_DC, seed)
                mask_W[np.ix_(idx_RD, idx_RD)] = block_mask(len(idx_RD), len(idx_RD), sparsity_DD, seed)
                mask_W[np.ix_(idx_X , idx_RD)] = block_mask(len(idx_X) , len(idx_RD), sparsity_DX, seed)

            # L -> R blocks
            mask_W[np.ix_(idx_RA, idx_LA)] = block_mask(len(idx_RA), len(idx_LA), sparsity_LA_RA, seed)
            mask_W[np.ix_(idx_RB, idx_LA)] = block_mask(len(idx_RB), len(idx_LA), sparsity_LA_RB, seed)
            mask_W[np.ix_(idx_RC, idx_LA)] = block_mask(len(idx_RC), len(idx_LA), sparsity_LA_RC, seed)
            mask_W[np.ix_(idx_RD, idx_LA)] = block_mask(len(idx_RD), len(idx_LA), sparsity_LA_RD, seed)

            mask_W[np.ix_(idx_RA, idx_LB)] = block_mask(len(idx_RA), len(idx_LB), sparsity_LB_RA, seed)
            mask_W[np.ix_(idx_RB, idx_LB)] = block_mask(len(idx_RB), len(idx_LB), sparsity_LB_RB, seed)
            mask_W[np.ix_(idx_RC, idx_LB)] = block_mask(len(idx_RC), len(idx_LB), sparsity_LB_RC, seed)
            mask_W[np.ix_(idx_RD, idx_LB)] = block_mask(len(idx_RD), len(idx_LB), sparsity_LB_RD, seed)

            mask_W[np.ix_(idx_RA, idx_LC)] = block_mask(len(idx_RA), len(idx_LC), sparsity_LC_RA, seed)
            mask_W[np.ix_(idx_RB, idx_LC)] = block_mask(len(idx_RB), len(idx_LC), sparsity_LC_RB, seed)
            mask_W[np.ix_(idx_RC, idx_LC)] = block_mask(len(idx_RC), len(idx_LC), sparsity_LC_RC, seed)
            mask_W[np.ix_(idx_RD, idx_LC)] = block_mask(len(idx_RD), len(idx_LC), sparsity_LC_RD, seed)

            mask_W[np.ix_(idx_RA, idx_LD)] = block_mask(len(idx_RA), len(idx_LD), sparsity_LD_RA, seed)
            mask_W[np.ix_(idx_RB, idx_LD)] = block_mask(len(idx_RB), len(idx_LD), sparsity_LD_RB, seed)
            mask_W[np.ix_(idx_RC, idx_LD)] = block_mask(len(idx_RC), len(idx_LD), sparsity_LD_RC, seed)
            mask_W[np.ix_(idx_RD, idx_LD)] = block_mask(len(idx_RD), len(idx_LD), sparsity_LD_RD, seed)

            # R -> L blocks
            if W_symmetric:
                mask_W[np.ix_(idx_L, idx_R)] = mask_W[np.ix_(idx_R, idx_L)].T
            else:
                mask_W[np.ix_(idx_LA, idx_RA)] = block_mask(len(idx_LA), len(idx_RA), sparsity_LA_RA, seed)
                mask_W[np.ix_(idx_LB, idx_RA)] = block_mask(len(idx_LB), len(idx_RA), sparsity_LA_RB, seed)
                mask_W[np.ix_(idx_LC, idx_RA)] = block_mask(len(idx_LC), len(idx_RA), sparsity_LA_RC, seed)
                mask_W[np.ix_(idx_LD, idx_RA)] = block_mask(len(idx_LD), len(idx_RA), sparsity_LA_RD, seed)

                mask_W[np.ix_(idx_LA, idx_RB)] = block_mask(len(idx_LA), len(idx_RB), sparsity_LB_RA, seed)
                mask_W[np.ix_(idx_LB, idx_RB)] = block_mask(len(idx_LB), len(idx_RB), sparsity_LB_RB, seed)
                mask_W[np.ix_(idx_LC, idx_RB)] = block_mask(len(idx_LC), len(idx_RB), sparsity_LB_RC, seed)
                mask_W[np.ix_(idx_LD, idx_RB)] = block_mask(len(idx_LD), len(idx_RB), sparsity_LB_RD, seed)

                mask_W[np.ix_(idx_LA, idx_RC)] = block_mask(len(idx_LA), len(idx_RC), sparsity_LC_RA, seed)
                mask_W[np.ix_(idx_LB, idx_RC)] = block_mask(len(idx_LB), len(idx_RC), sparsity_LC_RB, seed)
                mask_W[np.ix_(idx_LC, idx_RC)] = block_mask(len(idx_LC), len(idx_RC), sparsity_LC_RC, seed)
                mask_W[np.ix_(idx_LD, idx_RC)] = block_mask(len(idx_LD), len(idx_RC), sparsity_LC_RD, seed)

                mask_W[np.ix_(idx_LA, idx_RD)] = block_mask(len(idx_LA), len(idx_RD), sparsity_LD_RA, seed)
                mask_W[np.ix_(idx_LB, idx_RD)] = block_mask(len(idx_LB), len(idx_RD), sparsity_LD_RB, seed)
                mask_W[np.ix_(idx_LC, idx_RD)] = block_mask(len(idx_LC), len(idx_RD), sparsity_LD_RC, seed)
                mask_W[np.ix_(idx_LD, idx_RD)] = block_mask(len(idx_LD), len(idx_RD), sparsity_LD_RD, seed)



            # mask to and from free population
            mask_W[np.ix_(idx_X, idx_X)] = block_mask(len(idx_X), len(idx_X), sparsity_XX, seed)

            mask_W[np.ix_(idx_X , idx_LA)] = block_mask(len(idx_X) , len(idx_LA), sparsity_AX, seed)
            mask_W[np.ix_(idx_X , idx_LB)] = block_mask(len(idx_X) , len(idx_LB), sparsity_BX, seed)
            mask_W[np.ix_(idx_X , idx_LC)] = block_mask(len(idx_X) , len(idx_LC), sparsity_CX, seed)
            mask_W[np.ix_(idx_X , idx_LD)] = block_mask(len(idx_X) , len(idx_LD), sparsity_DX, seed)
            if W_symmetric:
                mask_W[np.ix_(idx_X, idx_R)] = mask_W[np.ix_(idx_X, idx_L)]
            else:
                mask_W[np.ix_(idx_X, idx_RA)] = block_mask(len(idx_X), len(idx_RA), sparsity_AX, seed)
                mask_W[np.ix_(idx_X, idx_RB)] = block_mask(len(idx_X), len(idx_RB), sparsity_BX, seed)
                mask_W[np.ix_(idx_X, idx_RC)] = block_mask(len(idx_X), len(idx_RC), sparsity_CX, seed)
                mask_W[np.ix_(idx_X, idx_RD)] = block_mask(len(idx_X), len(idx_RD), sparsity_DX, seed)

            mask_W[np.ix_(idx_LA, idx_X)] = block_mask(len(idx_LA), len(idx_X), sparsity_XA, seed)
            mask_W[np.ix_(idx_LB, idx_X)] = block_mask(len(idx_LB), len(idx_X), sparsity_XB, seed)
            mask_W[np.ix_(idx_LC, idx_X)] = block_mask(len(idx_LC), len(idx_X), sparsity_XC, seed)
            mask_W[np.ix_(idx_LD, idx_X)] = block_mask(len(idx_LD), len(idx_X), sparsity_XD, seed)
            if W_symmetric:
                mask_W[np.ix_(idx_R, idx_X)] = mask_W[np.ix_(idx_L, idx_X)]
            else:
                mask_W[np.ix_(idx_RA, idx_X)] = block_mask(len(idx_RA), len(idx_X), sparsity_AX, seed)
                mask_W[np.ix_(idx_RB, idx_X)] = block_mask(len(idx_RB), len(idx_X), sparsity_BX, seed)
                mask_W[np.ix_(idx_RC, idx_X)] = block_mask(len(idx_RC), len(idx_X), sparsity_CX, seed)
                mask_W[np.ix_(idx_RD, idx_X)] = block_mask(len(idx_RD), len(idx_X), sparsity_DX, seed)

            if diag_zero_W:
                mask_W.fill_diagonal_(0.0)
            return mask_W

        mask_W = define_mask_W()
        self.register_buffer("mask_W", mask_W)
        self.register_buffer("mask_U", block_mask(self.n_units, input_dim, sparsity_U, seed))

        # Example: all populations slow except motion onset (index 2 and 6)
        slow_pops = np.arange(n_slow_pops)  # array with population indices where the slow mode is applied (if len is 8 exclude free neurons, if 9 include them)
        self.W_slow_module = PopulationSlow(
            population_indices=self.population_indices,
            signs=self.signs,
            mask=self.mask_W,
            slow_populations=slow_pops,
            modes_per_population=2,
            gamma_init=0.995
        )

        # target penalty strengths
        self.fast_spectral_radius_penalty_strength = fast_spectral_radius_penalty_strength
        self.slow_antagonism_penalty_strength = slow_antagonism_penalty_strength
        self.rho_target_fast = rho_target_fast

        # effective penalty strengths according to schedule
        self.effective_fast_spectral_radius_penalty_strength = fast_spectral_radius_penalty_strength
        self.effective_slow_antagonism_penalty_strength = slow_antagonism_penalty_strength

        # state buffers (set in forward)
        self.h = None
        self.xs = None
        self.ys = None

        self.optimizer = optim.Adam([self.W_raw, self.U_raw], lr=lr, weight_decay=weight_decay)

    # ---------- transforms ----------
    def W_fast(self):
        W_clamp = torch.clamp(torch.abs(self.W_raw), self.clamp_weights_min, self.clamp_weights_max)
        return W_clamp * self.mask_W * self.signs.view(1, -1)

    def W(self):
        return self.W_fast() + self.W_slow_module(self.device) * self.mask_W

    def U(self):
        return torch.abs(self.U_raw) * self.mask_U

    # ---------- penalties ----------
    def spectral_radius_power(self, W, n_iter=50, tol=1e-6):
        v = torch.randn(W.shape[0], device=W.device)
        v = v / (torch.norm(v) + 1e-8)
        prev = 0.0
        for _ in range(n_iter):
            v_new = W @ v
            rho = torch.norm(v_new)
            v = v_new / (rho + 1e-8)
            if torch.abs(rho - prev) < tol:
                break
            prev = rho
        return rho

    def fast_spectral_radius_penalty(self, margin=0.05):
        if self.effective_fast_spectral_radius_penalty_strength == 0:
            return 0

        rho_fast = self.spectral_radius_power(self.W_fast())
        penalty = torch.relu(rho_fast - self.rho_target_fast - margin).pow(2)
        return penalty * self.effective_fast_spectral_radius_penalty_strength

    def _stimulus_gated_slow_antagonism_penalty(self, h, stim_side, v_L, v_R):
        proj_L = torch.einsum("ntu,u->nt", h, v_L)
        proj_R = torch.einsum("ntu,u->nt", h, v_R)
        desired = torch.where(
            stim_side[:, None] == 1,
            proj_L - proj_R,
            proj_R - proj_L
        )
        return torch.mean(torch.relu(-desired))

    def stimulus_gated_slow_antagonism_penalty(self, x_pred, stim_side):
        if self.effective_slow_antagonism_penalty_strength == 0:
            return 0

        v_L = self.W_slow_module.v_slow[:4].sum(dim=0)
        v_R = self.W_slow_module.v_slow[4:].sum(dim=0)
        v_L = v_L / (v_L.norm() + 1e-8)
        v_R = v_R / (v_R.norm() + 1e-8)
        return self.effective_slow_antagonism_penalty_strength * self._stimulus_gated_slow_antagonism_penalty(h=x_pred, stim_side=stim_side, v_L=v_L, v_R=v_R)

    # ---------- forward ----------
    def forward(self, x0, inputs):
        """
        Anchor readout:
          - dynamics still run on all neurons
          - population outputs y_k are averages over anchors only
        """
        if inputs.ndim == 1:
            inputs = inputs[None, :, None]
        elif inputs.ndim == 2:
            inputs = inputs.unsqueeze(0)  # (1,T,input_dim)
        N, T, _ = inputs.shape
        device = inputs.device

        # initial condition
        if x0 is None:
            x0 = torch.zeros(self.n_units, device=device)
        elif not torch.is_tensor(x0):
            x0 = torch.tensor(x0, dtype=torch.float32)
        if x0.ndim == 0:
            x0 = x0.unsqueeze(0).repeat(N, self.n_units)
        elif x0.ndim == 1:
            x0 = x0.unsqueeze(0).repeat(N, 1)
        x0 = x0.to(device)

        self.h = x0
        self.xs = torch.zeros(N, T, self.n_units, device=device)
        self.ys = torch.zeros(N, T, self.n_out, device=device)

        U = self.U().to(device)          # (n_units, input_dim)
        W = self.W().to(device)          # (n_units, n_units)

        beta = torch.exp(torch.tensor(-self.alpha, device=device))

        for t in range(T):
            rec = self.f(self.h) @ W.T                 # (N, n_units)
            inp = inputs[:, t, :] * U.T                # (N, n_units)

            drive = rec + inp
            self.h = beta * self.h + (1.0 - beta) * drive

            # self.h = self.h + (-self.h + rec + inp) * self.alpha  # standard forward Euler
            self.xs[:, t, :] = self.f(self.h)

        # anchor-only population readout
        for k in range(self.n_out):
            self.ys[:, :, k] = self.xs[:, :, self.population_indices[k]].mean(dim=-1)

        xs_f = DSService.apply_gcamp_kernel(self.xs, self.gcamp_tau_rise, self.gcamp_tau_decay, self.dt)
        ys_f = DSService.apply_gcamp_kernel(self.ys, self.gcamp_tau_rise, self.gcamp_tau_decay, self.dt)

        # keep device
        if not torch.is_tensor(xs_f):
            xs_f = torch.tensor(xs_f, dtype=torch.float32)
        if not torch.is_tensor(ys_f):
            ys_f = torch.tensor(ys_f, dtype=torch.float32)

        self.xs = xs_f.to(device)
        self.ys = ys_f.to(device)

        return self.xs, self.ys

    def downsample_signal(self, raw_signal, time_sample_list):
        # raw_signal: (N,T,dim), time_sample_list: (T_ds,)
        if not torch.is_tensor(time_sample_list):
            time_sample_list = torch.tensor(time_sample_list, dtype=torch.float32, device=raw_signal.device)
        t_raw = torch.arange(0, raw_signal.shape[1], device=raw_signal.device, dtype=torch.float32) * self.dt
        dt = torch.abs(t_raw[:, None] - time_sample_list[None, :])  # (T, T_ds)
        idx_sample = torch.argmin(dt, dim=0)                         # (T_ds,)
        return raw_signal[:, idx_sample, :]

    def _infer_stim_side(self, train_list, device):
        # Prefer explicit stim_side if provided by dataset items.
        if hasattr(train_list[0], "stim_side") and train_list[0].stim_side is not None:
            return torch.tensor([int(t.stim_side) for t in train_list], device=device)

        # Otherwise, mimic your original heuristic if possible.
        sides = []
        for t in train_list:
            x = np.asarray(t.input_signal)
            if x.ndim == 2 and x.shape[1] >= self.n_units:
                left = np.sum(x[:, :self.n_units_hemi])
                right = np.sum(x[:, self.n_units_hemi:self.n_units_anchor])
                sides.append(1 if left > right else -1)
            else:
                # fallback if input doesn't encode hemi-wise channels
                sides.append(1)
        return torch.tensor(sides, device=device)

    def fit(self, train_list, x0=None, n_epochs=1000, verbose=True, downsample_target_list=None):
        """
        Staged regularizers:
          Stage 1: weak spectral penalty, no antagonism
          Stage 2: full spectral penalty, no antagonism
          Stage 3: full spectral + antagonism
        """
        device = next(self.parameters()).device
        N = len(train_list)

        stage1_epochs = int(self.stage1_frac * n_epochs)
        stage2_epochs = int(self.stage2_frac * n_epochs)
        stage3_start = stage1_epochs + stage2_epochs

        # stack inputs/outputs
        inputs = [torch.tensor(t.input_signal, dtype=torch.float32) for t in train_list]
        outputs = [torch.tensor(t.output_signal, dtype=torch.float32) for t in train_list]

        inputs = torch.stack(inputs).to(device)    # (N,T,input_dim)
        outputs = torch.stack(outputs).to(device)  # (N,T,8)

        stim_side = self._infer_stim_side(train_list, device=device)

        # initial condition
        if x0 is None:
            x0_list = []
            for t in train_list:
                if getattr(t, "initial_value", None) is None:
                    x0_list.append(torch.zeros(self.n_units, device=device))
                else:
                    x0_list.append(torch.tensor(t.initial_value, dtype=torch.float32, device=device))
            x0 = torch.stack(x0_list).to(device)
        elif not torch.is_tensor(x0):
            x0 = torch.tensor(x0, dtype=torch.float32, device=device)
        if x0.ndim == 0:
            x0 = x0.unsqueeze(0).repeat(N, self.n_units)
        elif x0.ndim == 1:
            x0 = x0.unsqueeze(0).repeat(N, 1)
        x0 = x0.to(device)

        if self.verbose_every is None:
            self.verbose_every = 50

        for epoch in range(n_epochs):
            # schedule effective strengths
            if epoch < stage1_epochs:
                self.effective_fast_spectral_radius_penalty_strength = self.fast_spectral_radius_penalty_strength * 0.1
                self.effective_slow_antagonism_penalty_strength = 0.0
                stage = 1
            elif epoch < stage3_start:
                self.effective_fast_spectral_radius_penalty_strength = self.fast_spectral_radius_penalty_strength
                self.effective_slow_antagonism_penalty_strength = 0.0
                stage = 2
            else:
                self.effective_fast_spectral_radius_penalty_strength = self.fast_spectral_radius_penalty_strength
                self.effective_slow_antagonism_penalty_strength = self.slow_antagonism_penalty_strength
                stage = 3

            self.optimizer.zero_grad()

            x_pred, y_pred = self.forward(x0, inputs)

            if downsample_target_list is not None:
                y_pred = self.downsample_signal(y_pred, downsample_target_list)

            mse = (y_pred - outputs).pow(2).mean()
            loss = mse
            loss = loss + self.fast_spectral_radius_penalty()
            loss = loss + self.stimulus_gated_slow_antagonism_penalty(x_pred, stim_side)

            self.loss_mse = mse.item()
            self.loss_reg = (loss - mse).item()
            self.loss = loss.item()

            if verbose and (epoch % self.verbose_every == 0 or epoch in [stage1_epochs, stage3_start]):
                with torch.no_grad():
                    rho_fast = self.spectral_radius_power(self.W_fast()).item()
                    yL_mean = y_pred[..., :4].mean().item()
                    yR_mean = y_pred[..., 4:].mean().item()
                print(f"[Stage {stage}] Epoch {epoch:4d} | Loss {loss.item():.6e} | "
                      f"MSE {mse.item():.6e} | Reg {self.loss_reg:.6e} | "
                      f"rho_fast {rho_fast:.3f} | <yL> {yL_mean:.3f} | <yR> {yR_mean:.3f}")

            loss.backward()
            torch.nn.utils.clip_grad_norm_([self.W_raw, self.U_raw], max_norm=1.0)
            self.optimizer.step()

        return self.W()
