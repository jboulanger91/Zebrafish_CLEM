"""
RNNConnectome
=============

A rate-based recurrent network whose connectivity is *anatomically constrained*:
the set of allowed synapses, and the excitatory/inhibitory sign of each one, are
imposed by a measured connectome and never learned. Only the *magnitudes* of the
anatomically existing synapses (and the input gains) are fitted.

What the model is
-----------------
Each of `n_units` neurons carries a scalar pre-activation `h_i`, integrated with
an exponential-Euler step (exact for a drive held constant over `dt`):

    h(t+dt) = beta * h(t) + (1 - beta) * ( W @ f(h(t)) + U * u(t) ),
    beta    = exp(-dt / tau)

`f` is the activation (softplus by default), so the observable firing rate is
`x = f(h)`. The matrix is stored in **post-by-pre** orientation, i.e. `W[i, j]`
is the weight *from* j *onto* i, which is what `f(h) @ W.T` computes.

Constraints baked into `W()`
----------------------------
* `mask_W` comes from the connectome and carries both topology (zeros where no
  synapse exists) and Dale's law (the sign of every existing synapse). The
  fitted tensor only ever contributes a non-negative magnitude, so no gradient
  step can flip a synapse's sign or invent a synapse that anatomy forbids.
* `clamp_weights_min` keeps anatomically real synapses from being driven to
  zero, so the fitted network cannot silently prune the connectome.
* `mask_U` restricts which neurons the stimulus can reach.
* An optional `W_fixed` pins individual entries to prescribed values (entries
  left as NaN stay free).

Timescales
----------
`W()` is the sum of two parts. `W_fast()` is the per-synapse fitted magnitude.
`W_slow_module` (a `PopulationSlow`) adds a low-rank, population-structured
component with near-unit gain, which supplies the seconds-long timescales the
calcium data show but that a fast recurrent matrix cannot produce on its own.

Readout
-------
Dynamics run on every neuron; the 8 outputs are plain means over the 8 recorded
populations (left/right x iMI/cMI/MON/sMI). Both the population means and, on
request, the full population activity are convolved with a GCaMP kernel so the
model output is comparable to measured dF/F rather than to firing rate.

Training
--------
`fit` is full-batch BPTT over the whole trial, with a three-stage regulariser
curriculum: (1) weak spectral-radius penalty, (2) full spectral-radius penalty,
(3) additionally a stimulus-gated antagonism penalty that asks the slow modes of
the two hemispheres to separate in the direction the stimulus dictates.

Performance notes (see also `fit`'s docstring)
----------------------------------------------
The cost of one epoch is dominated by the `T`-step Python BPTT loop and, unless
avoided, by GCaMP-filtering all `n_units` channels. This implementation:
  * filters only the 8 readout channels during training, and obtains the
    antagonism penalty by projecting the *unfiltered* activity onto the two
    slow-mode directions and filtering those 2 channels instead. Time-domain
    convolution and a linear map over units commute, so this is the same number
    computed a factor ~n_units/2 more cheaply;
  * accumulates `xs` in a list and stacks once, instead of writing 8000 slices
    in place into a preallocated tensor (one autograd node instead of ~T);
  * hoists the input drive and the transpose of `W` out of the loop;
  * warm-starts the power iteration for the spectral radius under `no_grad` and
    differentiates only the final `||W v||` (envelope theorem), instead of
    backpropagating through 50 power steps;
  * can pack the fitted weights into a dense vector of only the anatomically
    allowed entries, which shrinks the Adam state and step from O(n^2) to
    O(nnz) (`pack_parameters=True`);
  * can trade compute for activation memory with gradient checkpointing over
    time chunks (`checkpoint_chunk`), which is what to reach for if the process
    starts swapping;
  * supports absolute stage boundaries, LR decay on plateau and early stopping,
    so run length is set by the loss curve rather than by a guessed epoch count.

All of the above are numerically equivalent to, or opt-in extensions of, the
original behaviour: with default arguments this class trains the same model.
"""

import copy

import numpy as np
import torch
from torch import nn, optim
from torch.utils.checkpoint import checkpoint

from model.core.PopulationSlow import PopulationSlow
from utils.services.ds_service import DSService
from utils.services.rnn_service import RNNService
from utils.config import ConfigurationRNN


class RNNConnectome(nn.Module):
    def __init__(
            self,
            dict_neurons,
            W_fixed=None,
            input_dim=1,
            tau=0.1, dt=0.01,
            lr=1e-3,
            weight_decay=1e-5,
            fast_spectral_radius_penalty_strength=1e-2,
            slow_antagonism_penalty_strength=5e-4,
            rho_target_fast=0.95,
            activation='softplus',
            seed=None,
            device=None,
            gcamp_tau_rise=0.25,
            gcamp_tau_decay=2.4,
            clamp_weights_min=None,
            clamp_weights_max=None,
            stage1_frac=0.3,
            stage2_frac=0.3,
            verbose_every=None,  # if None -> default to 50 prints per run
            n_slow_pops=8,
            # ---- performance / training-control options
            pack_parameters=False,   # fit only the anatomically allowed entries
            clamp_soft=True,        # magnitude floor with a live gradient
            power_iters=10,          # power steps per epoch, warm-started
            precompute_input_drive=True,   # hoist inputs*U out of the time loop
            checkpoint_chunk=None,   # e.g. 500 -> gradient checkpointing
    ):
        super().__init__()

        self.dict_neurons = dict_neurons

        # ---- device ---------------------------------------------------------
        # Kept as an attribute because W() historically passes it to
        # PopulationSlow; note nn.Module.to() is what actually moves tensors.
        if device is None:
            self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)
        self.to(self.device)

        # ---- loss bookkeeping (filled in by fit) ----------------------------
        self.loss = None
        self.loss_mse = None
        self.loss_reg = None
        self.history = {"loss": [], "mse": [], "reg": [], "lr": []}

        # ---- weight-magnitude constraints -----------------------------------
        self.clamp_weights_min = clamp_weights_min
        self.clamp_weights_max = clamp_weights_max
        self.clamp_soft = bool(clamp_soft)

        # ---- sizes ----------------------------------------------------------
        # idx_side_change is the first index belonging to the right hemisphere,
        # so it doubles as the size of the left hemisphere.
        self.n_units_hemi = dict_neurons["idx_side_change"]
        self.n_units = dict_neurons["W"].shape[0]
        self.n_out = 8  # left/right x (iMI, cMI, MON, sMI)

        # ---- integration constants ------------------------------------------
        self.dt = dt
        self.alpha = dt / tau  # dimensionless step; beta = exp(-alpha)
        self.gcamp_tau_rise = gcamp_tau_rise
        self.gcamp_tau_decay = gcamp_tau_decay

        # ---- curriculum / logging -------------------------------------------
        self.stage1_frac = float(stage1_frac)
        self.stage2_frac = float(stage2_frac)
        self.verbose_every = verbose_every

        # nn.Softplus() is an nn.Module, so this registers as a submodule (and
        # is therefore *not* picked up by RNNService.extract_custom_attrs --
        # meaning `activation` is not stored in the checkpoint; reconstruct the
        # model with the same `activation` you trained it with).
        self.f = RNNService.activation_dict[activation]

        # ---- performance switches -------------------------------------------
        self.pack_parameters = bool(pack_parameters)
        self.power_iters = int(power_iters)
        self.precompute_input_drive = bool(precompute_input_drive)
        self.checkpoint_chunk = checkpoint_chunk

        # =====================================================================
        # Anatomical masks
        # =====================================================================
        # mask_W is expected post-by-pre and to carry the E/I sign of each
        # synapse: entries are 0 (no synapse), +1 (excitatory) or -1
        # (inhibitory). Because W_fast() multiplies a non-negative magnitude by
        # this mask, topology and Dale's law are both structurally enforced.
        if "W_mask" in dict_neurons.keys():
            mask_W = torch.as_tensor(np.asarray(dict_neurons["W_mask"]), dtype=torch.float32)
        else:
            mask_W = torch.sign(torch.as_tensor(np.asarray(dict_neurons["W"]), dtype=torch.float32))

        if "U_mask" in dict_neurons.keys():
            mask_U = torch.as_tensor(np.asarray(dict_neurons["U_mask"]), dtype=torch.float32)
        else:
            mask_U = torch.sign(torch.as_tensor(np.asarray(dict_neurons["U"]), dtype=torch.float32))
        if mask_U.dim() == 1:
            mask_U = mask_U.unsqueeze(1)  # (n_units,) -> (n_units, 1)

        self.register_buffer("mask_W", mask_W)
        self.register_buffer("mask_U", mask_U)

        # Binary support of the mask, precomputed once. Used for packing and for
        # reporting; keeping it as a buffer avoids recomputing != 0 per epoch.
        self.register_buffer("mask_W_support", (mask_W != 0).to(torch.float32), persistent=False)
        self.n_synapses = int(self.mask_W_support.sum().item())

        # =====================================================================
        # Fitted parameters
        # =====================================================================
        if seed is not None:
            torch.manual_seed(seed)

        W_raw = torch.randn(self.n_units, self.n_units)
        # Mild left/right asymmetry, so the two hemispheres are not exactly
        # interchangeable at init and the optimiser can commit to a basin.
        W_raw[:self.n_units_hemi] *= 0.95
        W_raw[self.n_units_hemi:] *= 1.05
        W_raw = W_raw / np.sqrt(self.n_units)

        if self.pack_parameters:
            # Only the anatomically allowed entries are fitted. The dense matrix
            # is rebuilt in _W_magnitude() by scattering this vector back into
            # its flat positions. This does not make the forward pass cheaper
            # (the matmul still needs the dense matrix) but it cuts the Adam
            # state and the optimiser step from O(n^2) to O(nnz), and stops
            # weight decay from acting on entries that do not exist.
            idx_W_nz = torch.nonzero(mask_W.reshape(-1), as_tuple=False).squeeze(-1)
            self.register_buffer("idx_W_nz", idx_W_nz, persistent=False)
            self.W_vals = nn.Parameter(W_raw.reshape(-1)[idx_W_nz].clone())
            self.W_raw = None  # not a Parameter in this mode
        else:
            self.register_buffer("idx_W_nz", torch.empty(0, dtype=torch.long), persistent=False)
            self.W_raw = nn.Parameter(W_raw)

        self.U_raw = nn.Parameter(torch.randn(self.n_units, input_dim) / np.sqrt(max(1, input_dim)))

        # =====================================================================
        # Optionally pinned entries of W
        # =====================================================================
        # Convention: W_fixed holds the prescribed value where an entry is to be
        # held fixed and NaN where it is to stay free. Registered as buffers so
        # they follow .to(device) and land in state_dict.
        if W_fixed is None:
            self.has_W_fixed = False
            self.register_buffer("W_fixed", torch.zeros(self.n_units, self.n_units))
            self.register_buffer("W_fixed_mask", torch.zeros(self.n_units, self.n_units))
        else:
            W_fixed = torch.as_tensor(np.asarray(W_fixed), dtype=torch.float32)
            assert W_fixed.shape == (self.n_units, self.n_units), \
                "Shape of W_fixed does not match W. Wrong number of neurons"
            self.has_W_fixed = True
            # 1.0 where pinned, 0.0 where free. nan_to_num keeps NaNs from
            # poisoning the product (NaN * 0 is NaN, not 0).
            self.register_buffer("W_fixed_mask", (~torch.isnan(W_fixed)).to(torch.float32))
            self.register_buffer("W_fixed", torch.nan_to_num(W_fixed, nan=0.0))

        # =====================================================================
        # Population indices
        # =====================================================================
        # Taken verbatim from the connectome dictionary: these need not be
        # contiguous, and everything downstream (readout, PopulationSlow) uses
        # them by index rather than assuming a block layout.
        idx_LiMI = torch.as_tensor(dict_neurons["neurons"][ConfigurationRNN.SIDE_LEFT]["iMI"]["idx_list"], dtype=torch.long)
        idx_LcMI = torch.as_tensor(dict_neurons["neurons"][ConfigurationRNN.SIDE_LEFT]["cMI"]["idx_list"], dtype=torch.long)
        idx_LMON = torch.as_tensor(dict_neurons["neurons"][ConfigurationRNN.SIDE_LEFT]["MON"]["idx_list"], dtype=torch.long)
        idx_LsMI = torch.as_tensor(dict_neurons["neurons"][ConfigurationRNN.SIDE_LEFT]["sMI"]["idx_list"], dtype=torch.long)
        idx_L = torch.as_tensor(dict_neurons["neurons"][ConfigurationRNN.SIDE_LEFT]["idx_list"], dtype=torch.long)

        self.register_buffer("idx_LiMI", idx_LiMI)
        self.register_buffer("idx_LcMI", idx_LcMI)
        self.register_buffer("idx_LMON", idx_LMON)
        self.register_buffer("idx_LsMI", idx_LsMI)
        self.register_buffer("idx_L", idx_L)

        idx_RiMI = torch.as_tensor(dict_neurons["neurons"][ConfigurationRNN.SIDE_RIGHT]["iMI"]["idx_list"], dtype=torch.long)
        idx_RcMI = torch.as_tensor(dict_neurons["neurons"][ConfigurationRNN.SIDE_RIGHT]["cMI"]["idx_list"], dtype=torch.long)
        idx_RMON = torch.as_tensor(dict_neurons["neurons"][ConfigurationRNN.SIDE_RIGHT]["MON"]["idx_list"], dtype=torch.long)
        idx_RsMI = torch.as_tensor(dict_neurons["neurons"][ConfigurationRNN.SIDE_RIGHT]["sMI"]["idx_list"], dtype=torch.long)
        idx_R = torch.as_tensor(dict_neurons["neurons"][ConfigurationRNN.SIDE_RIGHT]["idx_list"], dtype=torch.long)

        self.register_buffer("idx_RiMI", idx_RiMI)
        self.register_buffer("idx_RcMI", idx_RcMI)
        self.register_buffer("idx_RMON", idx_RMON)
        self.register_buffer("idx_RsMI", idx_RsMI)
        self.register_buffer("idx_R", idx_R)

        # Readout order must match the column order of the target signals.
        self.population_indices = [
            self.idx_LiMI.tolist(),
            self.idx_LcMI.tolist(),
            self.idx_LMON.tolist(),
            self.idx_LsMI.tolist(),
            self.idx_RiMI.tolist(),
            self.idx_RcMI.tolist(),
            self.idx_RMON.tolist(),
            self.idx_RsMI.tolist(),
        ]
        # Same information as a padded index tensor plus a 0/1 weight matrix, so
        # the 8 population means become one batched matmul instead of 8 advanced
        # indexing operations that each copy a (N, T, n_pop) block.
        readout = torch.zeros(self.n_units, self.n_out)
        for k, idx in enumerate(self.population_indices):
            if len(idx) == 0:
                continue
            readout[torch.as_tensor(idx, dtype=torch.long), k] = 1.0 / len(idx)
        self.register_buffer("readout_W", readout, persistent=False)  # (n_units, n_out)

        # =====================================================================
        # Slow population modes
        # =====================================================================
        # slow_pops selects which of the 8 populations get a slow component;
        # np.arange(8) means all of them.
        slow_pops = np.arange(n_slow_pops)
        self.W_slow_module = PopulationSlow(
            population_indices=self.population_indices,
            mask=self.mask_W,
            slow_populations=slow_pops,
            modes_per_population=2,
            gamma_init=0.995
        )

        # =====================================================================
        # Penalty strengths
        # =====================================================================
        self.fast_spectral_radius_penalty_strength = fast_spectral_radius_penalty_strength
        self.slow_antagonism_penalty_strength = slow_antagonism_penalty_strength
        self.rho_target_fast = rho_target_fast

        # "effective" values are what the loss actually uses; fit() rewrites
        # them at every epoch according to the stage schedule.
        self.effective_fast_spectral_radius_penalty_strength = fast_spectral_radius_penalty_strength
        self.effective_slow_antagonism_penalty_strength = slow_antagonism_penalty_strength

        # Warm-start vector for the power iteration. Persisting it across epochs
        # is what lets `power_iters` be ~10 instead of ~50: consecutive epochs
        # change W only slightly, so the previous dominant eigenvector is an
        # excellent starting guess.
        v0 = torch.randn(self.n_units)
        self.register_buffer("v_power", v0 / v0.norm(), persistent=False)

        # =====================================================================
        # Rolling state (set by forward)
        # =====================================================================
        self.h = None
        self.xs = None
        self.ys = None
        self.xs_is_filtered = None  # tells callers what self.xs currently holds

        self.optimizer = optim.Adam(self.trainable_parameters(), lr=lr, weight_decay=weight_decay)

    # ==================================================================
    # helpers
    # ==================================================================
    def trainable_parameters(self):
        """The fitted tensors, whichever parameterisation is in use."""
        W_param = self.W_vals if self.pack_parameters else self.W_raw
        return [W_param, self.U_raw]

    def clear_state(self):
        """
        Drop the cached trajectories.

        Worth calling before saving: RNNService.extract_custom_attrs walks
        __dict__ and serialises whatever it finds, so leaving self.xs in place
        writes an (N, T, n_units) tensor into every checkpoint.
        """
        self.h = None
        self.xs = None
        self.ys = None
        self.xs_is_filtered = None

    # ==================================================================
    # transforms
    # ==================================================================
    def _W_magnitude(self):
        """
        Non-negative magnitude of every synapse, before signs are applied.

        In packed mode the fitted vector is scattered back into a dense matrix;
        index_put is out-of-place and differentiable, so the gradient reaches
        only the anatomically allowed entries.
        """
        if self.pack_parameters:
            flat = torch.zeros(self.n_units * self.n_units,
                               device=self.W_vals.device, dtype=self.W_vals.dtype)
            flat = flat.index_put((self.idx_W_nz,), self.W_vals)
            W_raw = flat.view(self.n_units, self.n_units)
        else:
            W_raw = self.W_raw

        mag = torch.abs(W_raw)

        if self.clamp_soft and self.clamp_weights_min:
            # Same guarantee as the hard clamp (an allowed synapse never reaches
            # zero) but the gradient stays alive below the floor, so weights
            # that dip under it can still be fitted afterwards.
            mag = self.clamp_weights_min + mag
            if self.clamp_weights_max is not None:
                mag = torch.clamp(mag, max=self.clamp_weights_max)
        else:
            mag = torch.clamp(mag, self.clamp_weights_min, self.clamp_weights_max)

        return mag

    def W_fast(self):
        """Fast recurrent matrix: fitted magnitude x anatomical sign/topology."""
        return self._W_magnitude() * self.mask_W

    def W(self):
        """
        Effective recurrent matrix, post-by-pre.

        Fast part plus the low-rank slow part, then any pinned entries are
        substituted in. The `has_W_fixed` short-circuit skips two full n x n
        elementwise operations per forward pass in the common case.
        """
        _W = self.W_fast() + self.W_slow_module(self.device) * self.mask_W
        if not self.has_W_fixed:
            return _W
        return _W * (1.0 - self.W_fixed_mask) + self.W_fixed * self.W_fixed_mask

    def U(self):
        """Input gains: non-negative magnitude x anatomical input mask."""
        return torch.abs(self.U_raw) * self.mask_U

    # ==================================================================
    # penalties
    # ==================================================================
    def spectral_radius_power(self, W, n_iter=50, tol=1e-6):
        """
        Plain power iteration, fully inside the autograd graph.

        Kept unchanged for callers that use it for reporting/analysis. During
        training, prefer spectral_radius_differentiable, which is far cheaper.
        """
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

    def spectral_radius_differentiable(self, W, n_iter=None):
        """
        Dominant singular/eigen magnitude with a one-step gradient.

        The eigenvector is refined under no_grad (warm-started from the previous
        epoch, so a handful of steps suffices) and then held fixed while the
        returned value ||W v|| is differentiated. For a simple dominant
        eigenvalue this is the correct gradient of the spectral radius by the
        envelope theorem, but the graph holds one matvec instead of n_iter of
        them. Same trick PyTorch's own spectral_norm uses.
        """
        n_iter = self.power_iters if n_iter is None else n_iter

        with torch.no_grad():
            v = self.v_power
            if v.shape[0] != W.shape[0]:  # defensive: n_units changed
                v = torch.randn(W.shape[0], device=W.device)
                v = v / (v.norm() + 1e-8)
            for _ in range(n_iter):
                v_new = W @ v
                nrm = v_new.norm()
                if nrm < 1e-12:
                    # W annihilated the vector; restart from a random direction.
                    v_new = torch.randn_like(v)
                    nrm = v_new.norm()
                v = v_new / (nrm + 1e-8)
            if v.shape == self.v_power.shape:
                self.v_power.copy_(v)

        return torch.norm(W @ v)

    def fast_spectral_radius_penalty(self, margin=0.05):
        """One-sided penalty: only rho above (target + margin) is punished."""
        if self.effective_fast_spectral_radius_penalty_strength == 0:
            return 0

        rho_fast = self.spectral_radius_differentiable(self.W_fast())
        penalty = torch.relu(rho_fast - self.rho_target_fast - margin).pow(2)
        return penalty * self.effective_fast_spectral_radius_penalty_strength

    def _antagonism_from_projections(self, proj_L, proj_R, stim_side):
        """
        Ask the hemisphere ipsilateral to the stimulus to lead.

        `desired` is positive when the correct hemisphere's slow mode dominates,
        so relu(-desired) charges nothing when the ordering is already right.
        """
        desired = torch.where(
            stim_side[:, None] == 1,
            proj_L - proj_R,
            proj_R - proj_L
        )
        return torch.mean(torch.relu(-desired))

    def _stimulus_gated_slow_antagonism_penalty(self, h, stim_side, v_L, v_R):
        """Original signature, kept for external callers: projects then scores."""
        proj_L = torch.einsum("ntu,u->nt", h, v_L)
        proj_R = torch.einsum("ntu,u->nt", h, v_R)
        return self._antagonism_from_projections(proj_L, proj_R, stim_side)

    def slow_mode_directions(self):
        """Unit vectors summarising the left- and right-hemisphere slow modes."""
        v_L = self.W_slow_module.v_slow[:4].sum(dim=0)
        v_R = self.W_slow_module.v_slow[4:].sum(dim=0)
        v_L = v_L / (v_L.norm() + 1e-8)
        v_R = v_R / (v_R.norm() + 1e-8)
        return v_L, v_R

    def stimulus_gated_slow_antagonism_penalty(self, x_pred, stim_side, x_is_filtered=True):
        """
        Antagonism penalty on the GCaMP-filtered activity.

        `x_is_filtered=False` says x_pred is the raw activity, in which case the
        two scalar projections are computed first and the GCaMP kernel is
        applied to those instead. Convolution along time and a linear map across
        units commute, so the result matches filtering all n_units channels and
        projecting afterwards, at a cost of 2 channels instead of n_units.
        """
        if self.effective_slow_antagonism_penalty_strength == 0:
            return 0

        v_L, v_R = self.slow_mode_directions()

        proj_L = torch.einsum("ntu,u->nt", x_pred, v_L)
        proj_R = torch.einsum("ntu,u->nt", x_pred, v_R)

        if not x_is_filtered:
            proj = torch.stack((proj_L, proj_R), dim=-1)  # (N, T, 2)
            proj = DSService.apply_gcamp_kernel(proj, self.gcamp_tau_rise,
                                                self.gcamp_tau_decay, self.dt)
            proj_L, proj_R = proj[..., 0], proj[..., 1]

        penalty = self._antagonism_from_projections(proj_L, proj_R, stim_side)
        return self.effective_slow_antagonism_penalty_strength * penalty

    # ==================================================================
    # forward
    # ==================================================================
    def _prepare_x0(self, x0, N, device):
        """Broadcast whatever initial condition was supplied to (N, n_units)."""
        if x0 is None:
            x0 = torch.zeros(self.n_units, device=device)
        elif not torch.is_tensor(x0):
            x0 = torch.tensor(x0, dtype=torch.float32)
        if x0.ndim == 0:
            x0 = x0.unsqueeze(0).repeat(N, self.n_units)
        elif x0.ndim == 1:
            x0 = x0.unsqueeze(0).repeat(N, 1)
        x0 = x0.to(device)
        assert x0.shape[-1] == self.n_units, (
            f"x0 has {x0.shape[-1]} entries but the model has {self.n_units} "
            f"units. Build the initial condition by scattering into idx_list.")
        return x0

    def _integrate_chunk(self, h, drive_chunk, Wt, beta, one_minus_beta):
        """
        Integrate one contiguous block of time steps.

        drive_chunk is (N, T_chunk, n_units) and already contains inputs * U
        when the drive was precomputed, or zeros-plus-input otherwise. Returns
        the final state and the stacked activities of the chunk.

        Two deliberate choices here: the activities are collected in a Python
        list and stacked once (one autograd node, rather than one CopySlices per
        step from writing into a preallocated tensor), and Wt is passed in
        already transposed so the loop does not rebuild a view T times.
        """
        xs_chunk = []
        for t in range(drive_chunk.shape[1]):
            rec = self.f(h) @ Wt                     # (N, n_units)
            drive = rec + drive_chunk[:, t, :]
            h = beta * h + one_minus_beta * drive
            xs_chunk.append(self.f(h))
        return h, torch.stack(xs_chunk, dim=1)       # (N, T_chunk, n_units)

    def forward(self, x0, inputs, filter_xs=True):
        """
        Run the network.

        Dynamics run on all neurons; the 8 outputs are population means. Both
        outputs are GCaMP-filtered so they are comparable to dF/F.

        filter_xs=True  (default, original behaviour) returns the filtered
                        per-neuron activity, which is what the plotting and
                        variance-analysis helpers expect.
        filter_xs=False returns the unfiltered activity and skips a convolution
                        over n_units channels. fit() uses this: the only
                        consumer of xs during training is the antagonism
                        penalty, which can filter its 2 projections instead.
        """
        # ---- shape normalisation -------------------------------------------
        if inputs.ndim == 1:
            inputs = inputs[None, :, None]
        elif inputs.ndim == 2:
            inputs = inputs.unsqueeze(0)  # (1, T, input_dim)
        N, T, _ = inputs.shape
        device = inputs.device

        x0 = self._prepare_x0(x0, N, device)

        U = self.U().to(device)  # (n_units, input_dim)
        W = self.W().to(device)  # (n_units, n_units), post-by-pre

        # .T is a view, and a transposed operand makes every GEMM in the loop
        # take a slower path; materialising it once is worth the one copy.
        Wt = W.T.contiguous()

        beta = torch.exp(torch.tensor(-self.alpha, device=device))
        one_minus_beta = 1.0 - beta

        # ---- input drive ----------------------------------------------------
        # inputs is (N, T, n_units) when the caller supplies a per-neuron
        # stimulus, or (N, T, input_dim) with input_dim broadcasting against U.T.
        # Doing the multiply once collapses T small elementwise ops (and T
        # autograd nodes) into one.
        if self.precompute_input_drive:
            drive_in = inputs * U.T  # broadcast: (N, T, n_units)
        else:
            drive_in = None

        # ---- time loop ------------------------------------------------------
        h = x0
        if self.checkpoint_chunk:
            # Gradient checkpointing: activations inside a chunk are recomputed
            # during the backward pass instead of being kept. Roughly +30%
            # compute for a ~chunk/T fraction of the activation memory. Reach
            # for this when memory pressure, not arithmetic, is the bottleneck.
            xs_parts = []
            step = int(self.checkpoint_chunk)
            for start in range(0, T, step):
                stop = min(start + step, T)
                if drive_in is not None:
                    drive_chunk = drive_in[:, start:stop, :]
                else:
                    drive_chunk = inputs[:, start:stop, :] * U.T
                h, xs_part = checkpoint(self._integrate_chunk, h, drive_chunk,
                                        Wt, beta, one_minus_beta,
                                        use_reentrant=False)
                xs_parts.append(xs_part)
            xs = torch.cat(xs_parts, dim=1)
        else:
            if drive_in is None:
                drive_in = inputs * U.T
            h, xs = self._integrate_chunk(h, drive_in, Wt, beta, one_minus_beta)

        self.h = h

        # ---- population readout ---------------------------------------------
        # One (N, T, n_units) x (n_units, n_out) matmul rather than 8 gathers;
        # readout_W already holds 1/len(pop) so this is the population mean.
        ys = xs @ self.readout_W  # (N, T, n_out)

        # ---- calcium indicator ----------------------------------------------
        # The kernel is sum-normalised (unit DC gain), so this low-passes
        # without rescaling the steady state.
        ys = DSService.apply_gcamp_kernel(ys, self.gcamp_tau_rise, self.gcamp_tau_decay, self.dt)
        if not torch.is_tensor(ys):
            ys = torch.tensor(ys, dtype=torch.float32)
        self.ys = ys.to(device)

        if filter_xs:
            xs = DSService.apply_gcamp_kernel(xs, self.gcamp_tau_rise, self.gcamp_tau_decay, self.dt)
            if not torch.is_tensor(xs):
                xs = torch.tensor(xs, dtype=torch.float32)
        self.xs = xs.to(device)
        self.xs_is_filtered = bool(filter_xs)

        return self.xs, self.ys

    # ==================================================================
    # target alignment
    # ==================================================================
    def downsample_signal(self, raw_signal, time_sample_list):
        """
        Nearest-neighbour resampling of a simulated signal onto data timestamps.

        raw_signal: (N, T, dim); time_sample_list: (T_ds,) in seconds.
        """
        if not torch.is_tensor(time_sample_list):
            time_sample_list = torch.tensor(time_sample_list, dtype=torch.float32,
                                            device=raw_signal.device)
        t_raw = torch.arange(0, raw_signal.shape[1], device=raw_signal.device,
                             dtype=torch.float32) * self.dt
        dt = torch.abs(t_raw[:, None] - time_sample_list[None, :])  # (T, T_ds)
        idx_sample = torch.argmin(dt, dim=0)                        # (T_ds,)
        return raw_signal[:, idx_sample, :]

    def _infer_stim_side(self, train_list, device):
        """
        Which hemisphere each trial stimulates: +1 left, -1 right.

        Uses an explicit `stim_side` attribute when the dataset provides one,
        otherwise compares the total input delivered to each hemisphere. The
        fallback assumes the stimulus array is per-neuron and that hemispheres
        split at idx_side_change.
        """
        if hasattr(train_list[0], "stim_side") and train_list[0].stim_side is not None:
            return torch.tensor([int(t.stim_side) for t in train_list], device=device)

        sides = []
        for t in train_list:
            x = np.asarray(t.input_signal)
            if x.ndim == 2 and x.shape[1] >= self.n_units:
                left = np.sum(x[:, :self.n_units_hemi])
                right = np.sum(x[:, self.n_units_hemi:self.n_units])
                sides.append(1 if left > right else -1)
            else:
                sides.append(1)
        return torch.tensor(sides, device=device)

    # ==================================================================
    # training
    # ==================================================================
    def fit(self, train_list, x0=None, n_epochs=1000, verbose=True,
            downsample_target_list=None,
            stage_boundaries=None,
            lr_schedule=None, lr_factor=0.5, lr_patience=150, lr_min=1e-5,
            early_stopping_patience=None, early_stopping_min_delta=0.0,
            restore_best=True):
        """
        Full-batch BPTT with a staged regulariser curriculum.

        Stage 1: spectral penalty at 10% strength, no antagonism.
        Stage 2: full spectral penalty, no antagonism.
        Stage 3: full spectral penalty plus antagonism.

        Stage lengths
        -------------
        By default the boundaries are `stage1_frac`/`stage2_frac` of n_epochs,
        which means changing n_epochs also changes the curriculum. Pass
        `stage_boundaries=(e1, e2)` to pin them in absolute epochs -- e.g.
        (600, 1200) puts the antagonism penalty in play from epoch 1200 no
        matter how long the run turns out to be.

        Stopping early
        --------------
        `lr_schedule="plateau"` halves the learning rate whenever the MSE stops
        improving, and `early_stopping_patience` ends the run once it stops
        improving at all. Early stopping is only armed after stage 3 begins, so
        it cannot truncate the curriculum, and with `restore_best` the best
        weights seen are reloaded before returning. Together these let you keep
        n_epochs as a generous ceiling instead of a guess.
        """
        device = next(self.parameters()).device
        N = len(train_list)

        # ---- stage schedule -------------------------------------------------
        if stage_boundaries is not None:
            stage1_epochs, stage3_start = int(stage_boundaries[0]), int(stage_boundaries[1])
        else:
            stage1_epochs = int(self.stage1_frac * n_epochs)
            stage2_epochs = int(self.stage2_frac * n_epochs)
            stage3_start = stage1_epochs + stage2_epochs

        # ---- stack the dataset once -----------------------------------------
        inputs = [torch.as_tensor(np.asarray(t.input_signal), dtype=torch.float32) for t in train_list]
        outputs = [torch.as_tensor(np.asarray(t.output_signal), dtype=torch.float32) for t in train_list]
        inputs = torch.stack(inputs).to(device)    # (N, T, input_dim or n_units)
        outputs = torch.stack(outputs).to(device)  # (N, T_ds, 8)

        stim_side = self._infer_stim_side(train_list, device=device)

        # ---- initial conditions ---------------------------------------------
        if x0 is None:
            x0_list = []
            for t in train_list:
                if getattr(t, "initial_value", None) is None:
                    x0_list.append(torch.zeros(self.n_units, device=device))
                else:
                    x0_list.append(torch.as_tensor(np.asarray(t.initial_value),
                                                   dtype=torch.float32, device=device))
            x0 = torch.stack(x0_list).to(device)
        elif not torch.is_tensor(x0):
            x0 = torch.tensor(x0, dtype=torch.float32, device=device)
        if x0.ndim == 0:
            x0 = x0.unsqueeze(0).repeat(N, self.n_units)
        elif x0.ndim == 1:
            x0 = x0.unsqueeze(0).repeat(N, 1)
        x0 = x0.to(device)
        assert x0.shape == (N, self.n_units), (
            f"x0 is {tuple(x0.shape)} but should be ({N}, {self.n_units}).")

        if self.verbose_every is None:
            self.verbose_every = 50

        # ---- optional LR schedule -------------------------------------------
        scheduler = None
        if lr_schedule == "plateau":
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode="min", factor=lr_factor,
                patience=lr_patience, min_lr=lr_min)
        elif lr_schedule == "cosine":
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=n_epochs, eta_min=lr_min)
        elif lr_schedule is not None:
            raise ValueError(f"Unknown lr_schedule: {lr_schedule}")

        best_mse = float("inf")
        best_state = None
        epochs_without_improvement = 0

        for epoch in range(n_epochs):
            # ---- regulariser schedule ---------------------------------------
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

            self.optimizer.zero_grad(set_to_none=True)

            # filter_xs=False: skip the GCaMP convolution over n_units channels.
            # The antagonism penalty below filters its 2 projections instead,
            # which is the same computation by commutativity of the two linear
            # operations, and is where most of the per-epoch saving comes from.
            x_pred, y_pred = self.forward(x0, inputs, filter_xs=False)

            if downsample_target_list is not None:
                y_pred = self.downsample_signal(y_pred, downsample_target_list)

            mse = (y_pred - outputs).pow(2).mean()
            loss = mse
            loss = loss + self.fast_spectral_radius_penalty()
            loss = loss + self.stimulus_gated_slow_antagonism_penalty(
                x_pred, stim_side, x_is_filtered=False)

            self.loss_mse = mse.item()
            self.loss = loss.item()
            self.loss_reg = self.loss - self.loss_mse

            lr_now = self.optimizer.param_groups[0]["lr"]
            self.history["loss"].append(self.loss)
            self.history["mse"].append(self.loss_mse)
            self.history["reg"].append(self.loss_reg)
            self.history["lr"].append(lr_now)

            if verbose and (epoch % self.verbose_every == 0 or epoch in [stage1_epochs, stage3_start]):
                with torch.no_grad():
                    # Reported with the cheap estimator so that printing does
                    # not cost a second 50-step power iteration.
                    rho_fast = self.spectral_radius_differentiable(self.W_fast()).item()
                    yL_mean = y_pred[..., :4].mean().item()
                    yR_mean = y_pred[..., 4:].mean().item()
                print(f"[Stage {stage}] Epoch {epoch:4d} | Loss {self.loss:.6e} | "
                      f"MSE {self.loss_mse:.6e} | Reg {self.loss_reg:.6e} | "
                      f"rho_fast {rho_fast:.3f} | <yL> {yL_mean:.3f} | <yR> {yR_mean:.3f} | "
                      f"lr {lr_now:.2e}")

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.trainable_parameters(), max_norm=1.0)
            self.optimizer.step()

            if scheduler is not None:
                if lr_schedule == "plateau":
                    scheduler.step(self.loss_mse)
                else:
                    scheduler.step()

            # ---- best-so-far tracking and early stopping --------------------
            if self.loss_mse < best_mse - early_stopping_min_delta:
                best_mse = self.loss_mse
                epochs_without_improvement = 0
                if restore_best and early_stopping_patience is not None:
                    best_state = copy.deepcopy(self.state_dict())
            else:
                epochs_without_improvement += 1

            # Only armed in stage 3, so stopping can never cut the curriculum
            # short before the antagonism penalty has had its say.
            if (early_stopping_patience is not None
                    and stage == 3
                    and epochs_without_improvement >= early_stopping_patience):
                if verbose:
                    print(f"Early stopping at epoch {epoch}: no MSE improvement "
                          f"for {epochs_without_improvement} epochs "
                          f"(best MSE {best_mse:.6e}).")
                break

        if restore_best and best_state is not None:
            self.load_state_dict(best_state)
            self.loss_mse = best_mse

        # Trajectories are large and are regenerated on demand; dropping them
        # keeps them out of the checkpoint written by extract_custom_attrs.
        self.clear_state()

        return self.W()
