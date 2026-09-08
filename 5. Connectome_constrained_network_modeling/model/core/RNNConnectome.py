"""
RNNConnectomeV2
===============

Anatomically constrained rate RNN, revised so that the *recurrence read off the
connectome* is the thing that has to generate the four population dynamics.

What changed relative to `RNNConnectome`, and why
-------------------------------------------------

1. **Initialisation is scaled by in-degree, then pinned to an operating point.**
   The old init was `randn / sqrt(n_units)`, which is *dense* scaling. The slow
   real mode of an E/I network is carried by the mean (row-sum) mode, whose
   eigenvalue goes like `<w> * (f_E - f_I) * k` with `k` the mean in-degree --
   a *sum* over afferents, so it is what sparsity destroys fastest. At the new
   connectome's ~3.5 synapses/neuron the old init lands ~8x below the gain the
   task needs, i.e. deep in the leaky regime where BPTT has no gradient for
   long timescales either.
   Here rows are scaled by `1/sqrt(in-degree)` and then a single scalar is
   **bisected** so that the continuous recurrent gain `rho(D W_fast)` equals
   `init_rho_target` exactly at init (default 0.90 -> `tau_eff = tau/0.1`).
   No guessing: the network starts at a known, reportable timescale.

2. **The spectral penalty measures the right object.**
   The old one penalised `norm(W @ v)`, i.e. `||W||_2`, not `rho(W)`. Sparse
   Dale matrices are strongly non-normal (`||W||_2 / rho(W)` ~ 2.9 at 2%
   density), and power iteration does not even converge when the dominant
   eigenvalue is complex, so the estimate was biased high and the penalty
   effectively forbade the slow regime. Here the penalised quantity is the
   dominant eigenvalue of the **effective discrete Jacobian**
   `J = beta I + (1 - beta) W diag(f'(h))`, obtained exactly (`torch.linalg.eig`
   at N ~ 200 costs ~5 MFLOP) with a first-order-exact differentiable surrogate
   (Rayleigh quotient on the frozen left/right eigenvector pair), and it is
   re-expressed as the **continuous gain** `g = (|mu| - beta)/(1 - beta)`, which
   has O(1) excursions and therefore a well-conditioned penalty. The penalty is
   one-sided: only `g` above `1 - tau/tau_eff_max` is charged, so slow dynamics
   are permitted rather than punished.
   Whether `mu_max` is real or complex is logged: complex means oscillation, not
   integration, and is as diagnostic as the magnitude.

3. **`PopulationSlow` is gone.** In the old model `W()` computed
   `W_slow_module(...) * mask_W`; masking a rank-1 outer product destroys it
   (`rho` 1.807 -> 0.213 at 5% density), its `eta` was never in the optimizer's
   parameter list so the gammas never took a step, and `slow_mode_directions()`
   split `v_slow[:4]/[4:]` while `modes_per_population=2` produces 16 modes.
   Beyond the bugs: a hand-built population integrator that is not in the
   connectome is exactly the confound this model exists to avoid. The slow
   timescales now have to come from `W_fast`. The hemispheric antagonism penalty
   is kept but its directions are built from the population readout instead.

4. **A shared, non-fitted `tau` of 0.2 s.** One global constant, identical for
   all units; no population gets a timescale of its own, so all differentiation
   still has to come from `W`. It only relaxes the precision demanded of the
   eigenvalue: `tau_eff = tau / (1 - g)`, so 20 s needs `g = 0.99` at 0.2 s
   instead of `g = 0.995` at 0.1 s, and a 0.001 error in `g` moves `tau_eff` by
   2 s instead of 4 s.

5. **A calibrated readout, profiled out in closed form.** A per-population gain
   and offset sits between the GCaMP-filtered population mean and the target,
   because dF/F is not firing rate and the constant relating them is unknown per
   cell type. Without it the recurrent gain has to serve two masters -- set the
   timescale *and* hit the absolute dF/F level -- and they conflict: enforcing
   the gain floor with no calibration drove the loss from 1.5 to 9.3. Being
   *linear* nuisance parameters they are computed analytically each epoch rather
   than fitted (fitted alongside 631 weights at a shared learning rate they
   barely moved), which makes the loss exactly invariant to a positive
   per-population gain and an offset -- the same invariance the
   connectome-prediction script's criteria have, so the two agree on what counts
   as a fit. They give no population a timescale of its own, so the dynamics
   still come from `W`. `readout_calibration="learn"` fits them instead;
   `False` disables them.

6. **A loss that scores shape, not level.** Each population's squared error is
   divided by that population's target variance (otherwise the absolute dF/F
   level dominates and the optimiser spends its capacity there), a term on the
   time derivative is added (this is what actually constrains timescales), and
   **per-population normalised R^2** is reported so a failing population cannot
   hide inside one pooled MSE. `loss_normalise=False` recovers the plain MSE.

7. **The slow band is a constraint, not a hope.** BPTT through a leaky network
   has a gradient horizon equal to its own dynamical horizon: with a slowest
   mode of 0.4 s there is no gradient anywhere that says "build a 20 s mode",
   while there is always one that says "shrink". Measured on this connectome
   with a ceiling only, the gain fell monotonically 0.64 -> 0.48 over 25
   epochs. So the spectral penalty is **two-sided**: `tau_eff` is confined to
   `[tau_eff_min, tau_eff_max]` (default 2-30 s). That states the modelling
   assumption out loud -- the network integrates on the seconds timescale; now
   show whether the connectome can do that *and* match the traces -- rather than
   leaving it to an optimiser that structurally cannot find it. Pass
   `tau_eff_min=None` to drop the floor; running both ways is itself the
   experiment. A projected-gradient step additionally rescales the magnitudes
   whenever a step leaves the stable set, evaluated at the trial's *peak*
   operating point, which makes divergence impossible rather than merely
   discouraged.

Retained from the tuned version
-------------------------------
Post-by-pre `W`, Dale's law and topology structurally enforced through
`mask_W`, `clamp_weights_min` (soft, gradient-preserving), `W_fixed` pinning,
`mask_U`, exponential-Euler integration, the `unbind`-based time loop with
`addmm`/`addcmul` fusion and reused `f(h)`, GCaMP filtering of only the 8
readout channels during training, the memoised downsampling indices, optional
parameter packing and gradient checkpointing, plateau LR schedule, early
stopping and best-weight restore.
"""

import copy

import numpy as np
import torch
from torch import nn, optim
from torch.utils.checkpoint import checkpoint

from utils.services.ds_service import DSService
from utils.services.rnn_service import RNNService
from utils.config import ConfigurationRNN


class RNNConnectome(nn.Module):

    # ==================================================================
    # construction
    # ==================================================================
    def __init__(
            self,
            dict_neurons,
            W_fixed=None,
            input_dim=1,
            # ---- integration ------------------------------------------------
            tau=0.2, dt=0.01,
            activation='softplus',
            # ---- optimiser --------------------------------------------------
            lr=1e-3,
            weight_decay=0.0,
            # ---- initialisation ---------------------------------------------
            init_rho_target=0.90,     # continuous gain rho(D W_fast) at init
            init_d_ref=1.0,           # slope the init gain is measured at
            init_scale_by_indegree=True,
            seed=None,
            # ---- spectral control -------------------------------------------
            spectral_penalty_strength=1.0,
            tau_eff_max=30.0,         # ceiling on the slowest mode (seconds)
            tau_eff_min=2.0,          # FLOOR on the slowest mode; None = no floor
            spectral_mode="rayleigh",  # "rayleigh" | "power"
            power_iters=10,
            hold_gain_epochs=0,
            hold_gain_target=0.95,
            hold_gain_strength=1.0,
            # ---- hemispheric antagonism -------------------------------------
            slow_antagonism_penalty_strength=5e-4,
            # ---- readout / loss shaping -------------------------------------
            readout_calibration="profile",   # "profile" | "learn" | False
            tie_hemispheres=True,
            loss_normalise=True,
            loss_derivative_weight=0.3,
            # ---- anatomy ----------------------------------------------------
            use_connectome_mask_U=False,
            input_populations=None,   # e.g. [0, 2, 4, 6] -> only iMI and MON
            clamp_weights_min=1e-3,
            clamp_weights_max=None,
            clamp_soft=True,
            # ---- indicator ---------------------------------------------------
            gcamp_tau_rise=0.25,
            gcamp_tau_decay=2.4,
            # ---- misc --------------------------------------------------------
            device=None,
            verbose_every=None,
            pack_parameters=False,
            precompute_input_drive=True,
            checkpoint_chunk=None,
    ):
        super().__init__()

        self.dict_neurons = dict_neurons

        self.device = torch.device("cpu") if device is None else torch.device(device)
        self.to(self.device)

        # ---- loss bookkeeping -----------------------------------------------
        self.loss = None
        self.loss_mse = None
        self.loss_mse_raw = None
        self.loss_reg = None
        self.history = {"loss": [], "mse": [], "mse_raw": [], "reg": [],
                        "lr": [], "gain": [], "tau_eff": [], "imag_frac": []}

        # ---- weight-magnitude constraints -----------------------------------
        self.clamp_weights_min = clamp_weights_min
        self.clamp_weights_max = clamp_weights_max
        self.clamp_soft = bool(clamp_soft)

        # ---- sizes ----------------------------------------------------------
        self.n_units_hemi = dict_neurons["idx_side_change"]
        self.n_units = dict_neurons["W"].shape[0]
        self.n_out = 8

        # ---- integration constants ------------------------------------------
        self.dt = float(dt)
        self.tau = float(tau)
        self.alpha = self.dt / self.tau
        self.beta = float(np.exp(-self.alpha))
        self.gcamp_tau_rise = gcamp_tau_rise
        self.gcamp_tau_decay = gcamp_tau_decay

        self.verbose_every = verbose_every
        self.f = RNNService.activation_dict[activation]
        self.activation_name = activation

        # ---- performance switches -------------------------------------------
        self.pack_parameters = bool(pack_parameters)
        self.power_iters = int(power_iters)
        self.precompute_input_drive = bool(precompute_input_drive)
        self.checkpoint_chunk = checkpoint_chunk
        self.spectral_mode = str(spectral_mode)

        # ---- loss shaping ----------------------------------------------------
        self.loss_normalise = bool(loss_normalise)
        self.loss_derivative_weight = float(loss_derivative_weight)
        if readout_calibration in (True, "learn"):
            self.readout_calibration = "learn"
        elif readout_calibration in (False, None, "none"):
            self.readout_calibration = False
        elif readout_calibration == "profile":
            self.readout_calibration = "profile"
        else:
            raise ValueError('readout_calibration must be "profile", "learn" or False')
        self.tie_hemispheres = bool(tie_hemispheres)

        # =====================================================================
        # Anatomical masks (post-by-pre; entries 0 / +1 / -1 carry Dale's law)
        # =====================================================================
        if "W_mask" in dict_neurons.keys():
            mask_W = torch.as_tensor(np.asarray(dict_neurons["W_mask"]), dtype=torch.float32)
        else:
            mask_W = torch.sign(torch.as_tensor(np.asarray(dict_neurons["W"]), dtype=torch.float32))

        mask_U = self._build_mask_U(dict_neurons, input_dim, use_connectome_mask_U,
                                    input_populations)

        self.register_buffer("mask_W", mask_W)
        self.register_buffer("mask_U", mask_U)
        self.register_buffer("mask_W_support", (mask_W != 0).to(torch.float32), persistent=False)
        self.n_synapses = int(self.mask_W_support.sum().item())

        # In-degree per postsynaptic neuron: the quantity the mean mode sums over.
        in_degree = self.mask_W_support.sum(dim=1)
        self.register_buffer("in_degree", in_degree, persistent=False)
        self.mean_in_degree = float(in_degree.mean().item())

        # =====================================================================
        # Population indices (taken verbatim; need not be contiguous)
        # =====================================================================
        self._register_population_indices(dict_neurons)

        readout = torch.zeros(self.n_units, self.n_out)
        for k, idx in enumerate(self.population_indices):
            if len(idx) == 0:
                continue
            readout[torch.as_tensor(idx, dtype=torch.long), k] = 1.0 / len(idx)
        self.register_buffer("readout_W", readout, persistent=False)

        # Hemisphere directions for the antagonism penalty, from the readout
        # rather than from a hand-built slow module.
        v_L = readout[:, :4].sum(dim=1)
        v_R = readout[:, 4:].sum(dim=1)
        self.register_buffer("v_hemi_L", v_L / (v_L.norm() + 1e-8), persistent=False)
        self.register_buffer("v_hemi_R", v_R / (v_R.norm() + 1e-8), persistent=False)

        # =====================================================================
        # Fitted parameters
        # =====================================================================
        if seed is not None:
            torch.manual_seed(seed)

        W_raw = torch.randn(self.n_units, self.n_units)
        if init_scale_by_indegree:
            # Row-wise: keeps the per-neuron total drive comparable across
            # neurons with very different numbers of afferents, which at 3.5
            # synapses/neuron differ by an order of magnitude.
            W_raw = W_raw / torch.sqrt(torch.clamp(in_degree, min=1.0))[:, None]
        else:
            W_raw = W_raw / np.sqrt(self.n_units)
        # Mild left/right asymmetry so the hemispheres are not interchangeable.
        W_raw[:self.n_units_hemi] *= 0.95
        W_raw[self.n_units_hemi:] *= 1.05

        self.U_raw = nn.Parameter(
            torch.randn(self.n_units, input_dim) / np.sqrt(max(1, self.n_units * input_dim)))

        # ---- indicator calibration -------------------------------------------
        # dF/F is not firing rate: the scale factor and baseline relating them
        # are genuinely unknown and differ by cell type. Without them the
        # network has to hit the absolute dF/F level with its recurrent gain,
        # which fights directly against being a slow integrator -- forcing the
        # gain up then blows the amplitude up and the MSE with it (measured:
        # loss 1.5 -> 9.3 once the gain floor was enforced without this).
        # These give no population a timescale of its own, so the dynamics still
        # have to come from W.
        n_cal = 4 if self.tie_hemispheres else self.n_out
        self.log_readout_gain = nn.Parameter(torch.zeros(n_cal))
        self.readout_offset = nn.Parameter(torch.zeros(n_cal))

        if self.pack_parameters:
            idx_W_nz = torch.nonzero(mask_W.reshape(-1), as_tuple=False).squeeze(-1)
            self.register_buffer("idx_W_nz", idx_W_nz, persistent=False)
            self.W_vals = nn.Parameter(W_raw.reshape(-1)[idx_W_nz].clone())
            self.W_raw = None
        else:
            self.register_buffer("idx_W_nz", torch.empty(0, dtype=torch.long), persistent=False)
            self.W_raw = nn.Parameter(W_raw)

        # =====================================================================
        # Optionally pinned entries of W
        # =====================================================================
        if W_fixed is None:
            self.has_W_fixed = False
            self.register_buffer("W_fixed", torch.zeros(self.n_units, self.n_units))
            self.register_buffer("W_fixed_mask", torch.zeros(self.n_units, self.n_units))
        else:
            W_fixed = torch.as_tensor(np.asarray(W_fixed), dtype=torch.float32)
            assert W_fixed.shape == (self.n_units, self.n_units), \
                "Shape of W_fixed does not match W. Wrong number of neurons"
            self.has_W_fixed = True
            self.register_buffer("W_fixed_mask", (~torch.isnan(W_fixed)).to(torch.float32))
            self.register_buffer("W_fixed", torch.nan_to_num(W_fixed, nan=0.0))

        # =====================================================================
        # Operating-point slope, and the init gain bisection
        # =====================================================================
        # d0 = f'(h) at the h that gives f(h) = 1, i.e. the slope the network
        # actually runs at for dF/F of order 1. For softplus this is ~0.632.
        self.d0 = self._activation_slope_at_unit_output()
        self.register_buffer("v_power", self._unit_vector(self.n_units), persistent=False)

        # The init gain is pinned at `init_d_ref` (default 1.0 = softplus' maximum
        # slope), NOT at the slope for unit output. Softplus' slope grows with
        # activity, so a network pinned to gain 0.9 at f'=0.632 sits at gain
        # ~1.1 once the stimulus drives it up -- and an 8000-step trial at gain
        # 1.1 overflows on epoch 0, which is how a run ends up all-NaN. Pinning
        # against the worst-case slope guarantees no operating point the trial
        # visits can be unstable; the gain hold and the MSE then pull it up.
        self.init_rho_target = float(init_rho_target)
        self.init_d_ref = float(init_d_ref)
        gain_before = self.recurrent_gain(d_scalar=self.init_d_ref)
        self._rescale_to_gain(self.init_rho_target, d_scalar=self.init_d_ref)
        gain_after = self.recurrent_gain(d_scalar=self.init_d_ref)
        gain_at_unit = self.recurrent_gain(d_scalar=self.d0)

        # =====================================================================
        # Penalties
        # =====================================================================
        self.spectral_penalty_strength = float(spectral_penalty_strength)
        self.tau_eff_max = float(tau_eff_max)
        self.gain_target = 1.0 - self.tau / self.tau_eff_max
        # A FLOOR as well as a ceiling. Without it the run drifts leaky: BPTT
        # through a network whose slowest mode is 0.4 s has a gradient horizon of
        # 0.4 s, in case there is no gradient anywhere that says "build a 20 s mode",
        # while there is always a gradient that says "shrink".
        # Making the slow band a *constraint* states the modelling assumption
        # out loud -- the network integrates on the seconds timescale, now show
        # whether the connectome can do that AND match the traces -- instead of
        # leaving it to an optimiser that structurally cannot find it.
        # Set tau_eff_min=None to remove the floor; comparing the two runs is
        # itself the experiment.
        self.tau_eff_min = None if tau_eff_min is None else float(tau_eff_min)
        self.gain_floor = (None if self.tau_eff_min is None
                           else 1.0 - self.tau / self.tau_eff_min)
        self.slow_antagonism_penalty_strength = float(slow_antagonism_penalty_strength)
        self.effective_slow_antagonism_penalty_strength = 0.0

        self.hold_gain_epochs = int(hold_gain_epochs)
        self.hold_gain_target = float(hold_gain_target)
        self.hold_gain_strength = float(hold_gain_strength)

        # ---- rolling state ---------------------------------------------------
        self.h = None
        self.xs = None
        self.ys = None
        self.xs_is_filtered = None
        self._D_cache = None
        self._last_calibration = None

        self.optimizer = optim.Adam(self.trainable_parameters(), lr=lr,
                                    weight_decay=weight_decay)

        # ---- init report -----------------------------------------------------
        mu, imag_frac = self.dominant_jacobian_eigenvalue(d_scalar=self.init_d_ref,
                                                           detach=True)
        print("[RNNConnectomeV2] init")
        print(f"  n_units {self.n_units} | synapses {self.n_synapses} "
              f"({100.0 * self.n_synapses / self.n_units ** 2:.2f}% density) | "
              f"mean in-degree {self.mean_in_degree:.2f}")
        print(f"  tau {self.tau:.3f} s | dt {self.dt:.4f} s | beta {self.beta:.4f} | "
              f"activation {activation} (f'|_x=1 = {self.d0:.3f})")
        print(f"  readout calibration: {self.readout_calibration or 'OFF'}"
              f"{' (L/R tied)' if self.readout_calibration and self.tie_hemispheres else ''}")
        print(f"  recurrent gain rho(D W_fast) at f'={self.init_d_ref:.3f} (worst case): "
              f"{gain_before:.4f} -> {gain_after:.4f} after rescale "
              f"(target {self.init_rho_target})")
        print(f"  same at f'={self.d0:.3f} (unit output): {gain_at_unit:.4f} "
              f"-> tau_eff {self.tau / max(1e-9, 1 - gain_at_unit):.2f} s")
        print(f"  |mu_max(J)| {mu:.6f} -> tau_eff {self._tau_eff(mu):.2f} s | "
              f"imag fraction {imag_frac:.3f} "
              f"({'REAL: integration' if imag_frac < 0.1 else 'COMPLEX: oscillatory'})")
        if self.gain_floor is None:
            print(f"  gain band: (none, {self.gain_target:.4f}]  -> tau_eff up to "
                  f"{self.tau_eff_max} s, NO floor (the run may drift leaky)")
        else:
            print(f"  gain band: [{self.gain_floor:.4f}, {self.gain_target:.4f}]  -> "
                  f"tau_eff constrained to [{self.tau_eff_min}, {self.tau_eff_max}] s")
        if self.hold_gain_epochs:
            print(f"  extra gain hold at {self.hold_gain_target} for the first "
                  f"{self.hold_gain_epochs} epochs")

    # ------------------------------------------------------------------
    def _build_mask_U(self, dict_neurons, input_dim, use_connectome_mask_U,
                      input_populations):
        """Which neurons the stimulus is allowed to reach."""
        if use_connectome_mask_U:
            if "U_mask" in dict_neurons.keys():
                mask_U = torch.as_tensor(np.asarray(dict_neurons["U_mask"]), dtype=torch.float32)
            else:
                mask_U = torch.sign(torch.as_tensor(np.asarray(dict_neurons["U"]), dtype=torch.float32))
            if torch.max(mask_U) <= 0:
                print("WARNING | all-zero mask U found. To keep stimulus dependency, "
                      "mask U was set to all-ones.")
                mask_U = torch.ones_like(mask_U, dtype=torch.float32)
        elif input_populations is not None:
            # Restrict the direct feedforward path to named populations. With an
            # all-ones mask every neuron has its own private copy of the
            # stimulus, which is the cheapest descent direction and leaves the
            # recurrence unrecruited.
            mask_U = torch.zeros(int(self.n_units), input_dim, dtype=torch.float32)
            for p in input_populations:
                idx = torch.as_tensor(
                    dict_neurons["neurons"][
                        ConfigurationRNN.SIDE_LEFT if p < 4 else ConfigurationRNN.SIDE_RIGHT
                    ][["iMI", "cMI", "MON", "sMI"][p % 4]]["idx_list"], dtype=torch.long)
                mask_U[idx] = 1.0
            print(f"[RNNConnectomeV2] stimulus restricted to populations {list(input_populations)}: "
                  f"{int(mask_U.sum().item())}/{self.n_units} neurons driven directly")
        else:
            mask_U = torch.ones(int(self.n_units), input_dim, dtype=torch.float32)

        if mask_U.dim() == 1:
            mask_U = mask_U.unsqueeze(1)
        return mask_U

    def _register_population_indices(self, dict_neurons):
        L, R = ConfigurationRNN.SIDE_LEFT, ConfigurationRNN.SIDE_RIGHT
        for side, tag in ((L, "L"), (R, "R")):
            for cell in ("iMI", "cMI", "MON", "sMI"):
                self.register_buffer(
                    f"idx_{tag}{cell}",
                    torch.as_tensor(dict_neurons["neurons"][side][cell]["idx_list"],
                                    dtype=torch.long))
            self.register_buffer(
                f"idx_{tag}",
                torch.as_tensor(dict_neurons["neurons"][side]["idx_list"], dtype=torch.long))

        self.population_indices = [
            self.idx_LiMI.tolist(), self.idx_LcMI.tolist(),
            self.idx_LMON.tolist(), self.idx_LsMI.tolist(),
            self.idx_RiMI.tolist(), self.idx_RcMI.tolist(),
            self.idx_RMON.tolist(), self.idx_RsMI.tolist(),
        ]

    @staticmethod
    def _unit_vector(n):
        v = torch.randn(n)
        return v / v.norm()

    def _activation_slope_at_unit_output(self):
        """f'(h0) where f(h0) = 1; falls back to f'(0) for saturating f."""
        lo, hi = -30.0, 30.0
        f = lambda z: self.f(torch.tensor([z])).item()
        if f(hi) < 1.0:                      # sigmoid / tanh: never reaches 1
            h0 = 0.0
        else:
            for _ in range(80):
                mid = 0.5 * (lo + hi)
                if f(mid) < 1.0:
                    lo = mid
                else:
                    hi = mid
            h0 = 0.5 * (lo + hi)
        h = torch.tensor([h0], requires_grad=True)
        y = self.f(h).sum()
        d, = torch.autograd.grad(y, h)
        return float(d.item())

    # ==================================================================
    # helpers
    # ==================================================================
    def trainable_parameters(self):
        W_param = self.W_vals if self.pack_parameters else self.W_raw
        params = [W_param, self.U_raw]
        if self.readout_calibration == "learn":
            params += [self.log_readout_gain, self.readout_offset]
        return params

    def calibrate(self, ys):
        """Apply the *learned* indicator gain and offset (mode "learn" only)."""
        if self.readout_calibration != "learn":
            return ys
        a, b = torch.exp(self.log_readout_gain), self.readout_offset
        if self.tie_hemispheres:
            a, b = torch.cat([a, a]), torch.cat([b, b])
        return ys * a[None, None, :] + b[None, None, :]

    def profile_calibration(self, y_pred, outputs, eps=1e-8):
        """
        Closed-form per-population indicator gain and offset.

        `a` and `b` in `target ~ a * model + b` are *linear nuisance
        parameters*, so for any dynamics their optimum has a closed form. Fitting
        them by gradient descent alongside 631 weights does not work -- at a
        shared learning rate they barely move (measured: they sat at x1.00+0.00
        while the fit went nowhere) -- and it is unnecessary. Profiling them out
        instead makes the loss exactly invariant to a positive per-population
        gain and an additive offset, which is the same invariance the
        connectome-prediction criteria have, so the two agree on what counts as
        a fit.

        Gradients flow through `a` and `b` (they are functions of `y_pred`),
        which is what profile likelihood requires. `a` is floored at 0 because a
        negative indicator gain is not physical.
        """
        yp = y_pred.reshape(-1, y_pred.shape[-1])
        yt = outputs.reshape(-1, outputs.shape[-1])
        yp_m, yt_m = yp.mean(0, keepdim=True), yt.mean(0, keepdim=True)
        cov = ((yp - yp_m) * (yt - yt_m)).mean(0)
        var = ((yp - yp_m) ** 2).mean(0)
        # Floored just above zero rather than at zero: a population whose
        # prediction is uncorrelated with its target would otherwise get a = 0,
        # which detaches it from the loss entirely and it can never recover.
        a = torch.clamp(cov / (var + eps), min=1e-3)
        b = yt_m.squeeze(0) - a * yp_m.squeeze(0)
        if self.tie_hemispheres:
            a = 0.5 * (a[:4] + a[4:]).repeat(2)
            b = 0.5 * (b[:4] + b[4:]).repeat(2)
        return a, b

    def clear_state(self):
        self.h = None
        self.xs = None
        self.ys = None
        self.xs_is_filtered = None
        self._D_cache = None
        self._last_calibration = None

    def _tau_eff(self, mu_abs):
        mu_abs = float(mu_abs)
        return self.dt / max(1e-9, 1.0 - mu_abs) if mu_abs < 1.0 else float("inf")

    # ==================================================================
    # transforms
    # ==================================================================
    def _W_magnitude(self):
        if self.pack_parameters:
            flat = torch.zeros(self.n_units * self.n_units,
                               device=self.W_vals.device, dtype=self.W_vals.dtype)
            flat = flat.index_put((self.idx_W_nz,), self.W_vals)
            W_raw = flat.view(self.n_units, self.n_units)
        else:
            W_raw = self.W_raw

        mag = torch.abs(W_raw)
        if self.clamp_soft and self.clamp_weights_min:
            mag = self.clamp_weights_min + mag
            if self.clamp_weights_max is not None:
                mag = torch.clamp(mag, max=self.clamp_weights_max)
        else:
            mag = torch.clamp(mag, self.clamp_weights_min, self.clamp_weights_max)
        return mag

    def W_fast(self):
        return self._W_magnitude() * self.mask_W

    def W(self):
        """Effective recurrent matrix, post-by-pre. No slow add-on any more."""
        _W = self.W_fast()
        if not self.has_W_fixed:
            return _W
        return _W * (1.0 - self.W_fixed_mask) + self.W_fixed * self.W_fixed_mask

    def U(self):
        return torch.abs(self.U_raw) * self.mask_U

    # ==================================================================
    # spectral machinery
    # ==================================================================
    @torch.no_grad()
    def operating_point_slope(self, x_mean=None, reduce="mean"):
        """
        Per-unit `f'(h)` at the operating point the network actually visits.

        `forward` returns `x = f(h)`, not `h`, so `h` is recovered elementwise by
        bisection (vectorised, once per epoch, negligible) and the slope is then
        taken by autograd. This keeps the whole thing activation-agnostic: for
        softplus it reduces to `1 - exp(-x)`, but relu/elu/tanh work unchanged.

        `D` is a summary of where the network is running, not something to
        differentiate through, so it is detached deliberately.

        `reduce="max"` gives the largest slope the trial visits, which is the
        binding one for stability: softplus' slope rises with activity, so a
        network safe at its mean operating point can still run away at its peak.
        The mean is what gets reported; the max is what the projection uses.
        """
        if x_mean is None:
            return torch.full((self.n_units,), self.d0, device=self.mask_W.device)

        x_mean = x_mean.detach().reshape(-1)
        lo = torch.full_like(x_mean, -30.0)
        hi = torch.full_like(x_mean, 30.0)
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            below = self.f(mid) < x_mean
            lo = torch.where(below, mid, lo)
            hi = torch.where(below, hi, mid)
        h = 0.5 * (lo + hi)

        with torch.enable_grad():
            hh = h.clone().requires_grad_(True)
            y = self.f(hh).sum()
            d, = torch.autograd.grad(y, hh)
        return d.detach()

    def effective_jacobian(self, W=None, D=None, d_scalar=None):
        """
        `J = beta I + (1 - beta) W diag(f'(h))`.

        Note the right-multiplication: `d/dh_j [W f(h)]_i = W_ij f'(h_j)`.
        (`D @ W` has the same spectrum, being similar, but `W @ D` is the
        Jacobian itself and is what the eigenvectors below refer to.)
        """
        W = self.W() if W is None else W
        if D is None:
            D = torch.full((self.n_units,), self.d0 if d_scalar is None else d_scalar,
                           device=W.device, dtype=W.dtype)
        eye = torch.eye(self.n_units, device=W.device, dtype=W.dtype)
        return self.beta * eye + (1.0 - self.beta) * (W * D[None, :])

    @torch.no_grad()
    def recurrent_gain(self, W=None, D=None, d_scalar=None):
        """Continuous recurrent gain `rho(D W)`; `tau_eff = tau / (1 - gain)`."""
        W = self.W_fast() if W is None else W
        if D is None:
            D = torch.full((self.n_units,), self.d0 if d_scalar is None else d_scalar,
                           device=W.device, dtype=W.dtype)
        lam = torch.linalg.eigvals(W * D[None, :])
        return float(lam.abs().max().item())

    def _dominant_pair(self, J_detached):
        """
        Frozen left/right eigenvector pair of the dominant eigenvalue.

        `Vinv @ J @ V = diag(lambda)`, so row k of `Vinv` is the left eigenvector
        normalised to `u @ v = 1`. Differentiating `u @ J @ v` with `u, v` held
        fixed gives exactly `d lambda / dJ` (first-order eigenvalue
        perturbation) while keeping one matmul in the graph instead of an eig.
        """
        lam, V = torch.linalg.eig(J_detached)
        k = int(torch.argmax(lam.abs()).item())
        Vinv = torch.linalg.inv(V)
        u = Vinv[k, :]
        v = V[:, k]
        return lam, k, u, v

    def dominant_jacobian_eigenvalue(self, W=None, D=None, d_scalar=None, detach=False):
        """
        Returns (|mu_max|, imaginary fraction).

        Differentiable in `W` unless `detach=True`. Falls back to power
        iteration if the eigenvector basis is too ill-conditioned to invert --
        note that power iteration on `J` is far better behaved than on `W`,
        because the `beta I` shift moves the whole spectrum away from zero and
        shrinks the norm/radius gap.
        """
        J = self.effective_jacobian(W=W, D=D, d_scalar=d_scalar)

        if detach:
            with torch.no_grad():
                lam = torch.linalg.eigvals(J)
                k = int(torch.argmax(lam.abs()).item())
                mu = lam[k]
                return float(mu.abs().item()), float((mu.imag / (mu.abs() + 1e-12)).abs().item())

        if self.spectral_mode == "rayleigh":
            try:
                with torch.no_grad():
                    lam, k, u, v = self._dominant_pair(J.detach())
                    imag_frac = float((lam[k].imag / (lam[k].abs() + 1e-12)).abs().item())
                    ur, ui = u.real, u.imag
                    vr, vi = v.real, v.imag
                # |u J v| with u, v frozen; all-real algebra so autograd stays simple.
                a = ur @ J @ vr - ui @ J @ vi
                b = ur @ J @ vi + ui @ J @ vr
                return torch.sqrt(a * a + b * b + 1e-24), imag_frac
            except Exception as exc:                                # pragma: no cover
                if not getattr(self, "_warned_eig", False):
                    print(f"WARNING | eig-based spectral estimate failed ({exc}); "
                          f"falling back to power iteration on J.")
                    self._warned_eig = True

        # power-iteration fallback, warm-started
        with torch.no_grad():
            v = self.v_power
            if v.shape[0] != J.shape[0]:
                v = self._unit_vector(J.shape[0]).to(J.device)
            for _ in range(self.power_iters):
                v_new = J @ v
                nrm = v_new.norm()
                if nrm < 1e-12:
                    v_new = torch.randn_like(v)
                    nrm = v_new.norm()
                v = v_new / (nrm + 1e-8)
            if v.shape == self.v_power.shape:
                self.v_power.copy_(v)
        return torch.norm(J @ v), float("nan")

    @torch.no_grad()
    def project_gain(self, D_peak, D_mean=None, iters=30):
        """
        Projected-gradient step onto the feasible set `gain in [floor, ceiling]`.

        Two-sided, and hard, because neither side works as a penalty:

        * **Ceiling.** Everything interesting lives in the fourth decimal place
          of `|mu|`, so by the time a quadratic penalty is large enough to
          notice, an 8000-step trial has already overflowed and every later
          epoch is NaN. Evaluated at the trial's *peak* operating point, since
          softplus' slope rises with activity and a network safe at its mean can
          still run away at its peak.
        * **Floor.** Measured on this connectome, a floor *penalty* at strength
          1.0 contributed 0.19 to the loss while the shape term was 0.65 and
          falling -- so the shape gradient simply paid it and the gain still
          slid 0.64 -> 0.49. The floor has to be a constraint. Evaluated at the
          mean operating point, which is the timescale the traces see.

        The floor is applied first and the ceiling second, so stability always
        wins; if that ordering leaves the gain below the floor the band is
        infeasible for the current `D` spread and the run says so.

        Returns (gain after projection, whether it acted).
        """
        acted = False

        if self.gain_floor is not None:
            D_mean = D_peak if D_mean is None else D_mean
            if self.recurrent_gain(D=D_mean) < self.gain_floor:
                self._rescale_to_gain(self.gain_floor, D=D_mean, iters=iters, quiet=True)
                acted = True

        if self.recurrent_gain(D=D_peak) > self.gain_target:
            self._rescale_to_gain(self.gain_target, D=D_peak, iters=iters, quiet=True)
            acted = True
            if (self.gain_floor is not None
                    and self.recurrent_gain(D=D_mean) < self.gain_floor - 1e-3
                    and not getattr(self, "_warned_band", False)):
                print(f"WARNING | the gain band [{self.gain_floor:.4f}, "
                      f"{self.gain_target:.4f}] is infeasible: the peak operating "
                      f"point is far enough above the mean that satisfying the "
                      f"ceiling breaks the floor. Widen it (raise tau_eff_min or "
                      f"tau_eff_max) or reduce the drive.")
                self._warned_band = True

        return self.recurrent_gain(D=D_peak if D_mean is None else D_mean), acted

    def spectral_penalty(self, D=None, epoch=None):
        """
        One-sided ceiling on the slowest mode, plus the optional early hold.

        Two-sided band: `g` is charged above `1 - tau/tau_eff_max` (stability)
        and, unless `tau_eff_min is None`, below `1 - tau/tau_eff_min` (so the
        run cannot quietly slide into the leaky regime where it has no gradient
        for long timescales).

        The penalised variable is the *continuous* gain
        `g = (|mu| - beta) / (1 - beta)`, whose excursions are O(1); penalising
        `|mu|` directly is badly conditioned because everything interesting
        happens in its fourth decimal place.
        """
        mu_abs, imag_frac = self.dominant_jacobian_eigenvalue(D=D)
        gain = (mu_abs - self.beta) / (1.0 - self.beta)

        pen = self.spectral_penalty_strength * torch.relu(gain - self.gain_target).pow(2)
        if self.gain_floor is not None:
            pen = pen + self.spectral_penalty_strength * \
                torch.relu(self.gain_floor - gain).pow(2)

        if epoch is not None and self.hold_gain_epochs and epoch < self.hold_gain_epochs:
            pen = pen + self.hold_gain_strength * (gain - self.hold_gain_target).pow(2)

        detached_mu = float(mu_abs.item()) if torch.is_tensor(mu_abs) else float(mu_abs)
        return pen, detached_mu, float(gain.item()) if torch.is_tensor(gain) else float(gain), imag_frac

    # ==================================================================
    # hemispheric antagonism
    # ==================================================================
    def stimulus_gated_antagonism_penalty(self, x_pred, stim_side, x_is_filtered=False):
        """
        Ask the hemisphere ipsilateral to the stimulus to lead.

        Directions come from the population readout. When `x_pred` is unfiltered,
        the two scalar projections are filtered instead of all `n_units`
        channels -- convolution in time and a linear map across units commute.
        """
        if self.effective_slow_antagonism_penalty_strength == 0:
            return 0.0

        proj_L = torch.einsum("ntu,u->nt", x_pred, self.v_hemi_L)
        proj_R = torch.einsum("ntu,u->nt", x_pred, self.v_hemi_R)

        if not x_is_filtered:
            proj = torch.stack((proj_L, proj_R), dim=-1)
            proj = DSService.apply_gcamp_kernel(proj, self.gcamp_tau_rise,
                                                self.gcamp_tau_decay, self.dt)
            proj_L, proj_R = proj[..., 0], proj[..., 1]

        desired = torch.where(stim_side[:, None] == 1, proj_L - proj_R, proj_R - proj_L)
        return self.effective_slow_antagonism_penalty_strength * torch.mean(torch.relu(-desired))

    # ==================================================================
    # forward
    # ==================================================================
    def _prepare_x0(self, x0, N, device):
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
            f"x0 has {x0.shape[-1]} entries but the model has {self.n_units} units.")
        return x0

    def _integrate_chunk(self, h, drive_chunk, Wt, beta, one_minus_beta):
        """
        One contiguous block of time steps.

        `unbind(1)` rather than `drive_chunk[:, t, :]` inside the loop: the drive
        requires grad, so every `select` would record a node whose backward
        allocates a full (N, T, n_units) zero tensor and scatters one row into
        it, T times per backward pass. `unbind` is one `StackBackward` and is
        bit-identical. `f(h)` is computed once per step and reused; `addmm` and
        `addcmul` fuse the update; activities are stacked once.
        """
        drive_steps = drive_chunk.unbind(1)
        xs_chunk = []
        fh = self.f(h)
        for drive_t in drive_steps:
            drive = torch.addmm(drive_t, fh, Wt)
            h = torch.addcmul(beta * h, drive, one_minus_beta)
            fh = self.f(h)
            xs_chunk.append(fh)
        return h, torch.stack(xs_chunk, dim=1)

    def forward(self, x0, inputs, filter_xs=True, keep_h=False):
        if inputs.ndim == 1:
            inputs = inputs[None, :, None]
        elif inputs.ndim == 2:
            inputs = inputs.unsqueeze(0)
        N, T, _ = inputs.shape
        device = inputs.device

        x0 = self._prepare_x0(x0, N, device)

        U = self.U().to(device)
        W = self.W().to(device)
        Wt = W.contiguous()

        beta = torch.exp(torch.tensor(-self.alpha, device=device))
        one_minus_beta = 1.0 - beta

        drive_in = inputs * U.T if self.precompute_input_drive else None

        h = x0
        if self.checkpoint_chunk:
            xs_parts = []
            step = int(self.checkpoint_chunk)
            for start in range(0, T, step):
                stop = min(start + step, T)
                drive_chunk = (drive_in[:, start:stop, :] if drive_in is not None
                               else inputs[:, start:stop, :] * U.T)
                h, xs_part = checkpoint(self._integrate_chunk, h, drive_chunk,
                                        Wt, beta, one_minus_beta, use_reentrant=False)
                xs_parts.append(xs_part)
            xs = torch.cat(xs_parts, dim=1)
        else:
            if drive_in is None:
                drive_in = inputs * U.T
            h, xs = self._integrate_chunk(h, drive_in, Wt, beta, one_minus_beta)

        self.h = h

        ys = xs @ self.readout_W
        ys = DSService.apply_gcamp_kernel(ys, self.gcamp_tau_rise, self.gcamp_tau_decay, self.dt)
        ys = self.calibrate(ys)
        self.ys = ys.to(device)

        if filter_xs:
            xs = DSService.apply_gcamp_kernel(xs, self.gcamp_tau_rise,
                                              self.gcamp_tau_decay, self.dt)
        self.xs = xs.to(device)
        self.xs_is_filtered = bool(filter_xs)

        return self.xs, self.ys

    # ==================================================================
    # target alignment
    # ==================================================================
    def downsample_signal(self, raw_signal, time_sample_list):
        """Nearest-neighbour resample onto data timestamps; indices memoised."""
        if not torch.is_tensor(time_sample_list):
            time_sample_list = torch.tensor(time_sample_list, dtype=torch.float32,
                                            device=raw_signal.device)
        cache = getattr(self, "_ds_idx_cache", None)
        if cache is None:
            cache = self._ds_idx_cache = {}
        key = (int(raw_signal.shape[1]), float(self.dt), str(raw_signal.device),
               tuple(time_sample_list.detach().cpu().reshape(-1).tolist()))
        idx_sample = cache.get(key)
        if idx_sample is None:
            t_raw = torch.arange(0, raw_signal.shape[1], device=raw_signal.device,
                                 dtype=torch.float32) * self.dt
            idx_sample = torch.argmin(torch.abs(t_raw[:, None] - time_sample_list[None, :]), dim=0)
            cache[key] = idx_sample
        return raw_signal[:, idx_sample, :]

    def _infer_stim_side(self, train_list, device):
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
    # loss
    # ==================================================================
    @staticmethod
    def _population_weights(outputs, eps=1e-8):
        """1 / per-population target variance, mean-normalised to 1."""
        var = outputs.reshape(-1, outputs.shape[-1]).var(dim=0, correction=0)
        w = 1.0 / (var + eps)
        return w / w.mean()

    def compute_loss_terms(self, y_pred, outputs, w_pop=None, w_pop_d=None):
        """
        Returns (loss_shape, mse_raw, y_cal).

        `loss_shape` is the population-normalised error plus the derivative
        term, computed on the calibrated prediction; `mse_raw` is the plain
        pooled MSE of the *uncalibrated* prediction, kept so numbers stay
        comparable with earlier runs.
        """
        mse_raw = (y_pred - outputs).pow(2).mean()

        if self.readout_calibration == "profile":
            a, b = self.profile_calibration(y_pred, outputs)
            self._last_calibration = (a.detach(), b.detach())
            y_pred = y_pred * a[None, None, :] + b[None, None, :]

        resid = y_pred - outputs

        if not self.loss_normalise:
            loss_shape = mse_raw
        else:
            loss_shape = (resid.pow(2) * w_pop).mean()

        if self.loss_derivative_weight > 0:
            dy = y_pred[:, 1:] - y_pred[:, :-1]
            dt_ = outputs[:, 1:] - outputs[:, :-1]
            dres = (dy - dt_).pow(2)
            if self.loss_normalise:
                dres = dres * w_pop_d
            loss_shape = loss_shape + self.loss_derivative_weight * dres.mean()

        return loss_shape, mse_raw, y_pred

    @staticmethod
    @torch.no_grad()
    def per_population_r2(y_pred, outputs):
        """R^2 per population, pooled over trials and time."""
        yp = y_pred.reshape(-1, y_pred.shape[-1])
        yt = outputs.reshape(-1, outputs.shape[-1])
        ss_res = (yp - yt).pow(2).sum(dim=0)
        ss_tot = (yt - yt.mean(dim=0, keepdim=True)).pow(2).sum(dim=0)
        return (1.0 - ss_res / (ss_tot + 1e-12)).cpu().numpy()

    # ==================================================================
    # training
    # ==================================================================
    def fit(self, train_list, x0=None, n_epochs=1000, verbose=True,
            downsample_target_list=None,
            antagonism_start=None,
            lr_schedule="plateau", lr_factor=0.5, lr_patience=150, lr_min=1e-5,
            early_stopping_patience=300, early_stopping_min_delta=0.0,
            restore_best=True, project_gain=True,
            pop_labels=("LiMI", "LcMI", "LMON", "LsMI", "RiMI", "RcMI", "RMON", "RsMI")):
        """
        Full-batch BPTT.

        The regulariser curriculum is much shorter than before, because the
        spectral term is now a *ceiling* rather than a shrinker: it is on from
        epoch 0 (one-sided, so it charges nothing until the slowest mode
        approaches `tau_eff_max`) and the only staged term is the hemispheric
        antagonism, which starts at `antagonism_start` (default: a third of the
        run). The early gain hold, if enabled, occupies the first
        `hold_gain_epochs` epochs.
        """
        device = next(self.parameters()).device
        N = len(train_list)

        if antagonism_start is None:
            antagonism_start = int(0.33 * n_epochs)

        inputs = torch.stack([torch.as_tensor(np.asarray(t.input_signal), dtype=torch.float32)
                              for t in train_list]).to(device)
        outputs = torch.stack([torch.as_tensor(np.asarray(t.output_signal), dtype=torch.float32)
                               for t in train_list]).to(device)

        stim_side = self._infer_stim_side(train_list, device=device)

        # ---- initial conditions ---------------------------------------------
        if x0 is None:
            x0_list = []
            for t in train_list:
                iv = getattr(t, "initial_value", None)
                if iv is None:
                    x0_list.append(torch.zeros(self.n_units, device=device))
                else:
                    v = torch.as_tensor(np.asarray(iv), dtype=torch.float32, device=device)
                    if not torch.isfinite(v).all():
                        # inv_softplus of ~0 is -inf; keep it finite.
                        v = torch.nan_to_num(v, nan=-10.0, neginf=-10.0, posinf=10.0)
                    x0_list.append(v)
            x0 = torch.stack(x0_list).to(device)
        elif not torch.is_tensor(x0):
            x0 = torch.tensor(x0, dtype=torch.float32, device=device)
        if x0.ndim == 1:
            x0 = x0.unsqueeze(0).repeat(N, 1)
        x0 = x0.to(device)
        assert x0.shape == (N, self.n_units), \
            f"x0 is {tuple(x0.shape)} but should be ({N}, {self.n_units})."

        if self.verbose_every is None:
            self.verbose_every = max(1, n_epochs // 50)

        # ---- loss weights (fixed for the run) -------------------------------
        w_pop = self._population_weights(outputs).to(device)
        d_out = outputs[:, 1:] - outputs[:, :-1]
        w_pop_d = self._population_weights(d_out).to(device)
        if verbose:
            print("[fit] population weights 1/var: " +
                  " ".join(f"{lab} {v:.2f}" for lab, v in zip(pop_labels, w_pop.tolist())))

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

        best_metric = float("inf")
        best_state = None
        epochs_without_improvement = 0
        n_projections = 0
        n_nonfinite = 0

        for epoch in range(n_epochs):
            self.effective_slow_antagonism_penalty_strength = (
                self.slow_antagonism_penalty_strength if epoch >= antagonism_start else 0.0)

            self.optimizer.zero_grad(set_to_none=True)

            x_pred, y_pred_full = self.forward(x0, inputs, filter_xs=False)

            if not torch.isfinite(x_pred).all():
                # Recover rather than spending the rest of the run on NaN.
                n_nonfinite += 1
                if verbose:
                    print(f"[{epoch:5d}] non-finite trajectory; restoring the best "
                          f"state, projecting the gain and halving the lr.")
                if best_state is not None:
                    self.load_state_dict(best_state)
                else:
                    with torch.no_grad():
                        self._rescale_to_gain(0.8 * self.gain_target,
                                              d_scalar=self.init_d_ref, quiet=True)
                for g in self.optimizer.param_groups:
                    g["lr"] = max(g["lr"] * 0.5, 1e-6)
                self.optimizer.state = type(self.optimizer.state)()
                if n_nonfinite > 10:
                    print("Aborting: 10 non-finite trajectories. Lower lr or "
                          "init_rho_target.")
                    break
                continue

            y_pred = (self.downsample_signal(y_pred_full, downsample_target_list)
                      if downsample_target_list is not None else y_pred_full)

            loss_shape, mse_raw, y_cal = self.compute_loss_terms(
                y_pred, outputs, w_pop, w_pop_d)
            loss = loss_shape

            # Operating point from this epoch's own trajectory: the gain that
            # matters is the one at the h the network actually visits, not the
            # one at h = 0.
            D_now = self.operating_point_slope(x_pred.detach().mean(dim=(0, 1)))
            spec_pen, mu_abs, gain, imag_frac = self.spectral_penalty(D=D_now, epoch=epoch)
            loss = loss + spec_pen

            ant = self.stimulus_gated_antagonism_penalty(x_pred, stim_side, x_is_filtered=False)
            loss = loss + ant

            self.loss = float(loss.item())
            self.loss_mse = float(loss_shape.item())
            self.loss_mse_raw = float(mse_raw.item())
            self.loss_reg = self.loss - self.loss_mse

            lr_now = self.optimizer.param_groups[0]["lr"]
            self.history["loss"].append(self.loss)
            self.history["mse"].append(self.loss_mse)
            self.history["mse_raw"].append(self.loss_mse_raw)
            self.history["reg"].append(self.loss_reg)
            self.history["lr"].append(lr_now)
            self.history["gain"].append(gain)
            self.history["tau_eff"].append(self._tau_eff(mu_abs))
            self.history["imag_frac"].append(imag_frac)

            if verbose and (epoch % self.verbose_every == 0
                            or epoch in (self.hold_gain_epochs, antagonism_start)):
                r2 = self.per_population_r2(y_cal, outputs)
                te = self._tau_eff(mu_abs)
                kind = "re" if (imag_frac == imag_frac and imag_frac < 0.1) else "im"
                band = ("lo" if (self.gain_floor is not None and gain < self.gain_floor - 1e-4)
                        else "hi" if gain > self.gain_target + 1e-4 else "in")
                print(f"[{epoch:5d}] loss {self.loss:.5e} | shape {self.loss_mse:.5e} | "
                      f"mse_uncal {self.loss_mse_raw:.3e} | reg {self.loss_reg:.2e} | "
                      f"gain {gain:.4f}[{band}] tau_eff {te:7.2f}s ({kind}) | "
                      f"lr {lr_now:.1e}")
                print("        R2  " + "  ".join(f"{lab} {v:5.2f}"
                                                 for lab, v in zip(pop_labels, r2)))

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.trainable_parameters(), max_norm=1.0)
            self.optimizer.step()

            # Project back inside the feasible gain band, using the peak operating
            # point of the trial just simulated for the ceiling and the mean for
            # the floor.
            if project_gain:
                D_peak = self.operating_point_slope(x_pred.detach().amax(dim=(0, 1)))
                _, acted = self.project_gain(D_peak, D_mean=D_now)
                n_projections += int(acted)

            if scheduler is not None:
                scheduler.step(self.loss_mse) if lr_schedule == "plateau" else scheduler.step()

            # Model selection on the shape loss, not the pooled MSE: the pooled
            # MSE is minimised by matching the mean level of every population.
            metric = self.loss_mse
            if not np.isfinite(metric):
                epochs_without_improvement += 1
            elif metric < best_metric - early_stopping_min_delta:
                best_metric = metric
                epochs_without_improvement = 0
                if restore_best:
                    best_state = {k: (v.detach().clone() if torch.is_tensor(v) else copy.deepcopy(v))
                                  for k, v in self.state_dict().items()}
            else:
                epochs_without_improvement += 1

            past_curriculum = epoch >= max(antagonism_start, self.hold_gain_epochs)
            if (early_stopping_patience is not None and past_curriculum
                    and epochs_without_improvement >= early_stopping_patience):
                if verbose:
                    print(f"Early stopping at epoch {epoch}: no improvement for "
                          f"{epochs_without_improvement} epochs (best {best_metric:.6e}).")
                break

        if restore_best and best_state is not None:
            self.load_state_dict(best_state)
            self.loss_mse = best_metric

        if verbose:
            print(f"\n[fit] gain projections: {n_projections} epochs | "
                  f"non-finite recoveries: {n_nonfinite}")
            self.report(train_list, x0=x0, downsample_target_list=downsample_target_list,
                        pop_labels=pop_labels)

        self.clear_state()
        return self.W()

    # ==================================================================
    # reporting
    # ==================================================================
    @torch.no_grad()
    def report(self, train_list, x0=None, downsample_target_list=None,
               pop_labels=("LiMI", "LcMI", "LMON", "LsMI", "RiMI", "RcMI", "RMON", "RsMI"),
               n_modes=6):
        """
        Post-fit diagnosis: the spectrum the fit actually reached, and how the
        slow modes project onto each population.

        This is the table that says whether the connectome produced distinct
        timescales or one fused mode: if every population's slowest strongly
        overlapping mode has the same `tau`, the readouts can only differ in
        amplitude.
        """
        device = next(self.parameters()).device
        inputs = torch.stack([torch.as_tensor(np.asarray(t.input_signal), dtype=torch.float32)
                              for t in train_list]).to(device)
        outputs = torch.stack([torch.as_tensor(np.asarray(t.output_signal), dtype=torch.float32)
                               for t in train_list]).to(device)
        if x0 is None:
            x0 = torch.zeros(len(train_list), self.n_units, device=device)

        _, y_full = self.forward(x0, inputs, filter_xs=False)
        y_pred = (self.downsample_signal(y_full, downsample_target_list)
                  if downsample_target_list is not None else y_full)
        with torch.enable_grad():
            _, _, y_cal = self.compute_loss_terms(
                y_pred, outputs, self._population_weights(outputs).to(device),
                self._population_weights(outputs[:, 1:] - outputs[:, :-1]).to(device))
        y_cal = y_cal.detach()
        r2 = self.per_population_r2(y_cal, outputs)

        J = self.effective_jacobian()
        lam = torch.linalg.eigvals(J)
        order = torch.argsort(lam.abs(), descending=True)
        lam_s = lam[order]
        _, V = torch.linalg.eig(J)
        V = V[:, order]

        print("\n=== RNNConnectomeV2 report ===")
        print("per-population R^2: " + "  ".join(f"{lab} {v:5.2f}"
                                                 for lab, v in zip(pop_labels, r2)))
        print(f"recurrent gain rho(D W_fast) = {self.recurrent_gain():.4f}  "
              f"(band [{'none' if self.gain_floor is None else format(self.gain_floor, '.4f')}, "
              f"{self.gain_target:.4f}])")
        if self.readout_calibration == "profile":
            a, b = self._last_calibration
        elif self.readout_calibration == "learn":
            a, b = torch.exp(self.log_readout_gain).detach(), self.readout_offset.detach()
        else:
            a = b = None
        if a is not None:
            a, b = a.cpu().numpy(), b.cpu().numpy()
            print(f"indicator calibration ({self.readout_calibration})  " +
                  "  ".join(f"{l} x{ai:.2f}{bi:+.2f}"
                            for l, ai, bi in zip(pop_labels, a, b)))
        print(f"\nslowest {n_modes} Jacobian modes:")
        print(f"  {'|mu|':>8} {'tau (s)':>9} {'kind':>5} | " +
              " ".join(f"{lab:>6}" for lab in pop_labels))
        for i in range(min(n_modes, lam_s.shape[0])):
            mu = lam_s[i]
            te = self._tau_eff(float(mu.abs().item()))
            kind = "re" if abs(float(mu.imag.item())) / (float(mu.abs().item()) + 1e-12) < 0.1 else "im"
            v = V[:, i]
            v = v / (v.abs().norm() + 1e-12)
            # |overlap| of each population's readout with this mode
            ov = (self.readout_W.T.to(v.dtype) @ v).abs()
            ov = ov / (ov.max() + 1e-12)
            print(f"  {float(mu.abs().item()):8.5f} {te:9.2f} {kind:>5} | " +
                  " ".join(f"{float(o):6.2f}" for o in ov))
        print("  (overlap normalised per row; a population whose slow-mode overlaps are all "
              "small\n   cannot integrate, and populations sharing one mode share its timescale.)")
        return {"r2": r2, "eigvals": lam_s.cpu().numpy()}

    # ==================================================================
    # init helper
    # ==================================================================
    @torch.no_grad()
    def _rescale_to_gain(self, target, d_scalar=None, D=None, lo=1e-4, hi=1e4,
                         iters=60, quiet=False):
        """
        Bisect a scalar on `W_raw` so that `rho(D W_fast) == target` at init.

        `W_fast` is monotone in this scalar (magnitudes are `clamp_min + c|W_raw|`),
        so bisection is exact and costs ~60 eigvals of a 174x174 matrix.
        """
        param = self.W_vals if self.pack_parameters else self.W_raw
        base = param.detach().clone()

        if d_scalar is None and D is None:
            d_scalar = self.d0

        def gain_at(c):
            param.copy_(base * c)
            return self.recurrent_gain(d_scalar=d_scalar, D=D)

        if gain_at(lo) > target:
            # The clamp_weights_min floor alone already exceeds the target.
            if not quiet:
                print(f"WARNING | the magnitude floor alone gives gain "
                      f"{gain_at(lo):.4f} > {target}. Lower clamp_weights_min.")
            param.copy_(base * lo)
            return
        g_hi = gain_at(hi)
        if g_hi < target:
            if not quiet:
                print(f"WARNING | cannot reach gain {target}: even c={hi} gives "
                      f"{g_hi:.4f}. Left at c={hi}. Check mask_W for excitatory loops.")
            return
        for _ in range(iters):
            mid = np.sqrt(lo * hi)
            if gain_at(mid) < target:
                lo = mid
            else:
                hi = mid
        param.copy_(base * np.sqrt(lo * hi))
