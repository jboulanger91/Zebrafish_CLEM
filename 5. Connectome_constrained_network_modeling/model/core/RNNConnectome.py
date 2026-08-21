import numpy as np
import torch
from torch import nn, optim

from model.core.PopulationSlow import PopulationSlow
from utils.services.ds_service import DSService
from utils.services.rnn_service import RNNService
from utils.configuration_rnn import ConfigurationRNN


class RNNConnectome(nn.Module):
    def __init__(
            self,
            dict_neurons,
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
            diag_zero_W=True,
            gcamp_tau_rise=0.25,
            gcamp_tau_decay=2.4,
            clamp_weights_min=None,
            clamp_weights_max=None,
            W_symmetric=False,
            stage1_frac=0.3,
            stage2_frac=0.3,
            verbose_every=None,  # if None -> default to 50 prints per run
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

        self.n_units_hemi = dict_neurons["idx_side_change"]
        self.n_units = dict_neurons["W"].shape[0]
        self.n_out = 8

        self.dt = dt
        self.alpha = dt / tau
        self.gcamp_tau_rise = gcamp_tau_rise
        self.gcamp_tau_decay = gcamp_tau_decay

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
        idx_LiMI = torch.tensor(dict_neurons["neurons"][ConfigurationRNN.SIDE_LEFT]["iMI"]["idx_list"])
        idx_LcMI = torch.tensor(dict_neurons["neurons"][ConfigurationRNN.SIDE_LEFT]["cMI"]["idx_list"])
        idx_LMON = torch.tensor(dict_neurons["neurons"][ConfigurationRNN.SIDE_LEFT]["MON"]["idx_list"])
        idx_LsMI = torch.tensor(dict_neurons["neurons"][ConfigurationRNN.SIDE_LEFT]["sMI"]["idx_list"])
        idx_L = torch.tensor(dict_neurons["neurons"][ConfigurationRNN.SIDE_LEFT]["idx_list"])

        self.register_buffer("idx_LiMI", idx_LiMI)
        self.register_buffer("idx_LcMI", idx_LcMI)
        self.register_buffer("idx_LMON", idx_LMON)
        self.register_buffer("idx_LsMI", idx_LsMI)
        self.register_buffer("idx_L", idx_L)

        # right hemisphere
        idx_RiMI = torch.tensor(dict_neurons["neurons"][ConfigurationRNN.SIDE_RIGHT]["iMI"]["idx_list"])
        idx_RcMI = torch.tensor(dict_neurons["neurons"][ConfigurationRNN.SIDE_RIGHT]["cMI"]["idx_list"])
        idx_RMON = torch.tensor(dict_neurons["neurons"][ConfigurationRNN.SIDE_RIGHT]["MON"]["idx_list"])
        idx_RsMI = torch.tensor(dict_neurons["neurons"][ConfigurationRNN.SIDE_RIGHT]["sMI"]["idx_list"])
        idx_R = torch.tensor(dict_neurons["neurons"][ConfigurationRNN.SIDE_RIGHT]["idx_list"])

        self.register_buffer("idx_RiMI", idx_RiMI)
        self.register_buffer("idx_RcMI", idx_RcMI)
        self.register_buffer("idx_RMON", idx_RMON)
        self.register_buffer("idx_RsMI", idx_RsMI)
        self.register_buffer("idx_R", idx_R)

        # idx_X = torch.arange(self.n_units_anchor, self.n_units)
        # self.register_buffer("idx_X", idx_X)
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

        # -------- biology constraints masks (E/I ratio, Dale's law, sparsity) --------
        if "W_mask" in dict_neurons.keys():
            mask_W = torch.tensor(dict_neurons["W_mask"], dtype=torch.float32)
        else:
            mask_W = torch.sign(dict_neurons["W"])
        if "U_mask" in dict_neurons.keys():
            mask_U = torch.tensor(dict_neurons["U_mask"], dtype=torch.float32)
        else:
            mask_U = torch.sign(dict_neurons["U"])
        if mask_U.dim() == 1: mask_U = mask_U.unsqueeze(1)
        self.register_buffer("mask_W", mask_W)
        self.register_buffer("mask_U", mask_U)

        # Example: all populations slow except motion onset (index 2 and 6)
        slow_pops = np.arange(n_slow_pops)  # array with population indices where the slow mode is applied
        self.W_slow_module = PopulationSlow(
            population_indices=self.population_indices,
            mask=self.mask_W,
            slow_populations=slow_pops,
            modes_per_population=2,
            gamma_init=0.995
        )

        # strengths
        self.fast_spectral_radius_penalty_strength = fast_spectral_radius_penalty_strength
        self.slow_antagonism_penalty_strength = slow_antagonism_penalty_strength
        self.rho_target_fast = rho_target_fast

        # effective (scheduled)
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
        return W_clamp * self.mask_W

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

    def fast_spectral_radius_penalty(self):
        if self.effective_fast_spectral_radius_penalty_strength == 0:
            return 0

        rho_fast = self.spectral_radius_power(self.W_fast())
        margin = 0.05
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
        return self.effective_slow_antagonism_penalty_strength * self._stimulus_gated_slow_antagonism_penalty(h=x_pred,
                                                                                                              stim_side=stim_side,
                                                                                                              v_L=v_L,
                                                                                                              v_R=v_R)

    # ---------- forward ----------
    def forward(self, x0, inputs):
        """
        Anchor readout:
          - dynamics still run on all neurons
          - population outputs y_k are averages over population activity
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

        U = self.U().to(device)  # (n_units, input_dim)
        W = self.W().to(device)  # (n_units, n_units)

        beta = torch.exp(torch.tensor(-self.alpha, device=device))

        for t in range(T):
            rec = self.f(self.h) @ W.T  # (N, n_units)
            inp = inputs[:, t, :] * U.T  # (N, n_units)

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
        idx_sample = torch.argmin(dt, dim=0)  # (T_ds,)
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
                right = np.sum(x[:, self.n_units_hemi:self.n_units])
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

        inputs = torch.stack(inputs).to(device)  # (N,T,input_dim)
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
