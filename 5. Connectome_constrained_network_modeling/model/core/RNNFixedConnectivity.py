import numpy as np
import torch
from torch import nn, optim

# Manually add root path for imports to improve interoperability
import sys; sys.path.insert(0, "../..")

from utils.services.ds_service import DSService
from utils.services.rnn_service import RNNService
from utils.configuration_rnn import ConfigurationRNN


class RNNFixedConnectivity(nn.Module):
    def __init__(
        self, W_norm, dict_neurons,
        n_beta=1, input_dim=1,
        tau=0.1, dt=1.0,
        fixed_U=None,
        sparsity_U=1,  # only considered if fixed_U is None
        lr=1e-3,
        weight_decay=1e-5,
        activation='softplus',
        seed=None,
        device=None,
        gcamp_tau_rise=0.25,
        gcamp_tau_decay=2.4,
        clamp_weights_min=None,
        clamp_weights_max=None,
        verbose_every=None,        # if None -> default to 50 prints per run
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
        # self.n_hemi = len(ConfigurationRNN.side_list)
        # self.n_units_anchor = self.n_hemi * self.n_units_hemi
        self.n_units = int(W_norm.shape[0])
        self.n_out = 8

        self.dt = dt
        self.alpha = dt / tau
        self.gcamp_tau_rise = gcamp_tau_rise
        self.gcamp_tau_decay = gcamp_tau_decay

        # self.stage1_frac = float(stage1_frac)
        # self.stage2_frac = float(stage2_frac)
        self.verbose_every = verbose_every

        self.f = RNNService.activation_dict[activation]

        # -------- parameters --------
        if seed is not None:
            torch.manual_seed(seed)

        self.register_buffer("W_norm", torch.tensor(W_norm, dtype=torch.float32))
        self.beta = nn.Parameter(torch.rand(n_beta, dtype=torch.float32).squeeze())

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
            # self.idx_X.tolist(),
        ]

        # -------- sparsity mask for U --------
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

        # state buffers (set in forward)
        self.h = None
        self.xs = None
        self.ys = None

        if fixed_U is None:
            self.U_raw = nn.Parameter(torch.randn(self.n_units, input_dim) / np.sqrt(max(1, input_dim)))
            self.optimizer = optim.Adam([self.beta, self.U_raw], lr=lr, weight_decay=weight_decay)
            self.register_buffer("mask_U", block_mask(self.n_units, input_dim, sparsity_U, seed).squeeze())
        else:
            self.optimizer = optim.Adam([self.beta], lr=lr, weight_decay=weight_decay)
            self.register_buffer("mask_U", block_mask(self.n_units, input_dim, 1, seed).squeeze())
            self.register_buffer("U_raw", fixed_U.float())

    def process_beta(self):
        if self.beta.dim() > 0:
            # map the 8 population-related beta values (associated to the input into each population) into a full matrix
            beta = torch.concat([torch.ones((len(pop), self.n_units), dtype=torch.float32) * self.beta[i_pop]
                                 for i_pop, pop in enumerate(self.population_indices)])
        else:
            beta = self.beta
        return beta

    # ---------- transforms ----------
    def W(self):
        beta = self.process_beta()
        W = self.W_norm * beta
        return W

    def U(self):
        return torch.abs(self.U_raw) * self.mask_U

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

    def downsample_signal(self, raw_signal, time_sample_list):
        # raw_signal: (N,T,dim), time_sample_list: (T_ds,)
        if not torch.is_tensor(time_sample_list):
            time_sample_list = torch.tensor(time_sample_list, dtype=torch.float32, device=raw_signal.device)
        t_raw = torch.arange(0, raw_signal.shape[1], device=raw_signal.device, dtype=torch.float32) * self.dt
        dt = torch.abs(t_raw[:, None] - time_sample_list[None, :])  # (T, T_ds)
        idx_sample = torch.argmin(dt, dim=0)                         # (T_ds,)
        return raw_signal[:, idx_sample, :]

    def fit(self, train_list, x0=None, n_epochs=1000, verbose=True, downsample_target_list=None):
        device = next(self.parameters()).device
        N = len(train_list)

        # stack inputs/outputs
        inputs = [torch.tensor(t.input_signal, dtype=torch.float32) for t in train_list]
        outputs = [torch.tensor(t.output_signal, dtype=torch.float32) for t in train_list]

        inputs = torch.stack(inputs).to(device)    # (N,T,input_dim)
        outputs = torch.stack(outputs).to(device)  # (N,T,8)

        # stim_side = self._infer_stim_side(train_list, device=device)  # UNCOMMENT when reintroduce regularization

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
            self.optimizer.zero_grad()

            x_pred, y_pred = self.forward(x0, inputs)

            if downsample_target_list is not None:
                y_pred = self.downsample_signal(y_pred, downsample_target_list)

            mse = (y_pred - outputs).pow(2).mean()
            loss = mse

            self.loss_mse = mse.item()
            self.loss_reg = (loss - mse).item()
            self.loss = loss.item()

            if verbose and (epoch % self.verbose_every == 0):  # or epoch in [stage1_epochs, stage3_start]):  # UNCOMMENT when reintroduce staged regularization
                with torch.no_grad():
                    rho = self.spectral_radius_power(self.W()).item()
                    yL_mean = y_pred[..., :4].mean().item()
                    yR_mean = y_pred[..., 4:].mean().item()
                print(f"[Epoch {epoch:4d} | Loss {loss.item():.6e} | "
                      f"MSE {mse.item():.6e} | Reg {self.loss_reg:.6e} | "
                      f"rho {rho:.3f} | <yL> {yL_mean:.3f} | <yR> {yR_mean:.3f}")

            loss.backward()
            torch.nn.utils.clip_grad_norm_([self.U_raw], max_norm=1.0)
            self.optimizer.step()

        return self.W()
