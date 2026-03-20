import numpy as np
import torch

from scipy.stats import wasserstein_distance
from torch import nn

from analysis.personal_dirs.Roberto.projects.rnn_ds.plot.style import RNNDSStyle
from analysis.personal_dirs.Roberto.projects.rnn_ds.utils.configuration_rnn import ConfigurationRNN
from analysis.personal_dirs.Roberto.projects.rnn_ds.utils.ds_service import DSService
from analysis.utils.figure_helper import Figure


class RNNService:
    activation_dict = {'relu': nn.ReLU(),
                       'elu': nn.ELU(),
                       'softplus': nn.Softplus(),
                       'sigmoid': nn.Sigmoid(),
                       'tanh': nn.Tanh()}

    @staticmethod
    @torch.no_grad()
    def compute_effective_jacobian(W, h):
        """
        W: (N, N) recurrent weight matrix
        h: (T, B, N) or (B, N) hidden states
        """
        if h.dim() == 3:
            h0 = h.reshape(-1, h.shape[-1])
        else:
            h0 = h

        # Softplus derivative: sigmoid
        phi_prime = torch.sigmoid(h0)
        phi_prime = phi_prime.mean(dim=0)  # average over time/batch

        D = torch.diag(phi_prime)
        W_eff = D @ W
        return W_eff

    @staticmethod
    @torch.no_grad()
    def eigen_timescales(W_eff, dt, tau):
        """
        Returns eigenvalues and associated timescales (seconds)
        """
        eigvals, eigvecs = torch.linalg.eig(W_eff)
        eigvals = eigvals.real  # imaginary parts correspond to oscillations

        alpha = dt / tau
        mu = 1 - alpha + alpha * eigvals  # discrete-time Jacobian eigenvalues

        # avoid numerical issues
        eps = 1e-6
        timescales = dt / torch.clamp(1 - mu, min=eps)

        return eigvals, mu, timescales, eigvecs

    @staticmethod
    @torch.no_grad()
    def slow_mode_alignment(eigvecs, timescales, W_out, k=5):
        """
        Measures alignment of slow modes with output weights
        """
        idx = torch.argsort(timescales, descending=True)[:k]

        alignments = []
        for i in idx:
            v = eigvecs[:, i].real
            v = v / torch.norm(v)

            proj = torch.norm(W_out @ v)
            alignments.append((timescales[i].item(), proj.item()))

        return alignments

    @classmethod
    def _as_tensor(cls, x, device):
        if torch.is_tensor(x):
            return x.to(device)
        return torch.tensor(x, dtype=torch.float32, device=device)

    @classmethod
    @torch.no_grad()
    def run_model_get_xs(cls, model, train_list, x0=None, use_raw_if_available=True):
        """
        Returns:
            xs: (N, T, n_units) torch.Tensor
        """
        device = next(model.parameters()).device

        inputs = torch.stack([cls._as_tensor(t.input_signal, device) for t in train_list])  # (N,T,input_dim) or (N,T)
        if inputs.ndim == 2:
            inputs = inputs[..., None]  # (N,T,1)

        N = inputs.shape[0]

        if x0 is None:
            x0_list = []
            for t in train_list:
                iv = getattr(t, "initial_value", None)
                if iv is None:
                    x0_list.append(torch.zeros(model.n_units, device=device))
                else:
                    x0_list.append(cls._as_tensor(iv, device).view(-1))
            x0 = torch.stack(x0_list, dim=0)  # (N, n_units)
        else:
            x0 = cls._as_tensor(x0, device)
            if x0.ndim == 1:
                x0 = x0[None, :].repeat(N, 1)

        # forward
        model.eval()
        _ = model.forward(x0, inputs)

        # choose xs
        if use_raw_if_available and hasattr(model, "xs_raw") and model.xs_raw is not None:
            xs = model.xs_raw
        else:
            xs = model.xs

        return xs

    @classmethod
    def per_neuron_variance(cls, xs, mode="time", correction=0):
        """
        xs: (N,T,U)
        mode:
          - "time": for each neuron/trial, var over time -> (N,U), then average over trials -> (U,)
          - "time_and_trials": flatten N and T and var -> (U,)
          - "trials": var across trials of time-mean -> (U,)
        correction: torch.var correction (0 => population variance)
        """
        if mode == "time":
            # var over time within each trial -> (N,U), then mean over trials -> (U,)
            v = xs.var(dim=1, correction=correction)  # (N,U)
            return v.mean(dim=0)  # (U,)
        elif mode == "time_and_trials":
            # var over flattened NT -> (U,)
            x = xs.reshape(-1, xs.shape[-1])  # (NT,U)
            return x.var(dim=0, correction=correction)  # (U,)
        elif mode == "trials":
            # variance across trials of trial-averaged activity
            m = xs.mean(dim=1)  # (N,U)
            return m.var(dim=0, correction=correction)  # (U,)
        else:
            raise ValueError(f"Unknown mode: {mode}")

    def _summarize_distribution(x):
        """
        x: 1D numpy array
        """
        return {
            "n": int(x.size),
            "mean": float(np.mean(x)) if x.size else np.nan,
            "median": float(np.median(x)) if x.size else np.nan,
            "p10": float(np.percentile(x, 10)) if x.size else np.nan,
            "p90": float(np.percentile(x, 90)) if x.size else np.nan,
        }

    def _compare_variance_by_population(
            cls, model_var, target_var, population_indices, pop_names=None
    ):
        """
        model_var, target_var: (U,) numpy arrays (per-neuron variance)
        population_indices: list[list[int]] length n_pops
        Returns list[dict] per population
        """
        results = []
        n_pops = len(population_indices)
        if pop_names is None:
            pop_names = [f"pop_{i}" for i in range(n_pops)]

        for i, idx in enumerate(population_indices):
            idx = np.array(idx, dtype=int)
            mv = model_var[idx]
            tv = target_var[idx]

            out = {
                "pop": pop_names[i],
                **{f"model_{k}": v for k, v in cls._summarize_distribution(mv).items()},
                **{f"target_{k}": v for k, v in cls._summarize_distribution(tv).items()},
                "mean_ratio_model_over_target": float((np.mean(mv) + 1e-12) / (np.mean(tv) + 1e-12)),
            }

            # Distance between distributions (optional)
            if mv.size and tv.size:
                out["wasserstein_1d"] = float(cls.wasserstein_distance(mv, tv))
            else:
                out["wasserstein_1d"] = np.nan

            results.append(out)

        return results

    @classmethod
    def check_neurons_variance(
            cls,
            model,
            train_list,
            target_xs,  # (N,T,U) or (T,U) or (N,T,U)
            x0=None,
            variance_mode="time",
            correction=0,
            use_raw_if_available=True,
            pop_names=None,
            print_table=True,
    ):
        """
        target_xs should be the dataset neuron-level activity you want to compare against.
        If you only have population averages in the dataset, you can't compare per-neuron variance
        unless you have neuron-level traces or you define a proxy.
        """
        device = next(model.parameters()).device

        # Run model
        xs_model = cls.run_model_get_xs(model, train_list, x0=x0, use_raw_if_available=use_raw_if_available)  # (N,T,U)

        # Load target xs
        xs_target = cls._as_tensor(target_xs, device)
        if xs_target.ndim == 2:
            xs_target = xs_target[None, :, :].repeat(xs_model.shape[0], 1, 1)  # broadcast to (N,T,U)
        if xs_target.shape != xs_model.shape:
            raise ValueError(f"Shape mismatch: model xs {tuple(xs_model.shape)} vs target xs {tuple(xs_target.shape)}")

        # Per-neuron variance vectors
        v_model = cls.per_neuron_variance(xs_model, mode=variance_mode, correction=correction).detach().cpu().numpy()
        v_target = cls.per_neuron_variance(xs_target, mode=variance_mode, correction=correction).detach().cpu().numpy()

        # Compare by population
        results = cls._compare_variance_by_population(
            v_model, v_target, model.population_indices, pop_names=pop_names
        )

        if print_table:
            # simple print
            cols = [
                "pop",
                "model_mean", "target_mean", "mean_ratio_model_over_target",
                "model_median", "target_median",
                "model_p10", "model_p90", "target_p10", "target_p90",
                "wasserstein_1d",
            ]
            print("\t".join(cols))
            for r in results:
                print("\t".join(str(r.get(c, "")) for c in cols))

        return results, v_model, v_target

    @staticmethod
    def plot_response(model, t, input_signal, xpos, ypos, t_exp=None, output_signal=None, x0=None,
                      fig=None, plot_title_label="", show_xaxis=True, show_yaxis=True, show_xs=False,
                      palette=RNNDSStyle.palette["neurons_4"], plot_size=RNNDSStyle.plot_size_big * 0.4,
                      padding=RNNDSStyle.padding/2, compute_tau=True, time_structure=ConfigurationRNN.time_structure_simulation_test):
        # Draw network response to low step function (used in training)
        x0 = torch.zeros(model.n_units) if x0 is None else x0
        with torch.no_grad():
            inputs = torch.tensor(input_signal, dtype=torch.float32)
            xs, y_pred = model.forward(x0, inputs)

        xs = np.squeeze(xs.detach().numpy())
        y_pred = np.squeeze(y_pred.detach().numpy())

        if compute_tau:
            print(f"\n{plot_title_label}")
            for i_pop in range(y_pred.shape[1]):
                if i_pop in [0, 1, 3, 4, 5, 7]:
                    tau_rise, tau_decay, _ = DSService.fit_tau_rise_decay(y_pred[:, i_pop], ConfigurationRNN.dt_simulation, time_structure["rest_start"], time_structure["rest_start"] + time_structure["stimulus"])
                    print(f"Population {i_pop}")
                    print(f"MODEL | tau_rise: {tau_rise} | tau_decay: {tau_decay}")
                    if output_signal is not None:
                        tau_rise, tau_decay, _ = DSService.fit_tau_rise_decay(output_signal[:, i_pop], ConfigurationRNN.dt_data, time_structure["rest_start"], time_structure["rest_start"] + time_structure["stimulus"])
                        print(f"DATA | tau_rise: {tau_rise} | tau_decay: {tau_decay}")
                    # ##### DEBUG START
                    # plt.figure()
                    # plt.title(f"Population {i_pop}")
                    # plt.plot(DSService._exp_rise(downsample_time_list, _["popt_rise"][0], _["popt_rise"][1], _["popt_rise"][2]), color="k", linestyle="--")
                    # plt.plot(output_signal[int(duration_rest_start/dt_data):int((duration_rest_start+duration_stimulus)/dt_data), i_pop], color="k")
                    # plt.show()
                    # ##### DEBUG END
        if fig is None:
            fig = Figure()

        ymin = 0
        ymax = 2
        for side in range(2):
            plot_title = f"{plot_title_label}" if side == 0 else ""
            plot_title = plot_title
            offset_index = side * 4
            offset_hemisphere = side * (model.nA + model.nB + model.nC + model.nD)
            plot_input = fig.create_plot(
                plot_title=plot_title,
                xpos=xpos, ypos=ypos, plot_height=plot_size/10,
                plot_width=plot_size,
                xmin=0, xmax=time_structure["duration"], xticks=None,
                ymin=0, ymax=1, yticks=[0, 1] if show_yaxis and side == 0 else None,
                yl="Input" if show_yaxis and side == 0 else None)
            plot_response = fig.create_plot(
                # plot_title="\nActivity L" if side == 0 else plot_title + "\nActivity R",
                xpos=xpos, ypos=ypos - plot_size - padding/3, plot_height=plot_size,
                plot_width=plot_size,
                xmin=0, xmax=time_structure["duration"], xl="Time (s)" if show_xaxis else None,
                xticks=[time_structure["rest_start"], time_structure["rest_start"] + time_structure["stimulus"]] if show_xaxis else None,
                ymin=ymin, ymax=ymax, yticks=[ymin, ymax] if show_yaxis and side == 0 else None,
                yl="Activity" if show_yaxis and side == 0 else None)
            # input_signal_vis = input_signal[:, offset_hemisphere] / (torch.max(input_signal)-torch.min(input_signal)) * (ymax-ymin) # Normalize to ymin-ymax range
            plot_input.draw_line(t, input_signal[:, offset_hemisphere], lc="k")
            if output_signal is not None:
                if t_exp is None:
                    t_exp = t
                plot_response.draw_line(t_exp, output_signal[:, 0+offset_index], lc=palette[0])
                plot_response.draw_line(t_exp, output_signal[:, 1+offset_index], lc=palette[1])
                plot_response.draw_line(t_exp, output_signal[:, 2+offset_index], lc=palette[2])
                plot_response.draw_line(t_exp, output_signal[:, 3+offset_index], lc=palette[3])
            if show_xs:
                plot_response.draw_line(t, xs[:, offset_hemisphere:offset_hemisphere+model.nA], lc=palette[0], lw=0.1, line_dashes=(1, 2))
                plot_response.draw_line(t, xs[:, offset_hemisphere+model.nA:offset_hemisphere+model.nA + model.nB], lc=palette[1], lw=0.1, line_dashes=(1, 2))
                plot_response.draw_line(t, xs[:, offset_hemisphere+model.nA + model.nB:offset_hemisphere+model.nA + model.nB + model.nC], lc=palette[2], lw=0.1, line_dashes=(1, 2))
                plot_response.draw_line(t, xs[:, offset_hemisphere+model.nA + model.nB + model.nC:offset_hemisphere+model.nA + model.nB + model.nC + model.nD],
                                              lc=palette[3], lw=0.1, line_dashes=(1, 2))
            plot_response.draw_line(t, y_pred[:, 0+offset_index], lc=palette[0], line_dashes=(1, 2))
            plot_response.draw_line(t, y_pred[:, 1+offset_index], lc=palette[1], line_dashes=(1, 2))
            plot_response.draw_line(t, y_pred[:, 2+offset_index], lc=palette[2], line_dashes=(1, 2))
            plot_response.draw_line(t, y_pred[:, 3+offset_index], lc=palette[3], line_dashes=(1, 2))
            xpos += plot_size + padding

        res = {"fig": fig,
               "xpos": xpos,
               "ypos": ypos,
               "xs": xs,
               "y_pred": y_pred}
        return res

    @staticmethod
    def plot_response_by_cell(model, t, input_signal, xpos, ypos, ct_list=ConfigurationRNN.cell_label_list, t_exp=None, output_signal=None, x0=None,
                      fig=None, plot_title_label="", show_xaxis=True, show_yaxis=True, show_xs=False,
                      palette=RNNDSStyle.palette["neurons_4"], plot_size=RNNDSStyle.plot_size_big * 0.4,
                      padding=RNNDSStyle.padding / 2, compute_tau=True,
                      time_structure=ConfigurationRNN.time_structure_simulation_test):
        # Draw network response to low step function (used in training)
        x0 = torch.zeros(model.n_units) if x0 is None else x0
        with torch.no_grad():
            inputs = torch.tensor(input_signal, dtype=torch.float32)
            xs, y_pred = model.forward(x0, inputs)

        if inputs.dim() < 3:
            range_input = range(1)
        else:
            range_input = range(inputs.shape[0])

        if fig is None:
            fig = Figure()

        if output_signal is None:
            data_range = range(1)
        else:
            data_range = range(2)
            if len(output_signal.shape) == 2:
                output_signal = torch.unsqueeze(torch.tensor(output_signal, dtype=torch.float32), 0)

        if compute_tau:
            print(f"\n{plot_title_label}")
            for i_ct in range(len(ct_list)):
                for i_input in range_input:
                    if i_ct in [0, 1, 3, 4, 5, 7]:
                        tau_rise, tau_decay, _ = DSService.fit_tau_rise_decay(y_pred[i_input, :, i_ct],
                                                                              ConfigurationRNN.dt_simulation,
                                                                              time_structure["rest_start"],
                                                                              time_structure["rest_start"] +
                                                                              time_structure["stimulus"])
                        print(f"Population {i_ct}")
                        print(f"MODEL | input {i_input} | tau_rise: {tau_rise} | tau_decay: {tau_decay}")
                        if output_signal is not None:
                            tau_rise, tau_decay, _ = DSService.fit_tau_rise_decay(
                                output_signal[i_input, :, i_ct],
                                ConfigurationRNN.dt_data,
                                time_structure["rest_start"],
                                time_structure["rest_start"] +
                                time_structure["stimulus"])
                            print(f"DATA | input {i_input} | tau_rise: {tau_rise} | tau_decay: {tau_decay}")

        ymin = 0
        ymax = 2
        ypos_start_here = ypos
        plot_height_input = plot_size / 10
        for i_ct, ct in enumerate(ct_list):
            for i_side, side in enumerate(ConfigurationRNN.side_list):
                offset_hemisphere = i_side * (model.nA + model.nB + model.nC + model.nD)
                plot_title = f"{ct['label']}" if i_side == 0 else None
                plot_input = fig.create_plot(
                    plot_title=plot_title,
                    xpos=xpos, ypos=ypos - plot_height_input * 3 * i_side, plot_height=plot_height_input,
                    plot_width=plot_size,
                    xmin=0, xmax=time_structure["duration"], xticks=None,
                    ymin=0, ymax=1, yticks=[0, 1] if show_yaxis and side == 0 else None,
                    yl=f"Input {side}" if i_ct == 0 else None)
                for i_input in range_input:
                    alpha_here = 0.3 + (0.7 * i_input / len(range_input)) if len(range_input) > 1 else 1
                    plot_input.draw_line(t, input_signal[i_input, :, offset_hemisphere], lc="k", alpha=alpha_here)
            for i_data in data_range:
                plot_response = fig.create_plot(
                    # plot_title="\nActivity L" if side == 0 else plot_title + "\nActivity R",
                    xpos=xpos, ypos=ypos - plot_size - padding , plot_height=plot_size,
                    plot_width=plot_size,
                    xmin=0, xmax=time_structure["duration"], xl="Time (s)" if show_xaxis and i_data == len(data_range)-1 else None,
                    xticks=[time_structure["rest_start"],
                            time_structure["rest_start"] + time_structure["stimulus"]] if show_xaxis and i_data==len(data_range)-1 else None,
                    ymin=ymin, ymax=ymax, yticks=[ymin, ymax] if show_yaxis and i_ct == 0 else None,
                    yl="Activity" if show_yaxis and side == 0 else None)

                for i_input in range_input:
                    alpha_here = 0.3 + (0.7 * i_input / len(range_input)) if len(range_input) > 1 else 1
                    for side in range(2):  # there are 2 hemispheres
                        offset_index = side * 4  # harcoded for 4 populations here
                        offset_hemisphere = side * (model.nA + model.nB + model.nC + model.nD)

                        line_dashes = None if side == 0 else (1, 2)
                        if i_data == 0:
                            if t_exp is None:
                                t_exp = t
                            plot_response.draw_line(t_exp, output_signal[i_input, :, ct[f"index{len(ct_list)}"] + offset_index], lc=palette[ct[f"index{len(ct_list)}"]], line_dashes=line_dashes, alpha=alpha_here)
                        else:
                            plot_response.draw_line(t, y_pred[i_input, :, ct[f"index{len(ct_list)}"] + offset_index], lc=palette[ct[f"index{len(ct_list)}"]], line_dashes=line_dashes, alpha=alpha_here)

                ypos -= plot_size + padding / 3
            xpos += plot_size + padding
            ypos = ypos_start_here

        res = {"fig": fig,
               "xpos": xpos,
               "ypos": ypos,
               "xs": xs,
               "y_pred": y_pred}
        return res
