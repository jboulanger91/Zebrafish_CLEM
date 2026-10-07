import pickle
import random
import torch
import numpy as np

from torch import nn
from math import comb
from pathlib import Path
from matplotlib.colors import SymLogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable


# Manually add root path for imports to improve interoperability
import sys;
sys.path.insert(0, "..")

from figures.style import RNNDSStyle
from utils.config import ConfigurationRNN, ConfigurationNeural
from utils.services.ds_service import DSService
from utils.math.operators import nanstd
from utils.figure_helper import Figure


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

    @classmethod
    def plot_response_by_cell(cls, model_list, t, input_signal, xpos, ypos, ct_list=ConfigurationRNN.cell_label_list,
                              t_exp=None, output_signal_array=None, x0=None,
                      fig=None, plot_title_label="", show_xaxis=True, show_yaxis=True, yrange=(0, 1), show_xs=False,
                      palette=RNNDSStyle.palette["neurons_4"], plot_size=RNNDSStyle.plot_size_big * 0.4,
                      padding=RNNDSStyle.padding / 2, compute_tau=False,
                      time_structure=ConfigurationRNN.time_structure_simulation_test, compute_performance_method="pearson"):
        # Loop over model, simulate them and extract mean and SEM of activity
        if torch.is_tensor(model_list):
            model_list = [model_list]
        xs_list = []
        y_pred_list = []
        for model in model_list:
            x0 = torch.zeros(model.n_units) if x0 is None else x0
            with torch.no_grad():
                inputs = torch.tensor(input_signal, dtype=torch.float32)
                xs, y_pred = model.forward(x0, inputs, filter_xs=True)
                xs_list.append(xs)
                y_pred_list.append(y_pred)

        xs_array = torch.stack(xs_list, dim=0)
        xs_mean = torch.nanmean(xs_array, dim=0)
        y_pred_array = torch.stack(y_pred_list, dim=0)
        y_pred_mean = torch.nanmean(y_pred_array, dim=0)
        y_pred_std = nanstd(y_pred_array, dim=0)

        # Draw network response to low step function (used in training)
        if inputs.dim() < 3:
            range_input = range(1)
        else:
            range_input = range(inputs.shape[0])

        if fig is None:
            fig = Figure()

        if output_signal_array is None:
            data_range = range(1)
        else:
            data_range = range(2)

            # output_signal_array is processed as an array with dimensions: [R, I, T, C]
            # where R is the number of recordings for cell c at time t given stimulation i
            if len(output_signal_array.shape) == 2:
                output_signal_array = torch.unsqueeze(torch.tensor(output_signal_array, dtype=torch.float32), 0)
            if len(output_signal_array.shape) == 3:
                output_signal_array = torch.unsqueeze(torch.tensor(output_signal_array, dtype=torch.float32), 0)

            output_signal_mean = torch.nanmean(output_signal_array, axis=0)
            output_signal_std = np.nanstd(output_signal_array, axis=0)

        if t_exp is None:
            t_exp = t
        if compute_tau:
            print(f"\n{plot_title_label}")
            stimulus_window_data_index_list = np.argwhere(np.logical_and(t_exp>=time_structure["rest_start"],
                                                                    t_exp<time_structure["rest_start"] + time_structure["stimulus"]/2))
            offest_time_pop3 = 10  # seconds
            stimulus_window_data_pop3_index_list = np.argwhere(np.logical_and(t_exp>=time_structure["rest_start"] + offest_time_pop3,
                                                                    t_exp<time_structure["rest_start"] + offest_time_pop3 + time_structure["stimulus"]/2))
            stimulus_window_model_index_list = np.argwhere(np.logical_and(t>=time_structure["rest_start"],
                                                                    t<time_structure["rest_start"] + time_structure["stimulus"]/2))

            for i_ct in range(len(ct_list)):
                for i_input in range_input:
                    if i_ct in [0, 1, 3, 4, 5, 7]:
                        tau_rise = DSService.compute_time_rise(t[stimulus_window_model_index_list],
                                                               y_pred_list[0][i_input, stimulus_window_model_index_list, i_ct])
                        print(f"Population {i_ct}")
                        print(f"MODEL | input {i_input} | tau_rise: {tau_rise}")
                        if output_signal_array is not None:
                            if i_ct in [3, 7]:
                                window_index_list = stimulus_window_data_pop3_index_list
                            else:
                                window_index_list = stimulus_window_data_index_list
                            tau_rise = DSService.compute_time_rise(t_exp[window_index_list],
                                                                   output_signal_array[0, i_input, window_index_list, i_ct])
                            print(f"DATA | input {i_input} | tau_rise: {tau_rise}")

        # plot input signals
        plot_height_input = plot_size / 10
        for i_side, side in enumerate(ConfigurationRNN.side_list):
            offset_hemisphere = i_side * model.n_units_hemi
            plot_input = fig.create_plot(
                xpos=xpos + i_side * (plot_size + padding), ypos=ypos, plot_height=plot_height_input,
                plot_width=plot_size,
                xmin=0, xmax=time_structure["duration"], xticks=None,
                ymin=yrange[0], ymax=yrange[-1], yticks=yrange if show_yaxis and side == 0 else None,
                yl=f"Input {side}")
            for i_input in range_input:
                alpha_here = 0.3 + (0.7 * i_input / len(range_input)) if len(range_input) > 1 else 1
                plot_input.draw_line(t, input_signal[i_input, :, offset_hemisphere], lc="k", alpha=alpha_here)

        # plot activity traces by cell type and hemisphere side
        ymin = 0
        ymax = 2
        xpos_start_here = xpos
        for i_ct, ct in enumerate(ct_list):
            for i_data in data_range:
                plot_response = fig.create_plot(
                    # plot_title="\nActivity L" if side == 0 else plot_title + "\nActivity R",
                    xpos=xpos, ypos=ypos - plot_size - padding , plot_height=plot_size,
                    plot_width=plot_size,
                    xmin=0, xmax=time_structure["duration"],
                    ymin=ymin, ymax=ymax,
                    yl=ct["label"] if i_data == 0 else None,
                    vspans=[[time_structure["rest_start"], time_structure["rest_start"]+time_structure["stimulus"], "k", 0.1]]
                )
                
                # draw reference units
                if i_ct == len(ct_list)-1 and i_data == 0:
                    xdelta = time_structure["duration"] / 10
                    ydelta = (ymax - ymin) / 10
                    plot_response.draw_line((time_structure["duration"]-5-xdelta, time_structure["duration"]-xdelta), (ymin+ydelta*2)*np.ones(2), lc="k")
                    plot_response.draw_text(time_structure["duration"]-5-xdelta, ymin+ydelta, "5 s")
                    plot_response.draw_line(xdelta*2*np.ones(2), (ydelta, 0.5+ydelta), lc="k")
                    plot_response.draw_text(1, ymin+ydelta, r"$\Delta$F/F"+"\n0.5", textlabel_rotation=270)

                for i_input in range_input:
                    alpha_here = 0.3 + (0.7 * i_input / len(range_input)) if len(range_input) > 1 or i_input<len(range_input)-1 else 1
                    for side in range(2):  # there are 2 hemispheres
                        offset_index = side * 4  # harcoded for 4 populations here

                        line_dashes = None if side == 0 else (1, 2)
                        if i_data == 0:
                            plot_response.draw_line(t_exp, output_signal_mean[i_input, :, ct[f"index{len(ct_list)}"] + offset_index], lc=palette[ct[f"index{len(ct_list)}"]], line_dashes=line_dashes, alpha=alpha_here, yerr=output_signal_std[i_input, :, ct[f"index{len(ct_list)}"] + offset_index])
                        else:
                            plot_response.draw_line(t, y_pred_mean[i_input, :, ct[f"index{len(ct_list)}"] + offset_index], lc=palette[ct[f"index{len(ct_list)}"]], line_dashes=line_dashes, alpha=alpha_here, yerr=y_pred_std[i_input, :, ct[f"index{len(ct_list)}"]])

                xpos += plot_size + padding
            ypos -= plot_size + padding / 3
            xpos = xpos_start_here

        # Compute amplitude-independent performance
        performance = cls.compute_performance(output_signal_mean, t_exp, y_pred_mean, t, compute_performance_method)
        print(f"Performance in reproducing data | {compute_performance_method}: {performance}")

        res = {"fig": fig,
               "xpos": xpos,
               "ypos": ypos,
               "xs": xs_mean,
               "y_pred": y_pred_mean,
               "y_pred_std": y_pred_std,
               "performance": performance}
        return res

    @classmethod
    def compute_performance(cls, output_signal_mean, t_exp, y_pred_mean, t_sim, compute_performance_method=None):
        if compute_performance_method == None:
            compute_performance_method = "pearson"

        if compute_performance_method in ["corr", "pearson"]:
            perf_f = DSService.pearson_correlation
        elif compute_performance_method in ["acf"]:
            perf_f = DSService.acf_distance
        elif compute_performance_method in ["psd", "jsd"]:
            perf_f = DSService.jsd_psd
        else:
            raise Exception(f"compute_performance_method {compute_performance_method} is not supported")

        if len(output_signal_mean.size()) == 3:
            output_signal_mean = torch.unsqueeze(output_signal_mean, 0)
        if len(y_pred_mean.size()) == 3:
            y_pred_mean = torch.unsqueeze(y_pred_mean, 0)

        performance = 0
        n_contributions = output_signal_mean.size()[0] * output_signal_mean.size()[1] * output_signal_mean.size()[3]
        for i_model in range(output_signal_mean.size()[0]):
            for i_input in range(output_signal_mean.size()[1]):
                for i_ct in range(output_signal_mean.size()[3]):
                    performance += np.abs(perf_f(output_signal_mean[i_model, i_input, :, i_ct], t_exp, y_pred_mean[i_model, i_input, :, i_ct], t_sim)[0]) / n_contributions

        return performance

    @staticmethod
    def plot_connectivity(W, U=None, neuron_identity_array=None, grid_pop=None, fig=None, xpos=RNNDSStyle.xpos_start, ypos=RNNDSStyle.ypos_start,
                          plot_size_matrix=RNNDSStyle.plot_size_big * 1.2, padding=RNNDSStyle.padding, value_lim=[-1, 1], value_lim_U=None, plot_title="W", plot_title_U="U",
                          cmap='PiYG', cmap_U=None, cmap_pop=RNNDSStyle.cmap_list["neurons_4"], show_colorbar=True, plot_size_vector_neurons=0.025, show_text=False, logscale=True):

        n_neurons = W.shape[0]
        # Draw input vector U after training
        plot_size_vector = plot_size_matrix / n_neurons

        if U is not None:
            if value_lim_U is None:
                value_lim_U = value_lim
            if cmap_U is None:
                cmap_U = cmap
            plot_U = fig.create_plot(plot_title=plot_title_U,
                                     xpos=xpos, ypos=ypos, plot_height=plot_size_matrix,
                                     plot_width=plot_size_vector,
                                     xmin=-0.5, xmax=0.5,  # xticklabels_rotation=90,
                                     # xticks=np.arange(n_neurons),
                                     ymin=-0.5, ymax=n_neurons - 0.5)

            xpos += plot_size_vector + padding
            im = plot_U.draw_image(U, (-0.5, 0.5, n_neurons - 0.5, -0.5),
                                   colormap=cmap_U, zmin=value_lim_U[0], zmax=value_lim_U[-1],
                                   image_interpolation=None)

            if show_text:
                for i in range(len(U)):
                    plot_U.ax.text(0, i, f"{U[-i-1, 0]:.02f}", ha="center", va="center", color="k")

        scale_width = 1 if show_colorbar else 1

        # Draw connectivity matrix W
        plot_W = fig.create_plot(plot_title=plot_title,
                                 xpos=xpos, ypos=ypos, plot_height=plot_size_matrix,
                                 plot_width=plot_size_matrix * scale_width,
                                 xmin=-0.5, xmax=n_neurons - 0.5,  # xticklabels_rotation=90,
                                 # xticks=np.arange(n_neurons),
                                 ymin=-0.5, ymax=n_neurons - 0.5)

        if neuron_identity_array is not None:
            # Draw neuron identity vectors around W
            plot_ni_c = fig.create_plot(xpos=xpos - plot_size_vector_neurons, ypos=ypos, plot_height=plot_size_matrix,
                                        plot_width=plot_size_vector_neurons,
                                        xmin=-0.5, xmax=0.5,
                                        ymin=-0.5, ymax=n_neurons - 0.5)
            im = plot_ni_c.draw_image(neuron_identity_array, (-0.5, 0.5, n_neurons - 0.5, -0.5),
                                      colormap=cmap_pop, zmin=0, zmax=1, image_interpolation=None)

            plot_ni_r = fig.create_plot(xpos=xpos, ypos=ypos + plot_size_matrix, plot_height=plot_size_vector_neurons,
                                        plot_width=plot_size_matrix,
                                        xmin=-0.5, xmax=n_neurons - 0.5,
                                        ymin=-0.5, ymax=0.5)
            im = plot_ni_r.draw_image(neuron_identity_array.T, (-0.5, n_neurons - 0.5, -0.5, 0.5),
                                      colormap=cmap_pop, zmin=0, zmax=1, image_interpolation=None)

        x_ = np.arange(n_neurons)
        x = np.tile(x_, (n_neurons, 1))
        y = x.T
        if logscale:
            norm = SymLogNorm(linthresh=0.03, linscale=1.0, vmin=-1, vmax=1, base=10)
        else:
            norm = None
        im = plot_W.draw_image(W, (-0.5, n_neurons - 0.5, n_neurons - 0.5, -0.5), norm_colormap=norm,
                               colormap=cmap, zmin=value_lim[0], zmax=value_lim[-1], image_interpolation=None)

        if show_text:
            for i in range(len(neuron_identity_array)):
                for j in range(len(neuron_identity_array)):
                    plot_W.ax.text(j, i, f"{W[-i-1, j]:.02f}", ha="center", va="center", color="k")

        if grid_pop is not None:
            plot_W_grid = fig.create_plot(xpos=xpos, ypos=ypos, plot_height=plot_size_matrix, plot_width=plot_size_matrix,
                                          xmin=-0.5, xmax=n_neurons - 0.5, ymin=-0.5, ymax=n_neurons - 0.5,
                                          helper_lines_lc="white",
                                          hlines=n_neurons - grid_pop - 0.5,
                                          vlines=grid_pop - 0.5)
        if show_colorbar:
            plot_bar = fig.create_plot(xpos=xpos + plot_size_matrix + 5*plot_size_vector_neurons, ypos=ypos,
                                       plot_height=plot_size_matrix,
                                       plot_width=plot_size_vector_neurons * 10,
                                       xmin=-0.5, xmax=0.5,
                                       ymin=-0.5, ymax=n_neurons - 0.5)
            plot_bar.figure.fig.colorbar(im, cax=plot_bar.ax, orientation='vertical',
                                       ticks=[value_lim[0], np.mean(value_lim), value_lim[-1]])
        xpos += plot_size_matrix + padding * 1.5

        return fig, xpos, ypos

    @staticmethod
    def extract_custom_attrs(model):
        """
        Grab everything in __dict__ that is NOT an nn.Module, Parameter,
        or private attribute — i.e. your custom hyperparams/config.
        """
        skip_types = (torch.nn.Module, torch.nn.Parameter)
        attrs = {}
        for k, v in model.__dict__.items():
            if k.startswith("_"):  # private / internal PyTorch bookkeeping
                continue
            if isinstance(v, skip_types):  # submodules handled by state_dict
                continue
            try:
                # test serializability with torch.save
                torch.save(v, Path("/dev/null") if Path("/dev/null").exists()
                else Path("nul"))  # Windows: "nul", Linux: /dev/null
            except Exception:
                print(f"  Skipping non-serializable attribute: {k} ({type(v).__name__})")
                continue
            attrs[k] = v
        return attrs

    @staticmethod
    def count_loops_over_quadruplets(A):
        """
        Count loops of length 1 (self-loop), 2 (mutual edge pair), or 3
        (directed triangle) summed across all possible 4-node quadruplets
        of a directed 0/1 connectivity matrix A (N x N), without ever
        enumerating quadruplets explicitly.

        Returns a dict with raw elementary loop counts and the total
        weighted by how many quadruplets each loop participates in.
        """
        A = np.array(A, dtype=np.int64)
        N = A.shape[0]
        if N < 4:
            raise ValueError("Need at least 4 nodes to form quadruplets.")

        diag = np.diag(A).copy()
        A0 = A.copy()
        np.fill_diagonal(A0, 0)  # off-diagonal-only graph

        # Elementary loop counts (each computed once, not per-quadruplet)
        L1 = int(diag.sum())  # self-loops
        L2 = int(np.sum(np.triu(A0 * A0.T, k=1)))  # mutual (2-node) loops
        L3 = int(round(np.trace(A0 @ A0 @ A0) / 3))  # directed 3-cycles

        # Number of quadruplets containing a fixed 1-, 2-, or 3-node loop
        w1 = comb(N - 1, 3)
        w2 = comb(N - 2, 2)
        w3 = comb(N - 3, 1)
        total_quadruplets = comb(N, 4)

        incidence_total = L1 * w1 + L2 * w2 + L3 * w3

        return {
            "self_loops": L1,
            "mutual_pairs": L2,
            "directed_triangles": L3,
            "weight_per_self_loop": w1,
            "weight_per_pair": w2,
            "weight_per_triangle": w3,
            "total_distinct_loops": L1 + L2 + L3,
            "quadruplet_incidence_total": incidence_total,  # sum over all quadruplets of loops-per-quadruplet
            "total_quadruplets": total_quadruplets,
            "avg_loops_per_quadruplet": incidence_total / total_quadruplets,
        }

    @staticmethod
    def check_connectivity_selected_neurons(A, selected_indices):
        """
        For each selected neuron (row index) in a directed connectivity matrix A,
        where A[i, j] == 1 means neuron i projects onto neuron j:

          - presynaptic partners: neurons that project onto the selected neuron
          - postsynaptic partners: neurons the selected neuron projects onto (1st order)
          - second-order postsynaptic partners: postsynaptic partners of the 1st-order
            postsynaptic partners (repetitions kept, since a neuron can be reached
            via multiple 1st-order partners)

        Returns a dict keyed by neuron index, each containing the three identity
        lists plus counts of second-order partners and how many of those (with
        repetition) also belong to the presynaptic or 1st-order postsynaptic sets.
        """
        A = np.asarray(A)
        _results = {}

        for i in selected_indices:
            presynaptic = np.where(A[:, i])[0].tolist()  # j -> i
            postsynaptic = np.where(A[i, :])[0].tolist()  # i -> j

            second_order_postsynaptic = []
            for j in postsynaptic:
                second_order_postsynaptic.extend(np.where(A[j, :])[0].tolist())

            other_groups = set(presynaptic) | set(selected_indices) | set(postsynaptic)
            second_order_group = set(second_order_postsynaptic)
            overlap_count = sum(1 for x in second_order_postsynaptic if x in other_groups)
            rediscovered_group = second_order_group & other_groups

            _results[i] = {
                "presynaptic": presynaptic,
                "postsynaptic": postsynaptic,
                "second_order_postsynaptic": second_order_postsynaptic,
                "n_second_order_postsynaptic": len(second_order_postsynaptic),
                "n_second_order_overlap_with_other_groups": overlap_count,
                "rediscovered_group": rediscovered_group
            }
        # results["all"] = \
        results = {
            "presynaptic": set().union(*[_results[i]["presynaptic"] for i in _results.keys()]),
            "n_presynaptic": np.sum([len(_results[i]["presynaptic"]) for i in _results.keys()]),
            "seed": set(selected_indices),
            "n_seed": len(selected_indices),
            "postsynaptic": set().union(*[_results[i]["postsynaptic"] for i in _results.keys()]),
            "n_postsynaptic": np.sum([len(_results[i]["postsynaptic"]) for i in _results.keys()]),
            "second_order_postsynaptic": set().union(
                *[_results[i]["second_order_postsynaptic"] for i in _results.keys()]),
            "n_second_order_postsynaptic": np.sum(
                [_results[i]["n_second_order_postsynaptic"] for i in _results.keys()]),
            "n_second_order_overlap_with_other_groups": np.sum(
                [_results[i]["n_second_order_overlap_with_other_groups"] for i in _results.keys()]),
            "rediscovered_group": set().union(*[_results[i]["rediscovered_group"] for i in _results.keys()])
        }

        return results

    @staticmethod
    def compute_sparsity_cell_sides(W_binary, dict_neurons: dict, cell: str):
        sparsity_cell = np.zeros(2)
        for i_side, side in enumerate(ConfigurationRNN.side_list):
            W_binary_side_cell = W_binary[np.ix_(dict_neurons["neurons"][side][cell]["idx_list"],
                                                 dict_neurons["neurons"][side][cell]["idx_list"])]
            sparsity_side_cell = np.sum(W_binary_side_cell) / np.size(W_binary_side_cell)
            sparsity_cell[i_side] = sparsity_side_cell
        return sparsity_cell

    @staticmethod
    def compute_sparsity(W, pop_idx_list):
        sparsity = np.zeros((len(pop_idx_list), len(pop_idx_list)))
        W_binary = np.abs(np.sign(W))
        for i, pop_idx in enumerate(pop_idx_list):
            for i_, pop_idx_ in enumerate(pop_idx_list):
                block = W_binary[np.ix_(pop_idx, pop_idx_)]
                sparsity[i, i_] = np.sum(block) / np.size(block)
        return sparsity

    @classmethod
    def compute_rediscovery_rate(cls, W, pop_idx_list):
        rediscovery_rate = np.zeros(len(pop_idx_list))
        W_binary = np.abs(np.sign(W))
        for i, pop_idx in enumerate(pop_idx_list):
            recurrency_pop = cls.check_connectivity_selected_neurons(W_binary, pop_idx)
            rediscovery_rate[i] = len(recurrency_pop["rediscovered_group"]) / len(
                recurrency_pop["presynaptic"] | recurrency_pop["seed"] | recurrency_pop["postsynaptic"])
        return rediscovery_rate
