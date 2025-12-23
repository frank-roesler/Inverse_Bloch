from time import time
import torch
import numpy as np
import os
import datetime
import json
from utils_bloch import blochsim_CK_batch
from utils_bloch.old.blochsim_CK import blochsim_CK
from matplotlib.collections import LineCollection
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.cm as cm
import torch.nn.functional as F


class TrainLogger:
    def __init__(self, targets, tconfig, bconfig, mconfig, sconfig):
        self.log = {
            "tconfig": tconfig,
            "bconfig": bconfig,
            "mconfig": mconfig,
            "sconfig": sconfig,
            "targets": targets,
        }
        self.log["step"] = tconfig.start_step
        self.start_logging = tconfig.start_logging
        self.best_loss = np.inf
        self.loss_weights = tconfig.loss_weights
        self.export_loc = os.path.join("results", datetime.datetime.now().strftime("%Y-%m-%d_%H-%M"))
        self.new_optimum = False
        self.t_B1_legacy = self.log["bconfig"].fixed_inputs["t_B1_legacy"].to(targets["target_xy"].device)

    def log_step(self, step, losses, model, optimizer):
        self.log["step"] = step
        self.log["tconfig"].start_step = step
        self.log["losses"] = losses
        self.log["model"] = model
        self.log["optimizer"] = optimizer
        with torch.no_grad():
            pulse, gradient = model(self.t_B1_legacy)
        self.log["pulse"] = pulse.detach().cpu()
        self.log["gradient"] = gradient.detach().cpu()
        self.save(losses)

    def save(self, losses):
        if not losses["total"][-1] < 0.99 * self.best_loss:
            self.new_optimum = False
            return
        self.best_loss = losses["total"][-1]
        if self.log["step"] <= self.start_logging:
            self.new_optimum = False
            return
        os.makedirs(self.export_loc, exist_ok=True)
        torch.save(self.log, os.path.join(self.export_loc, "train_log.pt"))
        self.export_json()
        print(f"Training log saved to {self.export_loc}")
        print("-" * 50)
        self.new_optimum = True

    def export_json(self):
        model_name = self.log["model"].name
        export_path = os.path.join(self.export_loc, f"sms_nn_{model_name}.json")
        dur = 1000 * self.log["bconfig"].fixed_inputs["t_B1"][-1].item()

        data = {
            "id": "sms_nn_150425_MLP_square2",
            "set": {
                "maxB1": torch.max(torch.abs(self.log["pulse"])).item(),
                "dur": dur,
                "pts": len(self.log["bconfig"].fixed_inputs["t_B1"]),
                "amplInt": None,
                "refocFract": None,
            },
            "conf": {"slthick": 0.02, "bs": 1.5, "tb": 4.8, "mb": 2},
            "rfPls": {
                "real": torch.real(self.log["pulse"]).squeeze().tolist(),
                "imag": torch.imag(self.log["pulse"]).squeeze().tolist() if self.log["pulse"].dtype == torch.complex64 else 0.0,
            },
        }
        with open(export_path, "w") as json_file:
            json.dump(data, json_file, indent=4)


class InfoScreen:
    def __init__(self, target_xy, where_slices_are, bloch_config, output_every=1, losses={}):
        self.t0 = time()
        self.t1 = time()
        self.output_every = output_every
        self.init_plots(losses)
        self.pos = bloch_config.fixed_inputs["pos"].cpu()
        self.t_B1 = bloch_config.fixed_inputs["t_B1"].cpu()
        self.target_xy = target_xy.sum(dim=-1).detach().cpu()
        self.target_z = torch.sqrt(1 - self.target_xy**2)
        self.where_slices_are = where_slices_are.detach().cpu().sum(dim=-1).bool().numpy()

    def init_plots(self, losses):
        self.fig = plt.figure(figsize=(14, 7), constrained_layout=False)
        spec = gridspec.GridSpec(2, 2, figure=self.fig)  # Create a 2x2 grid layout

        # First row: 3 plots spanning the entire width
        spec_top = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=spec[0, :])  # Split first row into 3 columns
        self.ax = [None] * 3
        self.ax[0] = self.fig.add_subplot(spec_top[0])
        self.ax[1] = self.fig.add_subplot(spec_top[1])
        self.ax[2] = self.fig.add_subplot(spec_top[2])
        # Second row: 2 equally wide plots
        self.ax_bottom_left = self.fig.add_subplot(spec[1, 0])
        self.ax_bottom_left.set_ylabel("M_z")
        self.ax_bottom_right = self.fig.add_subplot(spec[1, 1])
        self.ax_phase = self.ax_bottom_right.twinx()
        self.ax_phase.set_ylabel("Phase (rad)")
        self.ax_bottom_right.set_ylabel("|M_xy|")
        self.loss_lines = {}

        # Initialize plots for the first row
        self.pulse_real_plot = self.ax[0].plot([0], [1e-4], linewidth=0.8, label="Pulse real")[0]
        self.pulse_imag_plot = self.ax[0].plot([0], [1e-4], linewidth=0.8, label="Pulse imag")[0]
        self.pulse_abs_plot = self.ax[0].plot([0], [1e-4], linewidth=0.8, label="|Pulse|", linestyle="dotted")[0]
        self.grad_plot = self.ax[1].plot([0], [1], linewidth=0.8, label="Gradient")[0]
        cmap = cm.get_cmap("tab10", len(losses))
        for idx, loss_key in enumerate(losses.keys()):
            if loss_key == "boundary_vals_pulse":
                continue
            lw = 1.1 if loss_key == "total" else 0.7
            color = cmap(idx)
            (self.loss_lines[loss_key],) = self.ax[2].semilogy([0], [1e-1], linewidth=lw, label=loss_key, color=color)

        self.ax[2].legend(loc="lower left", bbox_to_anchor=(1.01, 0.5), borderaxespad=0, fontsize="x-small")
        self.fig.subplots_adjust(right=0.89)

        # Set titles and legends
        self.ax[0].set_title("Pulse [mT]")
        self.ax[1].set_title("Gradient [mT/m]")
        self.ax[2].set_title("Loss")
        self.ax_bottom_left.set_title("M_z")
        self.ax_bottom_right.set_title("M_xy")
        self.ax[0].legend(fontsize="small")
        self.ax[1].legend(fontsize="small")
        for handle in self.ax[2].get_legend().legend_handles:
            handle.set_linewidth(2)
        plt.show(block=False)

    def plot_info(self, step, losses, mz, mxy, pulse, gradient, trainlogger):
        """plots info curves during training"""
        if step % self.output_every == 0 or trainlogger.new_optimum:
            fmin = torch.min(self.pos).item()
            fmax = torch.max(self.pos).item()
            t = self.t_B1.detach().cpu().numpy()
            mz_plot = mz.detach().cpu().numpy()
            mxy_abs = np.abs(mxy.detach().cpu().numpy())
            pulse_real = np.real(pulse.detach().cpu().numpy())
            pulse_imag = np.imag(pulse.detach().cpu().numpy())

            pulse_abs = np.sqrt(pulse_real**2 + pulse_imag**2)
            gradient_for_plot = gradient.detach().cpu().numpy()
            phase = np.unwrap(np.angle(mxy.detach().cpu().numpy()), axis=-1)
            phasemin = np.min(phase[self.where_slices_are + mxy_abs > 0.5])
            phasemax = np.max(phase[self.where_slices_are + mxy_abs > 0.5])
            phasemin = phasemin - 0.5 * (phasemax - phasemin)
            phasemax = phasemax + 0.5 * (phasemax - phasemin)
            phase[~(self.where_slices_are + mxy_abs > 0.5)] = np.nan

            for collection in self.ax_bottom_left.collections:
                collection.remove()
            for collection in self.ax_bottom_right.collections:
                collection.remove()
            for collection in self.ax_phase.collections:
                collection.remove()

            self.grad_plot.set_xdata(t)
            self.pulse_real_plot.set_xdata(t)
            self.pulse_imag_plot.set_xdata(t)
            self.pulse_abs_plot.set_xdata(t)

            self.add_line_collection(self.ax_bottom_left, self.pos, mz_plot)
            self.add_line_collection(self.ax_bottom_right, self.pos, mxy_abs)
            self.add_line_collection(self.ax_phase, self.pos, phase, linestyle="dotted")
            self.add_line_collection(self.ax_bottom_left, self.pos, self.target_z, cmap="viridis")
            self.add_line_collection(self.ax_bottom_right, self.pos, self.target_xy, cmap="viridis")

            self.grad_plot.set_ydata(gradient_for_plot)
            self.pulse_real_plot.set_ydata(pulse_real)
            self.pulse_imag_plot.set_ydata(pulse_imag)
            self.pulse_abs_plot.set_ydata(pulse_abs)
            minLoss = np.inf
            for key, line in self.loss_lines.items():
                line.set_xdata(np.arange(len(losses[key])))
                line.set_ydata(losses[key])
                if key == "boundary_vals_pulse":
                    continue
                currentMin = 0.5 * np.min(losses[key][-2000:])
                if currentMin > 1e-6:
                    minLoss = min(minLoss, currentMin)
                else:
                    minLoss = 1e-6
            maxLoss = 2 * np.max(losses["total"][-2000:])

            self.ax_bottom_left.set_xlim(fmin, fmax)
            self.ax_bottom_right.set_xlim(fmin, fmax)
            self.ax_phase.set_xlim(fmin, fmax)
            self.ax[0].set_xlim(t[0], t[-1])
            self.ax[1].set_xlim(t[0], t[-1])
            self.ax[2].set_xlim(0, step + 1)
            self.ax_bottom_left.set_ylim(-0.1, 1.1)
            self.ax_bottom_right.set_ylim(-0.1, 1.1)
            self.ax_phase.set_ylim(phasemin, phasemax)
            self.ax[0].set_ylim((-np.max(pulse_abs), np.max(pulse_abs)))
            self.ax[1].set_ylim((-1.1 * np.max(np.abs(gradient_for_plot)).item(), 1.1 * np.max(np.abs(gradient_for_plot)).item()))
            self.ax[2].set_ylim((minLoss, maxLoss))

            self.fig.canvas.draw_idle()
            if trainlogger.new_optimum:
                os.makedirs(trainlogger.export_loc, exist_ok=True)
                plt.savefig(os.path.join(trainlogger.export_loc, "training.png"), dpi=300)
            self.fig.canvas.flush_events()

    def add_line_collection(self, ax, xdata, ydata, linestyle="solid", cmap="inferno"):
        line_list = [list(zip(xdata, ydata[b, :])) for b in range(ydata.shape[0])]
        self.add_line_collection_from_list(ax, line_list, linestyle, cmap)

    def add_line_collection_from_list(self, ax, line_list, linestyle, cmap="inferno"):
        cmap = cm.get_cmap(cmap, len(line_list) + 2)
        colors = [cmap(i + 1) for i in range(len(line_list) + 2)]
        line_collection = LineCollection(line_list, linewidths=0.7, colors=colors, linestyle=linestyle)
        ax.add_collection(line_collection)

    def print_info(self, step, loss, optimizer, best_loss):
        if not step % 10 == 0:
            return
        self.t1 = time() - self.t0
        self.t0 = time()
        print("Step: ", step)
        print(f"Loss: {loss['total']:.6f}")
        print(f"Best Loss: {best_loss:.6f}")
        for i, param_group in enumerate(optimizer.param_groups):
            print(f"Learning rate {i}: {param_group['lr']:.7f}")
        print(f"Time: {self.t1:.1f}")
        print("-" * 100)


class SliceProfileLoss(torch.nn.Module):
    def __init__(self, tconfig, bconfig, sconfig, target_xy, verbose=False):
        super(SliceProfileLoss, self).__init__()

        self.verbose = verbose
        self.metric = tconfig.loss_metric
        self.loss_weights = tconfig.loss_weights
        self.target_phase_offset = tconfig.phase_offset_in_rad
        self.scanner_params = sconfig.scanner_params
        self.n_slices = bconfig.n_slices
        self.posAx = bconfig.fixed_inputs["pos"].to(target_xy.device)
        self.pos_extent = self.posAx.max() - self.posAx.min()
        self.dz = torch.diff(self.posAx)
        self.delta_t = bconfig.fixed_inputs["dt_num"]

        self.active_losses = self.determine_active_losses()
        self.target_xy = target_xy.sum(dim=-1)
        self.target_z = torch.sqrt(1 - self.target_xy**2)
        self.slices_mask = target_xy > 1e-3
        self.mass_slice = self.compute_slice_masses()

    def determine_active_losses(self):
        active_losses = {}
        if self.loss_desired("loss_mxy"):
            active_losses["loss_mxy"] = self.compute_loss_mxy
        if self.loss_desired("loss_mz"):
            active_losses["loss_mz"] = self.compute_loss_mz
        if self.loss_desired("boundary_vals_pulse"):
            active_losses["boundary_vals_pulse"] = self.compute_boundary_loss
        if self.loss_desired("gradient_height_loss"):
            active_losses["gradient_height_loss"] = self.compute_gradient_height_loss
        if self.loss_desired("pulse_height_loss"):
            active_losses["pulse_height_loss"] = self.compute_pulse_height_loss
        if self.loss_desired("gradient_diff_loss"):
            active_losses["gradient_diff_loss"] = self.compute_gradient_diff_loss
        if self.loss_desired("phase_B0_diff"):
            active_losses["phase_B0_diff"] = self.compute_phase_B0_diff
        if self.loss_desired("phase_diff"):
            active_losses["phase_diff"] = self.compute_phase_diff_loss
        if self.loss_desired("phase_diff_var_loss"):
            active_losses["phase_diff_var_loss"] = self.compute_phase_diff_var_loss
        if self.loss_desired("phase_offset_loss"):
            active_losses["phase_offset_loss"] = self.compute_phase_offset_loss
        return active_losses

    def compute_phase_diff(self, phase):
        """Enforces linearity of phase on slices"""
        exp_phase = torch.exp(1j * phase)
        phase_diff = torch.gradient(exp_phase, spacing=(self.posAx,), dim=1)[0] / exp_phase
        return phase_diff.imag

    def compute_phase_mean_on_slices(self, phase):
        phase_means = torch.zeros((phase.shape[0], self.n_slices), device=phase.device)
        phase_on_slices = phase.unsqueeze(-1) * self.slices_mask
        for b in range(phase.shape[0]):
            for s in range(self.n_slices):
                phase_means[b, s] = torch.trapz(phase_on_slices[b, self.slices_mask[b, :, s], s], self.posAx[self.slices_mask[b, :, s]])
        phase_means = phase_means / self.mass_slice
        return phase_means

    def compute_slice_masses(self):
        """computes mass of each slice for normalization of phase means"""
        mass_slice = torch.zeros((self.slices_mask.shape[0], self.n_slices), device=self.slices_mask.device)
        for b in range(self.slices_mask.shape[0]):
            for s in range(self.n_slices):
                mass_slice[b, s] = torch.trapz(1.0 * self.slices_mask[b, self.slices_mask[b, :, s], s], self.posAx[self.slices_mask[b, :, s]])
        return mass_slice

    def compute_loss_mxy(self, z_profile, xy_profile, pulse, gradient, phase, phase_diff):
        """Computes approximation error between slice profile and target in L1 or L2 norm"""
        xy_profile_abs = torch.abs(xy_profile)
        if self.metric == "L2":
            integrand = (xy_profile_abs - self.target_xy) ** 2
        elif self.metric == "L1":
            integrand = torch.abs(xy_profile_abs - self.target_xy)
        else:
            raise ValueError("Invalid metric. Choose 'L2' or 'L1'.")
        loss_mxy = torch.trapz(integrand, self.posAx, dim=1).mean(dim=0) / self.pos_extent
        self.loss_mxy = loss_mxy
        return loss_mxy

    def compute_loss_mz(self, z_profile, xy_profile, pulse, gradient, phase, phase_diff):
        """Computes approximation error between slice profile and target in L1 or L2 norm"""
        if self.metric == "L2":
            integrand = (z_profile - self.target_z) ** 2
        elif self.metric == "L1":
            integrand = torch.abs(z_profile - self.target_z)
        else:
            raise ValueError("Invalid metric. Choose 'L2' or 'L1'.")
        loss_mz = torch.trapz(integrand, self.posAx, dim=1).mean(dim=0)
        return loss_mz / self.pos_extent

    def compute_boundary_loss(self, z_profile, xy_profile, pulse, gradient, phase, phase_diff):
        """Penalizes boundary values of the pulse at start and end"""
        boundary_vals_pulse = threshold_loss((torch.abs(pulse[0]) ** 2 + torch.abs(pulse[-1]) ** 2), 1e-11).mean(dim=0)
        return boundary_vals_pulse

    def compute_gradient_height_loss(self, z_profile, xy_profile, pulse, gradient, phase, phase_diff):
        """Penalizes maximum height of the gradient"""
        gradient_height_loss = threshold_loss(gradient, self.scanner_params["max_gradient"]).mean(dim=0)
        return gradient_height_loss

    def compute_gradient_diff_loss(self, z_profile, xy_profile, pulse, gradient, phase, phase_diff):
        """Penalizes large slope of the gradient (in time)"""
        if gradient.shape[0] > 1:
            gradient_diff_loss = threshold_loss(torch.diff(gradient.squeeze()), self.scanner_params["max_diff_gradient"] * self.delta_t * 1000).mean(dim=0)
        else:
            gradient_diff_loss = torch.zeros(1, device=gradient.device)
        return gradient_diff_loss

    def compute_pulse_height_loss(self, z_profile, xy_profile, pulse, gradient, phase, phase_diff):
        """Penalizes large pulse amplitude (to avoid SAR)"""
        pulse_height_loss = threshold_loss(pulse, self.scanner_params["max_pulse_amplitude"]).mean(dim=0)
        return pulse_height_loss

    def compute_phase_diff_var_loss(self, z_profile, xy_profile, pulse, gradient, phase, phase_diff):
        """Enforces equal slope pf phase in all slices"""
        phase_diff_var = torch.var(phase_diff[self.slices_mask.sum(dim=-1).bool()], dim=-1)
        phase_diff_var_actual = torch.var(phase_diff[xy_profile.abs() > 0.5], dim=-1)
        x = 5e2 * self.loss_mxy**2
        weight_factor = (x / (x + 1)) ** 2
        # print("WEIGHT FACTOR:", weight_factor.item())
        return weight_factor * phase_diff_var + (1 - weight_factor) * phase_diff_var_actual

    def compute_phase_B0_diff(self, z_profile, xy_profile, pulse, gradient, phase, phase_diff):
        """When training multiple B0 values at once, this loss penalizes large variation of the phase in B0 direction"""
        phases_slices = phase * self.slices_mask
        phase_B0_diff = phase.shape[0] * torch.mean(torch.diff(phases_slices, dim=0) ** 2)
        return phase_B0_diff

    def compute_phase_offset_loss(self, z_profile, xy_profile, pulse, gradient, phase, phase_diff):
        """Enforces phase offset of self.target_phase_offset between slices."""
        phase_means = self.compute_phase_mean_on_slices(torch_unwrap(phase))
        phase_offset_loss = torch.tan(0.5 * (torch.diff(phase_means, dim=-1) - self.target_phase_offset)) ** 2
        return phase_offset_loss.mean()

    def compute_phase_diff_loss(self, z_profile, xy_profile, pulse, gradient, phase, phase_diff):
        """Penalizes slope of phase on slices. Thereby enforces self-refocusoing"""
        phase_diff_loss = torch.trapz((phase_diff * self.slices_mask) ** 2, self.posAx, dim=-1) / self.mass_slice.sum(dim=-1)
        return phase_diff_loss.mean()

    def loss_desired(self, key):
        return key in self.loss_weights and self.loss_weights[key] > 0

    def forward(self, z_profile, xy_profile, pulse, gradient):
        step_losses = {}
        phase = torch.angle(xy_profile)
        phase_diff = self.compute_phase_diff(phase)

        for key, loss_fn in self.active_losses.items():
            loss_value = loss_fn(z_profile, xy_profile, pulse, gradient, phase, phase_diff)
            step_losses[key] = loss_value * self.loss_weights[key]

        if self.verbose:
            print("-" * 50)
            print("LOSSES:")
            for key, value in step_losses.items():
                print(f"{key}: {value.mean().item():.6f}")
            print("-" * 50)
        return step_losses


def get_device():
    device = torch.device("cpu") if torch.backends.mps.is_available() else torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print("Device:", device)
    return device


def move_to(tensor_list, device=torch.device("cpu")):
    out_list = []
    for tensor in tensor_list:
        tensor = tensor.to(device)
        out_list.append(tensor)
    return out_list


def pre_train(target_pulse, target_gradient, model, fixed_inputs, lr=1e-4, thr=1e-3, device=torch.device("cpu")):
    target_pulse = target_pulse.to(device)
    target_gradient = target_gradient.to(device)
    model, optimizer, scheduler, _ = init_training(model, lr=lr, device=device)
    t_B1 = fixed_inputs["t_B1"].to(device)
    loss = torch.inf
    step = 0
    while loss > thr:
        step += 1
        pulse, gradient = model(t_B1)

        loss_pulse = torch.mean(torch.abs(pulse - target_pulse) ** 2)
        loss_gradient = torch.mean((gradient - target_gradient) ** 2)
        loss = loss_pulse + loss_gradient / model.gradient_scale

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step(loss.item())
        if step % 1000 == 0:
            print(f"step: {step}, Loss: {loss.item():.6f}, lr: {optimizer.param_groups[0]['lr']}")
    # plt.figure()
    # plt.plot(torch.real(pulse).detach().cpu().numpy(), label="pulse real")
    # plt.plot(torch.real(target_pulse).detach().cpu().numpy(), label="target pulse real")
    # plt.figure()
    # plt.plot(torch.imag(pulse).detach().cpu().numpy(), label="pulse imag")
    # plt.plot(torch.imag(target_pulse).detach().cpu().numpy(), label="target pulse imag")
    # plt.figure()
    # plt.plot(gradient.detach().cpu().numpy(), label="gradient")
    # plt.show()
    return model


def init_training(tconfig, model, lr, device=torch.device("cpu")):
    model = model.to(device)
    model.train()
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=lr["pulse"], amsgrad=True)
    if model.name == "MixedModel":
        pulse_params = list(model.model1.parameters()) + list(model.model2.parameters())
        gradient_params = list(model.model3.parameters())
        optimizer = torch.optim.AdamW(
            [{"params": pulse_params, "lr": lr["pulse"]}, {"params": gradient_params, "lr": lr["gradient"]}],
            amsgrad=True,
        )
    elif model.name == "MixeMdodel_RealPulse":
        pulse_params = list(model.pulse_model.parameters())
        gradient_params = [model.gradient_value]
        optimizer = torch.optim.AdamW(
            [{"params": pulse_params, "lr": lr["pulse"]}, {"params": gradient_params, "lr": lr["gradient"]}],
            amsgrad=True,
        )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.9, patience=500, min_lr=1e-6)
    losses = {key: [] for key, value in tconfig.loss_weights.items() if value > 0}
    losses["total"] = []
    return model, optimizer, scheduler, losses


def threshold_loss(x, threshold):
    threshold_loss = torch.max(torch.abs(x)) - threshold
    threshold_loss[threshold_loss < 0] = 0.0
    return threshold_loss**2


def load_data(path, mode="inference", device="cpu", bconfig_override=None):
    from utils_bloch.setup import get_smooth_targets

    data_dict = torch.load(path, weights_only=False, map_location=device)
    target_z = data_dict["targets"]["target_z"]
    target_xy = data_dict["targets"]["target_xy"]
    losses = data_dict["losses"]
    model = data_dict["model"]
    optimizer = data_dict["optimizer"]
    tconfig = data_dict["tconfig"]
    bconfig = data_dict["bconfig"] if bconfig_override == None else bconfig_override
    mconfig = data_dict["mconfig"]
    sconfig = data_dict["sconfig"]
    target_xy = get_smooth_targets(tconfig, bconfig)
    target_z = torch.sqrt(1 - target_xy.sum(dim=-1) ** 2)

    if mode == "inference":
        pulse = data_dict["pulse"].detach().cpu()
        gradient = data_dict["gradient"].detach().cpu()
        return model, pulse, gradient, target_z, target_xy, tconfig, bconfig
    return model, target_xy, optimizer, losses, tconfig, bconfig, mconfig, sconfig


def torch_unwrap(phase, discont=torch.pi):
    """
    Unwrap a tensor of phase angles to remove discontinuities.
    Args:
        phase (torch.Tensor): Input tensor of phase angles.
        discont (float): Discontinuity threshold (default: π).
    Returns:
        torch.Tensor: Unwrapped phase tensor.
    """
    diff = torch.diff(phase)
    diff_mod = (diff + torch.pi) % (2 * torch.pi) - torch.pi
    diff_mod[diff_mod == -torch.pi] = torch.pi
    phase_unwrapped = torch.cumsum(torch.cat((phase[..., :1], diff_mod), dim=-1), dim=-1)
    return phase_unwrapped


def regularization_factor(x, threshold=1):
    return 1 / (1 + 10 * x**2 / (threshold**2 + x))


def regularize_model_gradients(model):
    total_grad_norm = 0.0
    for param in model.parameters():
        if param.grad is not None:
            param_norm = param.grad.data.norm(2)
            total_grad_norm += param_norm.item() ** 2
    total_grad_norm = total_grad_norm**0.5
    print(f"Total Gradient Norm: {total_grad_norm}")
    factor = regularization_factor(total_grad_norm, 200)
    for param in model.parameters():
        if param.grad is not None:
            param.grad.data.mul_(factor)
    print(f"Gradient norm scaling down by {factor}")
    return model


def train(model, target_xy, optimizer, scheduler, losses, device, tconfig, bconfig, mconfig, sconfig):
    target_z = torch.sqrt(1 - target_xy.sum(dim=-1) ** 2)
    B0, B0_list, M0, sens, t_B1, pos, target_xy, target_z, model = move_to(
        (
            bconfig.fixed_inputs["B0"],
            bconfig.fixed_inputs["B0_list"],
            bconfig.fixed_inputs["M0"],
            bconfig.fixed_inputs["sens"],
            bconfig.fixed_inputs["t_B1"],
            bconfig.fixed_inputs["pos"],
            target_xy,
            target_z,
            model,
        ),
        device,
    )
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device)

    if tconfig.pre_train_inputs:
        (model, B1, G, _, _, _, _, _, _) = load_data("results/120625_Mixed_1Slice_90deg_L2/train_log.pt")
        model = pre_train(
            target_pulse=B1,
            target_gradient=G,
            model=model,
            fixed_inputs=bconfig.fixed_inputs,
            lr={"pulse": 1e-2, "gradient": 1e-2},
            thr=0.0027,
            device=device,
        )

    trainLogger = TrainLogger({"target_z": target_z, "target_xy": target_xy}, tconfig, bconfig, mconfig, sconfig)
    loss_fn = SliceProfileLoss(tconfig, bconfig, sconfig, target_xy)
    infoscreen = InfoScreen(target_xy, loss_fn.slices_mask, bconfig, output_every=tconfig.plot_loss_frequency, losses=losses)
    for step in range(tconfig.start_step, tconfig.steps + 1):
        pulse, gradient = model(t_B1)

        mxy, mz = blochsim_CK_batch(
            B1=pulse,
            G=gradient,
            pos=pos,
            sens=sens,
            B0_list=B0_list,
            M0=M0,
            dt=bconfig.fixed_inputs["dt_num"],
            time_loop="complex",
        )

        currentLosses = loss_fn(mz, mxy, pulse, gradient)
        currentLoss = sum(currentLosses.values())

        currentLossItems = {key: value.item() for key, value in currentLosses.items()}
        currentLossItems["total"] = currentLoss.item()
        for key, value in currentLossItems.items():
            losses[key].append(value)

        with torch.no_grad():
            trainLogger.log_step(step, losses, model, optimizer)
            infoscreen.plot_info(step, losses, mz, mxy, pulse, gradient, trainLogger)
            infoscreen.print_info(step, currentLossItems, optimizer, trainLogger.best_loss)

        optimizer.zero_grad()
        currentLoss.backward()
        if tconfig.suppress_loss_peaks:
            # model = regularize_model_gradients(model)
            if step > 100 and losses[-1] > 2 * trainLogger.best_loss:
                (model, target_z, target_xy, optimizer, _, fixed_inputs, flip_angle, loss_metric, scanner_params, loss_weights, _) = load_data(
                    os.path.join(trainLogger.export_loc, "train_log.pt"), mode="train", device=device
                )
                for param_group in optimizer.param_groups:
                    param_group["lr"] *= 0.1
                print("Loss peak detected, reloading model and reducing learning rate.")
                continue
        optimizer.step()
        scheduler.step(currentLossItems["total"])

    from forward import forward

    if os.path.exists(trainLogger.export_loc):
        forward(os.path.join(trainLogger.export_loc, "train_log.pt"), npts_some_b0_values=7, Nz=1024, Nt=256, npts_off_resonance=256)
