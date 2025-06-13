from time import time
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.collections import LineCollection
import matplotlib.cm as cm
import torch
import numpy as np
from params import model_args, fixed_inputs
import os
import datetime
import json
from utils_bloch import blochsim_CK_batch, blochsim_CK_batch_realPulse
from utils_infer import plot_off_resonance


class TrainLogger:
    def __init__(self, fixed_inputs, flip_angle, loss_metric, targets, model_args, scanner_params, loss_weights, start_logging=1):
        self.log = {}
        self.start_logging = start_logging
        self.best_loss = np.inf
        self.fixed_inputs = fixed_inputs
        self.flip_angle = flip_angle
        self.loss_metric = loss_metric
        self.targets = targets
        self.model_args = model_args
        self.scanner_params = scanner_params
        self.loss_weights = loss_weights
        self.log["flip_angle"] = self.flip_angle
        self.log["loss_metric"] = self.loss_metric
        self.log["model_args"] = model_args
        self.log["scanner_params"] = self.scanner_params
        self.log["loss_weights"] = self.loss_weights
        self.log["fixed_inputs"] = self.fixed_inputs
        self.log["targets"] = self.targets

    def log_epoch(self, epoch, total_loss, losses, model, optimizer, pulse, gradient):
        self.log["epoch"] = epoch
        self.log["L2_loss"] = total_loss.item()
        self.log["losses"] = losses
        self.log["model"] = model
        self.log["optimizer"] = optimizer
        self.log["pulse"] = pulse
        self.log["gradient"] = gradient
        return self.save(epoch, losses)

    def save(self, epoch, losses, filename="results/train_log.pt"):
        if not losses[-1] < 0.99 * self.best_loss:
            return False
        self.best_loss = losses[-1]
        if epoch <= self.start_logging:
            return False
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        torch.save(self.log, filename)
        self.export_json()
        print(f"Training log saved to {filename}")
        print("-" * 50)
        return True

    def export_json(self, directory="results"):
        modelname = self.log["model"].name
        export_path = os.path.join(directory, f"sms_nn_{modelname}.json")
        dur = 1000 * self.log["fixed_inputs"]["t_B1"][-1].item()

        data = {
            "id": "sms_nn_150425_MLP_square2",
            "set": {
                "maxB1": torch.max(torch.abs(self.log["pulse"])).item(),
                "dur": dur,
                "pts": len(self.log["fixed_inputs"]["t_B1"]),
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
    def __init__(self, output_every=1):
        self.t0 = time()
        self.t1 = time()
        self.output_every = output_every
        self.init_plots()

    def init_plots(self):
        self.fig = plt.figure(figsize=(13, 7), constrained_layout=False)
        spec = gridspec.GridSpec(2, 2, figure=self.fig)  # Create a 2x2 grid layout

        # First row: 3 plots spanning the entire width
        spec_top = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=spec[0, :])  # Split first row into 3 columns
        self.ax = [None] * 3
        self.ax[0] = self.fig.add_subplot(spec_top[0])
        self.ax[1] = self.fig.add_subplot(spec_top[1])
        self.ax[2] = self.fig.add_subplot(spec_top[2])
        # Second row: 2 equally wide plots
        self.ax_bottom_left = self.fig.add_subplot(spec[1, 0])
        self.ax_bottom_right = self.fig.add_subplot(spec[1, 1])
        self.ax_phase = self.ax_bottom_right.twinx()
        self.ax_phase.set_ylabel("Phase (radians)")
        self.ax_bottom_right.set_ylabel("|M_xy|")

        # Initialize plots for the first row
        self.pulse_real_plot = self.ax[0].plot([0], [1e-4], linewidth=0.8, label="Pulse real")[0]
        self.pulse_imag_plot = self.ax[0].plot([0], [1e-4], linewidth=0.8, label="Pulse imag")[0]
        self.pulse_abs_plot = self.ax[0].plot([0], [1e-4], linewidth=0.8, label="|Pulse|", linestyle="dotted")[0]
        self.grad_plot = self.ax[1].plot([0], [1], linewidth=0.8, label="Gradient")[0]
        self.loss_plot = self.ax[2].semilogy([0], [1e-1], linewidth=0.8, label="Loss")[0]

        # Initialize plots for the second row
        self.target_z_plot = self.ax_bottom_left.plot([0], [1], linewidth=0.8, label="Target_z")[0]
        self.target_xy_plot = self.ax_bottom_right.plot([0], [1], linewidth=0.8, label="Target_xy")[0]

        # Set titles and legends
        self.ax[0].set_title("Pulse")
        self.ax[1].set_title("Gradient")
        self.ax[2].set_title("Loss")
        self.ax_bottom_left.set_title("M_z")
        self.ax_bottom_right.set_title("M_xy")
        self.ax[0].legend()
        self.ax[1].legend()
        self.ax[2].legend()
        plt.show(block=False)

    def plot_info(self, epoch, losses, pos, t_B1, target_z, target_xy, mz, mxy, pulse, gradient, export_figure):
        """plots info curves during training"""
        if epoch % 1000 == 0 and epoch > 0:
            freq_offsets_Hz = torch.linspace(-8000, 8000, 64)
            plot_off_resonance(pulse + 0j, gradient, fixed_inputs, freq_offsets_Hz=freq_offsets_Hz)
            plt.show(block=False)

        if epoch % self.output_every == 0 or export_figure:
            pos = pos.cpu()[:, 2]
            fmin = -0.09  # torch.min(pos).item()
            fmax = 0.09  # torch.max(pos).item()
            t = t_B1.detach().cpu().numpy()
            mz_plot = mz.detach().cpu().numpy()
            mxy_abs = np.abs(mxy.detach().cpu().numpy())
            tgt_z = target_z.detach().cpu().numpy()
            tgt_xy = target_xy.detach().cpu().numpy()
            pulse_real = np.real(pulse.detach().cpu().numpy())
            pulse_imag = np.imag(pulse.detach().cpu().numpy())
            pulse_abs = np.sqrt(pulse_real**2 + pulse_imag**2)
            gradient_for_plot = gradient.detach().cpu().numpy()
            phase = np.unwrap(np.angle(mxy.detach().cpu().numpy()), axis=-1)
            where_slices_are = tgt_xy > 0.5
            phasemin = np.min(phase[:, where_slices_are])
            phasemax = np.max(phase[:, where_slices_are])
            phasemin = phasemin - 0.5 * (phasemax - phasemin)
            phasemax = phasemax + 0.5 * (phasemax - phasemin)
            phase[:, ~where_slices_are] = np.nan

            for collection in self.ax_bottom_left.collections:
                collection.remove()
            for collection in self.ax_bottom_right.collections:
                collection.remove()
            for collection in self.ax_phase.collections:
                collection.remove()

            self.target_z_plot.set_xdata(pos)
            self.target_xy_plot.set_xdata(pos)
            self.grad_plot.set_xdata(t)
            self.pulse_real_plot.set_xdata(t)
            self.pulse_imag_plot.set_xdata(t)
            self.pulse_abs_plot.set_xdata(t)
            self.loss_plot.set_xdata(np.arange(len(losses)))

            self.add_line_collection(self.ax_bottom_left, pos, mz_plot)
            self.add_line_collection(self.ax_bottom_right, pos, mxy_abs)
            self.add_line_collection(self.ax_phase, pos, phase, linestyle="dotted")
            self.target_z_plot.set_ydata(tgt_z)
            self.target_xy_plot.set_ydata(tgt_xy)
            self.grad_plot.set_ydata(gradient_for_plot)
            self.pulse_real_plot.set_ydata(pulse_real)
            self.pulse_imag_plot.set_ydata(pulse_imag)
            self.pulse_abs_plot.set_ydata(pulse_abs)
            self.loss_plot.set_ydata(losses)

            self.ax_bottom_left.set_xlim(fmin, fmax)
            self.ax_bottom_right.set_xlim(fmin, fmax)
            self.ax_phase.set_xlim(fmin, fmax)
            self.ax[0].set_xlim(t[0], t[-1])
            self.ax[1].set_xlim(t[0], t[-1])
            self.ax[2].set_xlim(0, epoch + 1)
            self.ax_bottom_left.set_ylim(-0.1, 1.1)
            self.ax_bottom_right.set_ylim(-0.1, 1.1)
            self.ax_phase.set_ylim(phasemin, phasemax)
            self.ax[0].set_ylim((-np.max(pulse_abs), np.max(pulse_abs)))
            self.ax[2].set_ylim((0.9 * np.min(losses).item(), 1.1 * np.max(losses).item()))
            self.ax[1].set_ylim((-1.1 * np.max(np.abs(gradient_for_plot)).item(), 1.1 * np.max(np.abs(gradient_for_plot)).item()))

            self.fig.canvas.draw_idle()
            if export_figure:
                filename = "results/training.png"
                os.makedirs(os.path.dirname(filename), exist_ok=True)
                plt.savefig(filename, dpi=300)
            plt.pause(0.05)
            self.fig.canvas.flush_events()

    def add_line_collection(self, ax, xdata, ydata, linestyle="solid"):
        line_list = [list(zip(xdata, ydata[b, :])) for b in range(ydata.shape[0])]
        self.add_line_collection_from_list(ax, line_list, linestyle)

    def add_line_collection_from_list(self, ax, line_list, linestyle):
        cmap = cm.get_cmap("inferno", len(line_list) + 2)
        colors = [cmap(i + 1) for i in range(len(line_list) + 2)]
        line_collection = LineCollection(line_list, linewidths=0.7, colors=colors, linestyle=linestyle)
        ax.add_collection(line_collection)

    def print_info(self, epoch, loss, optimizer, best_loss):
        self.t1 = time() - self.t0
        self.t0 = time()
        print("Epoch: ", epoch)
        print(f"Loss: {loss:.6f}")
        print(f"Best Loss: {best_loss:.6f}")
        for i, param_group in enumerate(optimizer.param_groups):
            print(f"Learning rate {i}: {param_group['lr']:.7f}")
        print(f"Time: {self.t1:.1f}")
        print("-" * 100)


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
    epoch = 0
    while loss > thr:
        epoch += 1
        pulse, gradient = model(t_B1)

        loss_pulse = torch.mean(torch.abs(pulse - target_pulse) ** 2)
        loss_gradient = torch.mean((gradient - target_gradient) ** 2)
        loss = loss_pulse + loss_gradient / model_args["gradient_scale"]

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step(loss.item())
        if epoch % 1000 == 0:
            print(f"Epoch: {epoch}, Loss: {loss.item():.6f}, lr: {optimizer.param_groups[0]['lr']}")
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


def init_training(model, lr, device=torch.device("cpu")):
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
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.9, patience=100, min_lr=1e-7)
    losses = []
    return model, optimizer, scheduler, losses


def threshold_loss(x, threshold):
    threshold_loss = torch.max(torch.abs(x)) - threshold
    threshold_loss[threshold_loss < 0] = 0.0
    return threshold_loss**2


def loss_fn(
    posAx,
    z_profile,
    xy_profile,
    target_z,
    target_xy,
    pulse,
    gradient,
    delta_t,
    scanner_params,
    loss_weights,
    metric="L2",
    verbose=False,
):
    xy_profile_abs = torch.abs(xy_profile)
    if metric == "L2":
        loss_mxy = torch.mean((xy_profile_abs - target_xy) ** 2).mean(dim=0)
        loss_mz = torch.mean((z_profile - target_z) ** 2).mean(dim=0)
    elif metric == "L1":
        loss_mxy = torch.mean(torch.abs(xy_profile_abs - target_xy)).mean(dim=0)
        loss_mz = torch.mean(torch.abs(z_profile - target_z)).mean(dim=0)
    else:
        raise ValueError("Invalid metric. Choose 'L2' or 'L1'.")
    boundary_vals_pulse = (torch.abs(pulse[0]) ** 2 + torch.abs(pulse[-1]) ** 2).mean(dim=0)
    gradient_height_loss = threshold_loss(gradient, scanner_params["max_gradient"]).mean(dim=0)
    pulse_height_loss = threshold_loss(pulse, scanner_params["max_pulse_amplitude"]).mean(dim=0)
    if gradient.shape[0] > 1:
        gradient_diff_loss = threshold_loss(torch.diff(gradient.squeeze()), scanner_params["max_diff_gradient"] * delta_t).mean(dim=0)
    else:
        gradient_diff_loss = torch.zeros(1, device=z_profile.device)
    phase = torch_unwrap(torch.angle(xy_profile))
    phase_diff = torch.diff(phase)
    phase_ddiff = torch.diff(phase_diff)
    where_peaks_are = target_xy > 1e-6
    phase_ddiff = 100 * torch.mean(phase_ddiff[:, where_peaks_are[1:-1]] ** 2, dim=-1)
    phase_diff_var = torch.var(phase_diff[:, where_peaks_are[:-1]], dim=-1)
    phase_diff_loss = torch.mean(phase_diff[:, where_peaks_are[:-1]] ** 2, dim=-1)
    # phase_left = phase[:, (posAx < 0) & (where_peaks_are)].mean(dim=-1)
    # phase_right = phase[:, (posAx > 0) & (where_peaks_are)].mean(dim=-1)
    # phase_left_right = ((torch.abs(phase_left - phase_right) - np.pi) % 2 * np.pi) ** 2
    phase_loss = (phase_ddiff + phase_diff_var + phase_diff_loss).mean(dim=0)

    if verbose:
        print("-" * 50)
        print("LOSSES:")
        print("loss_mxy", loss_weights["loss_mxy"] * loss_mxy.item())
        print("loss_mz", loss_weights["loss_mz"] * loss_mz.item())
        print("boundary_vals_pulse", loss_weights["boundary_vals_pulse"] * boundary_vals_pulse.item())
        print("gradient_height_loss", loss_weights["gradient_height_loss"] * gradient_height_loss.item())
        print("pulse_height_loss", loss_weights["pulse_height_loss"] * pulse_height_loss.item())
        print("gradient_diff_loss", loss_weights["gradient_diff_loss"] * gradient_diff_loss.item())
        print("phase_loss", loss_weights["phase_loss"] * phase_loss.item())
        print("phase_diff_var", loss_weights["phase_loss"] * phase_diff_var.mean().item())
        print("phase_ddiff", loss_weights["phase_loss"] * phase_ddiff.mean().item())
        print("phase_diff_loss", loss_weights["phase_loss"] * phase_diff_loss.mean().item())
        # print("phase_left_right", loss_weights["phase_loss"] * phase_left_right.mean().item())
        print("-" * 50)
    return (
        loss_weights["loss_mxy"] * loss_mxy,
        loss_weights["loss_mz"] * loss_mz,
        loss_weights["boundary_vals_pulse"] * boundary_vals_pulse,
        loss_weights["gradient_height_loss"] * gradient_height_loss,
        loss_weights["pulse_height_loss"] * pulse_height_loss,
        loss_weights["gradient_diff_loss"] * gradient_diff_loss,
        loss_weights["phase_loss"] * phase_loss,
    )


def load_data(path, mode="inference", device="cpu"):
    creation_time = os.path.getctime(path)
    creation_datetime = datetime.datetime.fromtimestamp(creation_time)
    refactor_date = datetime.datetime(2025, 5, 23)
    if creation_datetime < refactor_date:
        return load_data_legacy(path, mode=mode)
    else:
        return load_data_new(path, mode=mode, device=device)


def load_data_new(path, mode="inference", device="cpu"):
    data_dict = torch.load(path, weights_only=False, map_location=device)
    target_z = data_dict["targets"]["target_z"]
    target_xy = data_dict["targets"]["target_xy"]
    epoch = data_dict["epoch"]
    losses = data_dict["losses"]
    model = data_dict["model"]
    optimizer = data_dict["optimizer"]
    fixed_inputs = data_dict["fixed_inputs"]
    flip_angle = data_dict["flip_angle"]
    loss_metric = data_dict["loss_metric"]
    scanner_params = data_dict["scanner_params"]
    loss_weights = data_dict["loss_weights"]
    if mode == "inference":
        pulse = data_dict["pulse"].detach().cpu()
        gradient = data_dict["gradient"].detach().cpu()
        return pulse, gradient, target_z, target_xy, fixed_inputs
    return model, target_z, target_xy, optimizer, losses, fixed_inputs, flip_angle, loss_metric, scanner_params, loss_weights, epoch


def load_data_legacy(path, mode="inference"):
    import params

    data_dict = torch.load(path, weights_only=False, map_location="cpu")
    target_z = data_dict["targets"]["target_z"]
    target_xy = data_dict["targets"]["target_xy"]
    epoch = data_dict["epoch"]
    losses = data_dict["losses"]
    model = data_dict["model"]
    optimizer = data_dict["optimizer"]
    (pos, dt, dx, Nz, sens, B0, tAx, fAx, t_B1, M0, inputs) = data_dict["inputs"]
    fixed_inputs = params.fixed_inputs

    fixed_inputs["pos"] = pos
    fixed_inputs["dt"] = dt
    fixed_inputs["dx"] = dx
    fixed_inputs["Nz"] = Nz
    fixed_inputs["sens"] = sens
    fixed_inputs["B0"] = B0
    fixed_inputs["tAx"] = tAx
    fixed_inputs["fAx"] = fAx
    fixed_inputs["t_B1"] = t_B1
    fixed_inputs["M0"] = M0
    fixed_inputs["inputs"] = inputs
    flip_angle = data_dict["flip_angle"] if "flip_angle" in data_dict else params.flip_angle
    loss_metric = data_dict["loss_metric"] if "loss_metric" in data_dict else params.loss_metric
    scanner_params = data_dict["scanner_params"] if "scanner_params" in data_dict else params.scanner_params
    loss_weights = data_dict["loss_weights"] if "loss_weights" in data_dict else params.loss_weights

    if mode == "inference":
        pulse = data_dict["pulse"].detach().cpu()
        gradient = data_dict["gradient"].detach().cpu()
        return pulse, gradient, target_z, target_xy, fixed_inputs
    return model, target_z, target_xy, optimizer, losses, fixed_inputs, flip_angle, loss_metric, scanner_params, loss_weights, epoch


def load_data_old(path):
    import params

    data_dict = torch.load(path, weights_only=False, map_location="cpu")
    epoch = data_dict["epoch"]
    losses = data_dict["losses"]
    model = data_dict["model"]
    optimizer = data_dict["optimizer"]
    (pos, dt, dx, Nz, sens, B0, tAx, fAx, t_B1, M0, inputs) = data_dict["inputs"]
    target_z = data_dict["targets"]["target_z"]
    target_xy = data_dict["targets"]["target_xy"]
    pulse = data_dict["pulse"].detach().cpu()
    gradient = data_dict["gradient"].detach().cpu()

    fixed_inputs = params.fixed_inputs

    fixed_inputs["pos"] = pos
    fixed_inputs["dt"] = dt
    fixed_inputs["dx"] = dx
    fixed_inputs["Nz"] = Nz
    fixed_inputs["sens"] = sens
    fixed_inputs["B0"] = B0
    fixed_inputs["tAx"] = tAx
    fixed_inputs["fAx"] = fAx
    fixed_inputs["t_B1"] = t_B1
    fixed_inputs["M0"] = M0
    fixed_inputs["inputs"] = inputs
    return epoch, losses, model, optimizer, pulse, gradient, target_z, target_xy, fixed_inputs


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


def train(
    model,
    target_z,
    target_xy,
    optimizer,
    scheduler,
    losses,
    fixed_inputs,
    flip_angle,
    loss_metric,
    scanner_params,
    loss_weights,
    start_epoch,
    epochs,
    device,
    start_logging,
    plot_loss_freq,
    pre_train_inputs=False,
    suppress_loss_peaks=False,
):
    B0, B0_list, M0, sens, t_B1, pos, target_z, target_xy, model = move_to(
        (
            fixed_inputs["B0"],
            fixed_inputs["B0_list"],
            fixed_inputs["M0"],
            fixed_inputs["sens"],
            fixed_inputs["t_B1"],
            fixed_inputs["pos"],
            target_z,
            target_xy,
            model,
        ),
        device,
    )
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device)

    if pre_train_inputs:
        (B1, G, _, _, fixed_inputs) = load_data("results/120625_Mixed_1Slice_90deg_L2/train_log.pt")
        model = pre_train(
            target_pulse=B1,
            target_gradient=G,
            model=model,
            fixed_inputs=fixed_inputs,
            lr={"pulse": 1e-2, "gradient": 1e-2},
            thr=0.0027,
            device=device,
        )

    infoscreen = InfoScreen(output_every=plot_loss_freq)
    trainLogger = TrainLogger(
        fixed_inputs,
        flip_angle,
        loss_metric,
        {"target_z": target_z, "target_xy": target_xy},
        model_args,
        scanner_params,
        loss_weights,
        start_logging=start_logging,
    )
    for epoch in range(start_epoch, epochs + 1):
        # if epoch % 1000 == 0 and epoch > 0:
        #     loss_weights["phase_loss"] *= 0.1
        pulse, gradient = model(t_B1)

        # shift = 0.0025
        # exponent = 1j * torch.cumsum(gradient, dim=0) * fixed_inputs["dt"] * 2 * torch.pi * shift * fixed_inputs["gam"]
        # pulse_left = pulse * torch.exp(-exponent)
        # pulse_right = pulse * torch.exp(exponent)
        # pulse = pulse_left + pulse_right

        mxy, mz = blochsim_CK_batch(
            B1=pulse,
            G=gradient,
            pos=pos,
            sens=sens,
            B0_list=B0_list,
            M0=M0,
            dt=fixed_inputs["dt"],
            time_loop="complex",
        )
        (loss_mxy, loss_mz, boundary_vals_pulse, gradient_height_loss, pulse_height_loss, gradient_diff_loss, phase_loss) = loss_fn(
            fixed_inputs["pos"][:, 2],
            mz,
            mxy,
            target_z,
            target_xy,
            pulse,
            gradient,
            1000 * fixed_inputs["dt"],
            scanner_params=scanner_params,
            loss_weights=loss_weights,
            metric=loss_metric,
            verbose=True,
        )
        loss = loss_mxy + loss_mz + gradient_height_loss + gradient_diff_loss + pulse_height_loss + boundary_vals_pulse + phase_loss

        lossItem = loss.item()
        losses.append(lossItem)
        optimizer.zero_grad()
        loss.backward()
        if suppress_loss_peaks:
            # model = regularize_model_gradients(model)
            if epoch > 100 and losses[-1] > 2 * trainLogger.best_loss:
                (model, target_z, target_xy, optimizer, _, fixed_inputs, flip_angle, loss_metric, scanner_params, loss_weights, _) = load_data(
                    "results/train_log.pt", mode="train", device=device
                )
                for param_group in optimizer.param_groups:
                    param_group["lr"] *= 0.1
                print("Loss peak detected, reloading model and reducing learning rate.")
                continue
        optimizer.step()
        scheduler.step(lossItem)

        new_optimum = trainLogger.log_epoch(epoch, loss, losses, model, optimizer, pulse, gradient)
        infoscreen.plot_info(epoch, losses, pos, t_B1, target_z, target_xy, mz, mxy, pulse, gradient, new_optimum)
        infoscreen.print_info(epoch, lossItem, optimizer, trainLogger.best_loss)
