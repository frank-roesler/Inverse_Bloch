from time import time
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import torch
import numpy as np
from params import model_args
import os
import json


class TrainLogger:
    def __init__(self, start_logging=1):
        self.log = {}
        self.start_logging = start_logging
        self.best_loss = np.inf

    def log_epoch(
        self,
        epoch,
        L2_loss,
        D_loss,
        losses,
        model,
        optimizer,
        pulse,
        gradient,
        fixed_inputs,
        flip_angle,
        loss_metric,
        targets,
        axes,
    ):
        self.log["epoch"] = epoch
        self.log["L2_loss"] = L2_loss.item()
        self.log["D_loss"] = D_loss.item()
        self.log["losses"] = losses
        self.log["model"] = model
        self.log["optimizer"] = optimizer
        self.log["inputs"] = fixed_inputs
        self.log["targets"] = targets
        self.log["pulse"] = pulse
        self.log["gradient"] = gradient
        self.log["axes"] = axes
        self.log["flip_angle"] = flip_angle
        self.log["loss_metric"] = loss_metric
        return self.save(epoch, losses)

    def save(self, epoch, losses, filename="results/train_log.pt"):
        if epoch <= self.start_logging:
            return False
        if not losses[-1] < 0.999 * self.best_loss:
            return False
        self.best_loss = losses[-1]
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        torch.save(self.log, filename)
        self.export_json()
        print(f"Training log saved to {filename}")
        print("-" * 50)
        return True

    def export_json(self, directory="results"):
        modelname = self.log["model"].name
        export_path = os.path.join(directory, f"sms_nn_{modelname}.json")
        dur = 1000 * self.log["axes"]["t_B1"][-1].item()

        data = {
            "id": "sms_nn_150425_MLP_square2",
            "set": {
                "maxB1": torch.max(torch.abs(self.log["pulse"])).item(),
                "dur": dur,
                "pts": len(self.log["axes"]["t_B1"]),
                "amplInt": None,
                "refocFract": None,
            },
            "conf": {"slthick": 0.02, "bs": 1.5, "tb": 4.8, "mb": 2},
            "rfPls": {
                "real": torch.real(self.log["pulse"]).squeeze().tolist(),
                "imag": torch.imag(self.log["pulse"]).squeeze().tolist(),
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
        self.fig, self.ax = plt.subplots(2, 3, figsize=(13, 6), constrained_layout=True)

        self.ax_phase = self.ax[1, 1].twinx()
        self.ax_phase.set_ylabel("Phase (radians)")

        # Initialize plots
        self.pulse_real_plot = self.ax[0, 0].plot([0], [1e-4], linewidth=0.9, label="Pulse real")[0]
        self.pulse_imag_plot = self.ax[0, 0].plot([0], [1e-4], linewidth=0.9, label="Pulse imag")[0]
        self.pulse_abs_plot = self.ax[0, 0].plot([0], [1e-4], linewidth=0.9, label="|Pulse|", linestyle="dotted")[0]
        self.grad_plot = self.ax[0, 1].plot([0], [1], linewidth=0.9, label="Gradient")[0]
        self.loss_plot = self.ax[0, 2].semilogy([0], [1e-1], linewidth=0.9, label="Loss")[0]
        self.target_z_plot = self.ax[1, 0].plot([0], [1], linewidth=0.9, label="Target_z")[0]
        self.target_xy_plot = self.ax[1, 1].plot([0], [1], linewidth=0.9, label="Target_xy")[0]

        # Set titles and legends
        self.ax[0, 0].set_title("Pulse")
        self.ax[0, 1].set_title("Gradient")
        self.ax[0, 2].set_title("Loss")
        self.ax[1, 0].set_title("M_z profile")
        self.ax[1, 1].set_title("M_xy profile")
        self.ax[1, 2].set_title("Slice mean of |M_xy(t)|")
        self.ax[0, 0].legend()
        self.ax[0, 1].legend()
        self.ax[0, 2].legend()
        self.ax[1, 0].legend()

        # Combine legends from both axes
        lines_bottom_right, labels_bottom_right = self.ax[1, 1].get_legend_handles_labels()
        lines_phase, labels_phase = self.ax_phase.get_legend_handles_labels()
        self.ax[1, 1].legend(lines_bottom_right + lines_phase, labels_bottom_right + labels_phase, loc="upper right")

    def plot_info(
        self,
        epoch,
        losses,
        pos,
        t_B1,
        target_z,
        target_xy,
        mz,
        mxy,
        mxy_t_integrated,
        pulse,
        gradient,
        export_figure,
    ):
        """plots info curves during training"""
        if epoch % self.output_every == 0 or epoch < 10:
            pos = pos.detach().cpu().numpy()[:, 2]
            fmin = np.min(pos).item()
            fmax = np.max(pos).item()
            t = t_B1.detach().cpu().numpy()
            mz_plot = mz.detach().cpu().numpy()
            mxy_abs = np.abs(mxy.detach().cpu().numpy())
            mxy_t_integrated = mxy_t_integrated.detach().cpu().numpy()
            target_z = target_z.detach().cpu().numpy()
            target_xy = target_xy.detach().cpu().numpy()
            pulse_real = np.real(pulse.detach().cpu().numpy())
            pulse_imag = np.imag(pulse.detach().cpu().numpy())
            pulse_abs = np.sqrt(pulse_real**2 + pulse_imag**2)
            gradient_for_plot = gradient.detach().cpu().numpy()
            phase = np.unwrap(np.angle(mxy.detach().cpu().numpy()))
            phasemin = np.min(phase)
            phasemax = np.max(phase)
            phase[:, target_xy < 0.5] = np.nan

            for collection in self.ax[1, 0].collections:
                collection.remove()
            for collection in self.ax[1, 1].collections:
                collection.remove()
            for collection in self.ax[1, 2].collections:
                collection.remove()
            for collection in self.ax_phase.collections:
                collection.remove()
            self.target_z_plot.set_xdata(pos)
            self.target_xy_plot.set_xdata(pos)
            self.grad_plot.set_xdata(t)
            self.pulse_real_plot.set_xdata(t)
            self.pulse_imag_plot.set_xdata(t)
            self.pulse_abs_plot.set_xdata(t)
            self.loss_plot.set_xdata(np.arange(epoch + 1))

            self.target_z_plot.set_ydata(target_z)
            self.target_xy_plot.set_ydata(target_xy)
            self.add_line_collection(self.ax[1, 0], pos, mz_plot)
            self.add_line_collection(self.ax[1, 1], pos, mxy_abs)
            self.add_line_collection(self.ax_phase, pos, phase)
            self.grad_plot.set_ydata(gradient_for_plot)
            self.pulse_real_plot.set_ydata(pulse_real)
            self.pulse_imag_plot.set_ydata(pulse_imag)
            self.pulse_abs_plot.set_ydata(pulse_abs)
            self.loss_plot.set_ydata(losses)
            line_list = [list(zip(t.squeeze(), mxy_t_integrated[0, c, :])) for c in range(mxy_t_integrated.shape[1])]
            self.add_line_collection_from_list(self.ax[1, 2], line_list)

            self.ax[1, 0].set_xlim(fmin, fmax)
            self.ax[1, 1].set_xlim(fmin, fmax)
            self.ax_phase.set_xlim(fmin, fmax)
            self.ax[0, 0].set_xlim(t[0], t[-1])
            self.ax[0, 1].set_xlim(t[0], t[-1])
            self.ax[1, 2].set_xlim(t[0], t[-1])
            self.ax[0, 2].set_xlim(0, epoch + 1)
            self.ax[1, 0].set_ylim(-0.1, 1.1)
            self.ax[1, 1].set_ylim(-0.1, 1.1)
            self.ax_phase.set_ylim(phasemin, phasemax)
            self.ax[0, 0].set_ylim((-np.max(pulse_abs), np.max(pulse_abs)))
            self.ax[0, 2].set_ylim((0.9 * np.min(losses).item(), 1.1 * np.max(losses).item()))
            self.ax[0, 1].set_ylim((-1.1 * np.max(np.abs(gradient_for_plot)).item(), 1.1 * np.max(np.abs(gradient_for_plot)).item()))

            self.fig.canvas.draw()
            if export_figure:
                filename = "results/training.png"
                os.makedirs(os.path.dirname(filename), exist_ok=True)
                plt.savefig(filename, dpi=300)
            plt.pause(0.001)

    def add_line_collection(self, ax, xdata, ydata):
        line_list = [list(zip(xdata, ydata[b, :])) for b in range(ydata.shape[0])]
        self.add_line_collection_from_list(ax, line_list)

    def add_line_collection_from_list(self, ax, line_list):
        cmap = cm.get_cmap("inferno", len(line_list) + 2)
        colors = [cmap(i + 1) for i in range(len(line_list) + 2)]
        line_collection = LineCollection(line_list, linewidths=0.8, colors=colors)
        ax.add_collection(line_collection)

    def print_info(self, epoch, loss, optimizer):
        self.t1 = time() - self.t0
        self.t0 = time()
        print("Epoch: ", epoch)
        print(f"Loss: {loss:.5f}")
        for i, param_group in enumerate(optimizer.param_groups):
            print(f"Learning rate {i}: {param_group['lr']:.6f}")
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


def pre_train(target_pulse, target_gradient, model, lr=1e-4, thr=1e-3, device=torch.device("cpu")):
    import params

    target_pulse = target_pulse.to(device)
    target_gradient = target_gradient.to(device)
    model, optimizer, scheduler, _ = init_training(model, lr=lr, device=device)
    t_B1 = params.t_B1.to(device)
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
        # scheduler.step(loss.item())
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
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.9, patience=30, min_lr=2e-6)
    losses = []
    return model, optimizer, scheduler, losses


def threshold_loss(x, threshold):
    threshold_loss = torch.max(torch.abs(x)) - threshold
    threshold_loss[threshold_loss < 0] = 0.0
    return threshold_loss**2


def compute_phase_center_loss(phase, slice_centers, slice_half_width):
    slice_masks = torch.stack(
        [
            (torch.arange(phase.size(1), device=phase.device) >= c - slice_half_width)
            & (torch.arange(phase.size(1), device=phase.device) < c + slice_half_width)
            for c in slice_centers
        ],
        dim=0,
    )  # Shape: (num_slices, num_points)
    slice_masks = slice_masks.T  # Shape: (num_points, num_slices)
    phase_center_loss = torch.einsum("bp,ps->bs", phase, slice_masks.float()) / slice_masks.sum(dim=0, keepdim=True)
    phase_center_loss = torch.diff(phase_center_loss, dim=1)
    factor = torch.pi / len(slice_centers)
    phase_center_loss = torch.mean((torch.abs(phase_center_loss) - factor) ** 2, dim=0)

    return phase_center_loss


def compute_phase_center_loss_old(xy_profile, phase, slice_centers, slice_half_width):
    phase_center_loss = torch.zeros((xy_profile.shape[0], len(slice_centers)), device=xy_profile.device)
    for i, c in enumerate(slice_centers):
        phase_center_loss[:, i] = torch.mean(phase[:, c - slice_half_width : c + slice_half_width], dim=1)
    phase_center_loss = torch.diff(phase_center_loss, dim=1)
    factor = torch.pi / len(slice_centers)
    phase_center_loss = torch.mean(((torch.abs(phase_center_loss) - factor) % (2 * torch.pi)) ** 2, dim=0)
    return phase_center_loss


def loss_fn(
    z_profile,
    xy_profile,
    target_z,
    target_xy,
    pulse,
    gradient,
    mxy_t_integrated,
    delta_t,
    weights,
    slice_centers,
    slice_half_width,
    scanner_params,
    loss_weights,
    metric="L2",
    verbose=False,
):
    xy_profile_abs = torch.abs(xy_profile)
    if metric == "L2":
        loss_mxy = torch.mean((xy_profile_abs - target_xy) ** 2).sum(dim=0)
        loss_mz = torch.mean((z_profile - target_z) ** 2).sum(dim=0)
    elif metric == "L1":
        loss_mxy = torch.mean(torch.abs(xy_profile_abs - target_xy)).sum(dim=0)
        loss_mz = torch.mean(torch.abs(z_profile - target_z)).sum(dim=0)
    else:
        raise ValueError("Invalid metric. Choose 'L2' or 'L1'.")
    boundary_vals_pulse = (torch.abs(pulse[0]) ** 2 + torch.abs(pulse[-1]) ** 2).sum(dim=0)
    gradient_height_loss = threshold_loss(gradient, scanner_params["max_gradient"]).sum(dim=0)
    pulse_height_loss = threshold_loss(pulse, scanner_params["max_pulse_amplitude"]).sum(dim=0)
    gradient_diff_loss = threshold_loss(torch.diff(gradient.squeeze()), scanner_params["max_diff_gradient"] * delta_t).sum(dim=0)
    where_slices = target_xy > 1e-1
    phase = torch_unwrap(torch.angle(xy_profile))
    phase_diff = torch.diff(phase)
    phase_ddiff = torch.diff(phase_diff)
    phase_ddiff = phase_ddiff[:, where_slices[1:-1]]
    phase_diff_var = torch.var(phase_diff[:, where_slices[:-1]])
    phase_loss = (torch.mean(phase_ddiff**2) + phase_diff_var).sum(dim=0)

    # t0 = time()
    phase_center_loss = 0  # compute_phase_center_loss_old(xy_profile, phase, slice_centers, slice_half_width)
    # t1 = time()
    # phase_center_loss = compute_phase_center_loss(phase, slice_centers, slice_half_width)
    # t2 = time()
    # print("compute_phase_center_loss_old", t1 - t0)
    # print("compute_phase_center_loss", t2 - t1)
    # print("error", phase_center_loss_old - phase_center_loss)
    # com = delta_t * (mxy_t_integrated[:, :, -1] * mxy_t_integrated.shape[-1] - torch.sum(mxy_t_integrated, dim=-1))
    center_of_mass_loss = 0  # torch.var(com, dim=1).sum(dim=0)

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
        # print("center_of_mass_loss", loss_weights["center_of_mass_loss"] * center_of_mass_loss.item())
        # print("phase_center_loss", loss_weights["phase_center_loss"] * phase_center_loss.item())
        print("-" * 50)
    return (
        loss_weights["loss_mxy"] * loss_mxy,
        loss_weights["loss_mz"] * loss_mz,
        loss_weights["boundary_vals_pulse"] * boundary_vals_pulse,
        loss_weights["gradient_height_loss"] * gradient_height_loss,
        loss_weights["pulse_height_loss"] * pulse_height_loss,
        loss_weights["gradient_diff_loss"] * gradient_diff_loss,
        loss_weights["phase_loss"] * phase_loss,
        loss_weights["center_of_mass_loss"] * center_of_mass_loss,
        loss_weights["phase_center_loss"] * phase_center_loss,
    )


def load_data(path):
    data_dict = torch.load(path, weights_only=False, map_location="cpu")
    # epoch = data_dict["epoch"]
    # L2_loss = data_dict["L2_loss"]
    # D_loss = data_dict["D_loss"]
    # losses = data_dict["losses"]
    # model = data_dict["model"]
    # optimizer = data_dict["optimizer"]
    # inputs = data_dict["inputs"]
    targets = data_dict["targets"]
    axes = data_dict["axes"]
    pulse = data_dict["pulse"].detach().cpu()
    gradient = data_dict["gradient"].detach().cpu()
    return pulse, gradient, axes, targets


def torch_unwrap(phase):
    """
    Unwrap a tensor of phase angles to remove discontinuities.
    Args:
        phase (torch.Tensor): Input tensor of phase angles.
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
