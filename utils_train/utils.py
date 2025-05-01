from time import time
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torch
import numpy as np
from params import get_fixed_inputs, model_args
import os
import json


class TrainLogger:
    def __init__(self, start_logging=1):
        self.log = {}
        self.start_logging = start_logging
        self.best_loss = np.inf

    def log_epoch(self, epoch, L2_loss, D_loss, losses, model, optimizer, pulse, gradient, inputs, targets, axes):
        self.log["epoch"] = epoch
        self.log["L2_loss"] = L2_loss.item()
        self.log["D_loss"] = D_loss.item()
        self.log["losses"] = losses
        self.log["model"] = model
        self.log["optimizer"] = optimizer
        self.log["inputs"] = inputs
        self.log["targets"] = targets
        self.log["pulse"] = pulse
        self.log["gradient"] = gradient
        self.log["axes"] = axes
        self.save(epoch, losses)
        self.export_json()

    def save(self, epoch, losses, filename="results/train_log.pt"):
        if epoch <= self.start_logging:
            return
        if not losses[-1] < 0.999 * self.best_loss:
            return
        self.best_loss = losses[-1]
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        torch.save(self.log, filename)
        print(f"Training log saved to {filename}")

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
        self.fig = plt.figure(figsize=(11, 6), constrained_layout=False)
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

        # Initialize plots for the first row
        self.pulse_real_plot = self.ax[0].plot([0], [1e-4], linewidth=1, label="Pulse real")[0]
        self.pulse_imag_plot = self.ax[0].plot([0], [1e-4], linewidth=1, label="Pulse imag")[0]
        self.grad_plot = self.ax[1].plot([0], [1], linewidth=1, label="Gradient")[0]
        self.loss_plot = self.ax[2].semilogy([0], [1e-1], linewidth=1, label="Loss")[0]

        # Initialize plots for the second row
        self.target_z_plot = self.ax_bottom_left.plot([0], [1], linewidth=1, label="Target_z")[0]
        self.mz_plot = self.ax_bottom_left.plot([0], [1], linewidth=1, label="M_z")[0]
        self.target_xy_plot = self.ax_bottom_right.plot([0], [1], linewidth=1, label="Target_xy")[0]
        self.mxy_plot = self.ax_bottom_right.plot([0], [1], linewidth=1, label="|M_xy|")[0]
        self.phase_plot = self.ax_phase.plot([0], [1], linewidth=1, label="Phase", color="g")[0]

        # Set titles and legends
        self.ax[0].set_title("Pulse")
        self.ax[1].set_title("Gradient")
        self.ax[2].set_title("Loss")
        self.ax_bottom_left.set_title("M_z")
        self.ax_bottom_right.set_title("M_xy")
        self.ax[0].legend()
        self.ax[1].legend()
        self.ax[2].legend()
        self.ax_bottom_left.legend()
        # self.ax_bottom_right.legend()
        # self.ax_phase.legend()

        # Combine legends from both axes
        lines_bottom_right, labels_bottom_right = self.ax_bottom_right.get_legend_handles_labels()
        lines_phase, labels_phase = self.ax_phase.get_legend_handles_labels()
        self.ax_bottom_right.legend(
            lines_bottom_right + lines_phase, labels_bottom_right + labels_phase, loc="upper right"
        )

    def plot_info(self, epoch, losses, fAx, t_B1, target_z, target_xy, mz, mxy, pulse, gradient):
        """plots info curves during training"""
        if epoch % self.output_every == 0:
            fAx = fAx.cpu()[:, 2]
            fmin = -0.07  # torch.min(fAx).item()
            fmax = 0.07  # torch.max(fAx).item()
            t = t_B1.detach().cpu().numpy()
            mz_plot = mz.detach().cpu().numpy()
            mxy_abs = np.abs(mxy.detach().cpu().numpy())
            tgt_z = target_z.detach().cpu().numpy()
            tgt_xy = target_xy.detach().cpu().numpy()
            pulse_real = np.real(pulse.detach().cpu().numpy())
            pulse_imag = np.imag(pulse.detach().cpu().numpy())
            gradient_for_plot = gradient.detach().cpu().numpy()
            phase = np.unwrap(np.angle(mxy.detach().cpu().numpy()))
            phasemin = np.min(phase)
            phasemax = np.max(phase)
            phase[tgt_xy < 0.5] = np.nan

            self.ax_bottom_left.set_xlim(fmin, fmax)
            self.ax_bottom_right.set_xlim(fmin, fmax)
            self.ax_phase.set_xlim(fmin, fmax)
            self.ax[0].set_xlim(t[0], t[-1])
            self.ax[1].set_xlim(t[0], t[-1])
            self.ax[2].set_xlim(0, epoch + 1)
            self.ax_bottom_left.set_ylim(-0.1, 1.1)
            self.ax_bottom_right.set_ylim(-0.1, 1.1)
            self.ax_phase.set_ylim(phasemin, phasemax)
            self.ax[0].set_ylim(
                (-np.max(np.sqrt(pulse_real**2 + pulse_imag**2)), np.max(np.sqrt(pulse_real**2 + pulse_imag**2)))
            )
            self.ax[2].set_ylim((0.9 * np.min(losses).item(), 1.1 * np.max(losses).item()))
            self.ax[1].set_ylim(
                (-1.1 * np.max(np.abs(gradient_for_plot)).item(), 1.1 * np.max(np.abs(gradient_for_plot)).item())
            )

            self.target_z_plot.set_xdata(fAx)
            self.target_xy_plot.set_xdata(fAx)
            self.mz_plot.set_xdata(fAx)
            self.mxy_plot.set_xdata(fAx)
            self.phase_plot.set_xdata(fAx)
            self.grad_plot.set_xdata(t)
            self.pulse_real_plot.set_xdata(t)
            self.pulse_imag_plot.set_xdata(t)
            self.loss_plot.set_xdata(np.arange(epoch + 1))

            self.target_z_plot.set_ydata(tgt_z)
            self.target_xy_plot.set_ydata(tgt_xy)
            self.mz_plot.set_ydata(mz_plot)
            self.mxy_plot.set_ydata(mxy_abs)
            self.phase_plot.set_ydata(phase)
            self.grad_plot.set_ydata(gradient_for_plot)
            self.pulse_real_plot.set_ydata(pulse_real)
            self.pulse_imag_plot.set_ydata(pulse_imag)
            self.loss_plot.set_ydata(losses)

            self.fig.canvas.draw()
            filename = "results/training.png"
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            plt.savefig(filename, dpi=300)
            plt.show(block=False)
            plt.pause(0.001)

    def print_info(self, epoch, L2_loss, lr):
        self.t1 = time() - self.t0
        self.t0 = time()
        print("Epoch: ", epoch)
        print(f"Loss: {L2_loss.item():.5f}")
        print(f"learning rate: {lr:.6f}")
        print(f"Time: {self.t1:.1f}")
        print("-" * 100)


def get_device():
    device = (
        torch.device("cpu")
        if torch.backends.mps.is_available()
        else torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    )
    print("Device:", device)
    return device


def move_to(tensor_list, device=torch.device("cpu")):
    out_list = []
    for tensor in tensor_list:
        tensor = tensor.to(device)
        out_list.append(tensor)
    return out_list


def pre_train(target_pulse, target_gradient, model, lr=1e-4, thr=1e-3, device=torch.device("cpu")):
    target_pulse = target_pulse.to(device)
    target_gradient = target_gradient.to(device)
    model, optimizer, scheduler, _ = init_training(model, lr=lr, device=device)
    inputs, dt, Nz, sens, B0, tAx, fAx, t_B1 = get_fixed_inputs()
    t_B1 = t_B1.to(device)
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
    plt.figure()
    plt.plot(torch.real(pulse).detach().cpu().numpy(), label="pulse real")
    plt.plot(torch.real(target_pulse).detach().cpu().numpy(), label="target pulse real")
    plt.figure()
    plt.plot(torch.imag(pulse).detach().cpu().numpy(), label="pulse imag")
    plt.plot(torch.imag(target_pulse).detach().cpu().numpy(), label="target pulse imag")
    plt.figure()
    plt.plot(gradient.detach().cpu().numpy(), label="gradient")
    plt.show()
    return model


def init_training(model, lr, device=torch.device("cpu")):
    model = model.to(device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.9, patience=100, min_lr=5e-6)
    losses = []
    return model, optimizer, scheduler, losses


def threshold_loss(x, threshold):
    threshold_loss = torch.max(torch.abs(x)) - threshold
    threshold_loss[threshold_loss < 0] = 0.0
    return threshold_loss**2


def loss_fn(z_profile, xy_profile, tgt_z, tgt_xy, pulse, gradient):
    xy_profile_abs = torch.abs(xy_profile)
    L2_loss_mxy = torch.mean((xy_profile_abs - tgt_xy) ** 2)
    L2_loss_mz = torch.mean((z_profile - tgt_z) ** 2)
    boundary_vals_pulse = torch.abs(pulse[0]) ** 2 + torch.abs(pulse[-1]) ** 2
    gradient_height_loss = threshold_loss(gradient, 50)
    pulse_height_loss = threshold_loss(pulse, 0.016)
    # pulse_height_loss = torch.max(torch.abs(pulse))
    gradient_diff_loss = threshold_loss(torch.diff(gradient.squeeze()), 1)
    phase_diff = torch.diff(torch_unwrap(torch.angle(xy_profile)))
    phase_ddiff = torch.diff(phase_diff)
    phase_ddiff = phase_ddiff[tgt_xy[1:-1] > 1e-6]
    phase_diff_var = torch.var(phase_diff[tgt_xy[:-1] > 1e-6])
    phase_loss = torch.mean(phase_ddiff**2) + phase_diff_var
    print("-" * 50)
    print("LOSSES:")
    print("L2_loss_mxy", L2_loss_mxy.item())
    print("L2_loss_mz", L2_loss_mz.item())
    print("boundary_vals_pulse", boundary_vals_pulse.item() * 100)
    print("gradient_height_loss", gradient_height_loss.item() / 10)
    print("pulse_height_loss", pulse_height_loss.item() * 100)
    print("gradient_diff_loss", gradient_diff_loss.item())
    print("phase_loss", phase_loss.item() * 10)
    print("-" * 50)

    return (
        L2_loss_mxy,
        L2_loss_mz,
        100 * boundary_vals_pulse,
        gradient_height_loss / 10,
        100 * pulse_height_loss,
        gradient_diff_loss,
        10 * phase_loss,
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
    # targets = data_dict["targets"]
    pulse = data_dict["pulse"].detach().cpu()
    gradient = data_dict["gradient"].detach().cpu()
    return pulse, gradient


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
