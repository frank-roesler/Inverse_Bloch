from time import time
import matplotlib.pyplot as plt
import torch
import numpy as np
from params import gradient_scale, get_fixed_inputs
import os


def get_device():
    device = (
        torch.device("mps")
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


class TrainLogger:
    def __init__(self, save_every=1):
        self.log = {}
        self.save_every = save_every
        self.best_loss = np.inf

    def log_epoch(self, epoch, L2_loss, D_loss, losses, model, optimizer, pulse, gradient):
        self.log["epoch"] = epoch
        self.log["L2_loss"] = L2_loss.item()
        self.log["D_loss"] = D_loss.item()
        self.log["losses"] = losses
        self.log["model"] = model
        self.log["optimizer"] = optimizer
        self.log["pulse"] = pulse
        self.log["gradient"] = gradient
        self.save(epoch, losses)

    def save(self, epoch, losses, filename="results/train_log.pt"):
        if epoch == 0:
            return
        if not epoch % self.save_every == 0:
            return
        if not np.mean(losses[-4:]) < self.best_loss:
            return
        self.best_loss = np.mean(losses[-4:])
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        torch.save(self.log, filename)
        print(f"Training log saved to {filename}")


class InfoScreen:
    def __init__(self, output_every=1):
        self.t0 = time()
        self.t1 = time()
        self.output_every = output_every
        self.init_plots()

    def init_plots(self):
        self.fig, self.ax = plt.subplots(2, 3, figsize=(12, 7), constrained_layout=False)
        self.ax[1, 0].remove()
        self.ax[1, 1].remove()
        self.ax[1, 2].remove()
        self.big_ax = self.fig.add_subplot(2, 1, 2)
        self.target_z_plot = self.big_ax.plot([0], [1], linewidth=1, label="Target_z")[0]
        self.mz_plot = self.big_ax.plot([0], [1], linewidth=1, label="M_z")[0]
        self.mxy_plot = self.big_ax.plot([0], [1], linewidth=1, label="M_xy")[0]
        self.error_plot = self.big_ax.plot([0], [1], linewidth=0.5, label="Error (z and xy)")[0]
        self.grad_plot = self.ax[0, 1].plot([0], [1], linewidth=1, label="Gradient")[0]
        self.pulse_real_plot = self.ax[0, 0].plot([0], [1e-4], linewidth=1, label="Pulse real")[0]
        self.pulse_imag_plot = self.ax[0, 0].plot([0], [1e-4], linewidth=1, label="Pulse imag")[0]
        self.loss_plot = self.ax[0, 2].semilogy([0], [1e-1], linewidth=1, label="Loss")[0]
        self.big_ax.legend()
        self.ax[0, 0].legend()
        self.ax[0, 2].legend()
        self.ax[0, 1].legend()
        self.big_ax.set_title("Frequency profile")
        self.ax[0, 0].set_title("Pulse")
        self.ax[0, 2].set_title("Loss")
        self.ax[0, 1].set_title("Gradient")

    def plot_info(self, epoch, losses, fAx, t_B1, target_z, target_xy, mz, mxy, pulse, gradient):
        """plots info curves during training"""
        fmin = -20  # torch.min(fAx).item()
        fmax = 20  # torch.max(fAx).item()
        if epoch % self.output_every == 0:
            t = t_B1.detach().cpu().numpy()
            mz_plot = mz.detach().cpu().numpy()
            mxy_abs = np.abs(mxy.detach().cpu().numpy())
            tgt_z = target_z.detach().cpu().numpy()
            tgt_xy = target_xy.detach().cpu().numpy()
            pulse_real = np.real(pulse.detach().cpu().numpy())
            pulse_imag = np.imag(pulse.detach().cpu().numpy())
            gradient_for_plot = gradient.detach().cpu().numpy()

            error = np.sqrt((mz_plot - tgt_z) ** 2 + (mxy_abs - tgt_xy) ** 2)

            self.big_ax.set_xlim(fmin, fmax)
            self.ax[0, 0].set_xlim(t[0], t[-1])
            self.ax[0, 1].set_xlim(t[0], t[-1])
            self.ax[0, 2].set_xlim(0, epoch + 1)
            self.big_ax.set_ylim(-0.1, 1.1)
            self.ax[0, 0].set_ylim(
                (
                    -1.1 * np.max(np.sqrt(pulse_real**2 + pulse_imag**2)),
                    1.1 * np.max(np.sqrt(pulse_real**2 + pulse_imag**2)),
                )
            )
            self.ax[0, 2].set_ylim((0.9 * np.min(losses).item(), 1.1 * np.max(losses).item()))
            self.ax[0, 1].set_ylim(
                (-1.1 * np.max(np.abs(gradient_for_plot)).item(), 1.1 * np.max(np.abs(gradient_for_plot)).item())
            )

            self.target_z_plot.set_xdata(fAx)
            self.mz_plot.set_xdata(fAx)
            self.mxy_plot.set_xdata(fAx)
            self.error_plot.set_xdata(fAx)
            self.grad_plot.set_xdata(t)
            self.pulse_real_plot.set_xdata(t)
            self.pulse_imag_plot.set_xdata(t)
            self.loss_plot.set_xdata(np.arange(epoch + 1))

            self.target_z_plot.set_ydata(tgt_z)
            self.mz_plot.set_ydata(mz_plot)
            self.mxy_plot.set_ydata(mxy_abs)
            self.error_plot.set_ydata(error)
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

    def print_info(self, epoch, L2_loss, D_Loss):
        self.t1 = time() - self.t0
        self.t0 = time()
        print("Epoch: ", epoch)
        print(f"L2 Loss: {L2_loss.item():.4f}")
        print(f"D Loss: {D_Loss.item():.4f}")
        print(f"Time: {self.t1:.1f}")
        print("-" * 100)


def pre_train(target_pulse, target_gradient, model, lr=1e-4, thr=1e-3, device=torch.device("cpu")):
    target_pulse = target_pulse.to(device)
    target_gradient = target_gradient.to(device)
    model, optimizer, losses = init_training(model, lr=lr, device=device)
    inputs, dt, Nz, sens, B0, tAx, fAx, t_B1 = get_fixed_inputs()
    t_B1 = t_B1.to(device)
    loss = torch.inf
    epoch = 0
    while loss > thr:
        epoch += 1
        model_output = model(t_B1)

        loss_pulse_real = torch.mean((model_output[:, 0:1] - torch.real(target_pulse)) ** 2)
        loss_pulse_imag = torch.mean((model_output[:, 1:2] - torch.imag(target_pulse)) ** 2)
        loss_gradient = torch.mean((gradient_scale * model_output[:, 2:] - target_gradient) ** 2)
        loss = loss_pulse_real + loss_pulse_imag + loss_gradient / gradient_scale

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch % 1000 == 0:
            print(f"Epoch: {epoch}, Loss: {loss.item():.6f}")
    # plt.figure()
    # plt.plot(model_output[:, 0:1].detach().cpu().numpy(), label="pulse real")
    # plt.plot(torch.real(target_pulse).detach().cpu().numpy(), label="target pulse real")
    # plt.figure()
    # plt.plot(model_output[:, 1:2].detach().cpu().numpy(), label="pulse imag")
    # plt.plot(torch.imag(target_pulse).detach().cpu().numpy(), label="target pulse imag")
    # plt.figure()
    # plt.plot(model_output[:, 2:].detach().cpu().numpy(), label="gradient")
    # plt.show()
    return model


def init_training(model, lr, device=torch.device("cpu")):
    model = model.to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    losses = []
    return model, optimizer, losses


def loss_fn(z_profile, xy_profile, tgt_z, tgt_xy, pulse, gradient):
    xy_profile_abs = torch.abs(xy_profile)
    L2_loss = torch.mean((z_profile - tgt_z) ** 2) + torch.mean((xy_profile_abs - tgt_xy) ** 2)
    boundary_vals_pulse = torch.abs(pulse[0]) ** 2 + torch.abs(pulse[-1]) ** 2
    boundary_vals_grad = gradient[0] ** 2 + gradient[-1] ** 2
    D_Loss = boundary_vals_pulse + boundary_vals_grad / gradient_scale**2  # Dirichlet loss
    # H1_loss = torch.mean(findiff(xy_profile-1+target)**2) + torch.mean(findiff(z_profile-target)**2) # NOT IMPLEMENTED
    return L2_loss, D_Loss  # + H1_loss
