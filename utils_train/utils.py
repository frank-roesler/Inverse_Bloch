from time import time
import matplotlib.pyplot as plt
import torch
import numpy as np
from params import gradient_scale, get_fixed_inputs


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
        if not epoch % self.save_every == 0:
            return
        if not np.mean(losses[-4:]) < self.best_loss:
            return
        self.best_loss = np.mean(losses[-4:])
        torch.save(self.log, filename)
        print(f"Training log saved to {filename}")


class InfoScreen:
    def __init__(self, output_every=1):
        self.t0 = time()
        self.t1 = time()
        self.output_every = output_every
        self.init_plots()

    def init_plots(self):
        self.fig, self.ax = plt.subplots(1, 4, figsize=(15, 4), constrained_layout=True)
        self.p1 = self.ax[0].plot([0], [1], linewidth=1, label="Target_z")[0]
        self.p2 = self.ax[0].plot([0], [1], linewidth=1, label="M_z")[0]
        self.p3 = self.ax[0].plot([0], [1], linewidth=1, label="M_xy")[0]
        self.p4 = self.ax[0].plot([0], [1], linewidth=0.5, label="Error")[0]
        self.p5 = self.ax[3].plot([0], [1], linewidth=1, label="Gradient")[0]
        self.q = self.ax[1].plot([0], [1e-4], linewidth=1, label="Pulse real")[0]
        self.qq = self.ax[1].plot([0], [1e-4], linewidth=1, label="Pulse imag")[0]
        self.r = self.ax[2].semilogy([0], [1e-1], linewidth=1, label="Loss")[0]
        self.ax[0].legend()
        self.ax[1].legend()
        self.ax[2].legend()
        self.ax[3].legend()
        self.ax[0].set_title("Frequency profile")
        self.ax[1].set_title("Pulse")
        self.ax[2].set_title("Loss")
        self.ax[3].set_title("Gradient")

    def plot_info(self, epoch, losses, fAx, t_B1, target_z, target_xy, mz, mxy, pulse, gradient):
        """plots info curves during training"""
        fmin = -10  # torch.min(fAx).item()
        fmax = 10  # torch.max(fAx).item()
        if epoch % self.output_every == 0:
            t = t_B1.detach().cpu().numpy()
            mz_plot = mz.detach().cpu().numpy()
            mxy_abs = np.abs(mxy.detach().cpu().numpy())
            tgt_z = target_z.detach().cpu().numpy()
            tgt_xy = target_xy.detach().cpu().numpy()
            pulse_real = np.real(pulse.detach().cpu().numpy())
            pulse_imag = np.imag(pulse.detach().cpu().numpy())
            gradient_plot = gradient.detach().cpu().numpy()

            error = np.sqrt((mz_plot - tgt_z) ** 2 + (mxy_abs - tgt_xy) ** 2)

            self.ax[0].set_xlim(fmin, fmax)
            self.ax[1].set_xlim(t[0], t[-1])
            self.ax[3].set_xlim(t[0], t[-1])
            self.ax[2].set_xlim(0, epoch + 1)
            self.ax[0].set_ylim(-0.1, 1.1)
            self.ax[1].set_ylim(
                (
                    -1.1 * np.max(np.sqrt(pulse_real**2 + pulse_imag**2)),
                    1.1 * np.max(np.sqrt(pulse_real**2 + pulse_imag**2)),
                )
            )
            self.ax[2].set_ylim((0.9 * np.min(losses).item(), 1.1 * np.max(losses).item()))
            self.ax[3].set_ylim(
                (-1.1 * np.max(np.abs(gradient_plot)).item(), 1.1 * np.max(np.abs(gradient_plot)).item())
            )

            self.p1.set_xdata(fAx)
            self.p2.set_xdata(fAx)
            self.p3.set_xdata(fAx)
            self.p4.set_xdata(fAx)
            self.p5.set_xdata(t)
            self.p1.set_ydata(tgt_z)
            self.p2.set_ydata(mz_plot)
            self.p3.set_ydata(mxy_abs)
            self.p4.set_ydata(error)
            self.p5.set_ydata(gradient_plot)

            self.q.set_xdata(t)
            self.qq.set_xdata(t)
            self.q.set_ydata(pulse_real)
            self.qq.set_ydata(pulse_imag)

            self.r.set_xdata(np.arange(epoch + 1))
            self.r.set_ydata(losses)

            self.fig.canvas.draw()
            plt.savefig("results/training.png", dpi=300)
            plt.show(block=False)
            plt.pause(0.001)

    def print_info(self, epoch, L2_loss, D_Loss):
        self.t1 = time() - self.t0
        self.t0 = time()
        print("Epoch: ", epoch)
        print(f"L2 Loss: {L2_loss.item():.3f}")
        print(f"D Loss: {D_Loss.item():.3f}")
        print(f"Time: {self.t1:.1f}")
        print("-" * 100)


def pre_train(target_pulse, target_gradient, model):
    device = (
        torch.device("mps")
        if torch.backends.mps.is_available()
        else torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    )
    target_pulse = target_pulse.to(device)
    target_gradient = target_gradient.to(device)
    model, optimizer, losses = init_training(model, lr=1e-2, device=device)
    inputs, dt, Nz, sens, B0, tAx, fAx, t_B1 = get_fixed_inputs(module=torch, device=device)
    for epoch in range(10000):
        model_output = model(t_B1)

        loss_pulse_real = torch.mean((model_output[:, 0:1] - torch.real(target_pulse)) ** 2)
        loss_pulse_imag = torch.mean((model_output[:, 1:2] - torch.imag(target_pulse)) ** 2)
        loss_gradient = torch.mean((gradient_scale * model_output[:, 2:] - target_gradient) ** 2)
        loss = loss_pulse_real + loss_pulse_imag + loss_gradient

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch % 100 == 0:
            print(f"Epoch: {epoch}, Loss: {loss.item():.4f}")
    plt.plot(loss_pulse_real.detach().cpu().numpy(), label="pulse real")
    # plt.plot(loss_pulse_imag.detach().cpu().numpy(), label="pulse imag")
    # plt.plot(loss_gradient.detach().cpu().numpy(), label="gradient")
    plt.show()
    return model


def init_training(model, lr, device=torch.device("cpu")):
    model = model.to(device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    losses = []
    return model, optimizer, losses


def loss_fn(z_profile, xy_profile, tgt_z, tgt_xy, pulse, gradient):
    xy_profile_abs = torch.abs(xy_profile)
    L2_loss = torch.mean((z_profile - tgt_z) ** 2) + torch.mean((xy_profile_abs - tgt_xy) ** 2)
    D_Loss = (
        torch.sqrt(torch.abs(pulse[0]) ** 2 + torch.abs(pulse[-1]) ** 2)
        + torch.sqrt(gradient[0] ** 2 + gradient[-1] ** 2) / gradient_scale
    )
    # H1_loss = torch.mean(findiff(xy_profile-1+target)**2) + torch.mean(findiff(z_profile-target)**2)
    return L2_loss, 10 * D_Loss  # + H1_loss
