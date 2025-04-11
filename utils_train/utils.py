from time import time
import matplotlib.pyplot as plt
import torch
import numpy as np
from params import gradient_scale


class InfoScreen:
    def __init__(self, output_every=1):
        self.t0 = time()
        self.t1 = time()
        self.output_every = output_every
        self.init_plots()

    def init_plots(self):
        self.fig, self.ax = plt.subplots(1, 4, figsize=(15, 4), constrained_layout=True)
        self.p1 = self.ax[0].plot([0], [1], linewidth=1, label="Target")[0]
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
        fmin = torch.min(fAx).item()
        fmax = torch.max(fAx).item()
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
            self.ax[0].set_ylim(0,1)
            self.ax[1].set_ylim(
                (
                    -1.1*np.max(np.sqrt(pulse_real**2+pulse_imag**2)),
                    1.1*np.max(np.sqrt(pulse_real**2+pulse_imag**2))
                )
            )
            self.ax[2].set_ylim((0.9 * np.min(losses).item(), 1.1 * np.max(losses).item()))
            self.ax[3].set_ylim((-1.1 * np.max(np.abs(gradient_plot)).item(), 1.1 * np.max(np.abs(gradient_plot)).item()))

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
            self.q.set_ydata(pulse_real)
            self.qq.set_ydata(pulse_imag)

            self.r.set_xdata(np.arange(epoch + 1))
            self.r.set_ydata(losses)

            self.fig.canvas.draw()
            plt.show(block=False)
            plt.pause(0.001)

    def print_info(self, epoch, L2_loss,D_Loss):
        self.t1 = time() - self.t0
        self.t0 = time()
        print("Epoch: ", epoch)
        print(f"L2 Loss: {L2_loss.item():.3f}")
        print(f"D Loss: {D_Loss.item():.3f}")
        print(f"Time: {self.t1:.1f}")
        print("-" * 100)


def init_training(model_pulse, lr, device=torch.device("cpu")):
    model_pulse = model_pulse.to(device)
    optimizer_pulse = torch.optim.Adam(model_pulse.parameters(), lr=lr)
    losses = []
    best_loss = np.inf
    saved = False
    save_timer = 0
    return model_pulse, optimizer_pulse, losses, best_loss, saved, save_timer


def loss_fn(z_profile, xy_profile, tgt_z, tgt_xy, pulse, gradient):
    xy_profile_abs = torch.abs(xy_profile)
    L2_loss = torch.mean((z_profile - tgt_z) ** 2) + torch.mean((xy_profile_abs - tgt_xy) ** 2)
    D_Loss = torch.sqrt(torch.abs(pulse[0])**2 + torch.abs(pulse[-1])**2) + torch.sqrt(gradient[0]**2 + gradient[-1]**2)/gradient_scale
    # H1_loss = torch.mean(findiff(xy_profile-1+target)**2) + torch.mean(findiff(z_profile-target)**2)
    return L2_loss, D_Loss  # + H1_loss
