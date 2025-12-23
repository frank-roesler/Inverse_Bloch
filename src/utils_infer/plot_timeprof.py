from utils_bloch import blochsim_CK_timeprofile
import matplotlib.cm as cm
import torch
import matplotlib.pyplot as plt
import os


def plot_timeprof(B1, G, fixed_inputs, target_xy, path=None):
    n_b0_values = target_xy.shape[0]

    slice_mask = target_xy > 0.5
    slice_centers = torch.sum(slice_mask * torch.arange(slice_mask.shape[1]).unsqueeze(-1), dim=1) / slice_mask.sum(dim=1)
    slice_centers = slice_centers.long()
    half_width = (torch.sum(slice_mask[0, :, :].sum(dim=-1)) / slice_mask.shape[-1] / 2).long().item()

    for b in range(n_b0_values):
        freq_offsets_ppm = torch.linspace(-4.7, 0.0, n_b0_values)

    cmap = cm.get_cmap("inferno", len(slice_centers[0]) + 2)
    colors = [cmap(i + 1) for i in range(len(slice_centers[0]))]
    mxy, mz = blochsim_CK_timeprofile(B1, G, **fixed_inputs)
    for b in range(slice_centers.shape[0]):
        centers = slice_centers[b, :]
        fig, ax = plt.subplots(figsize=(12, 6))
        fig.suptitle(f"B0: {freq_offsets_ppm[b].item():.1f} ppm", fontsize=16)
        for i, c in enumerate(centers):
            color = colors[i]
            timeprof = torch.mean(torch.abs(mxy[b, c - half_width : c + half_width, :]), dim=0)
            com = timeprof[-1] - torch.sum(timeprof)
            comInt = int(com.item())
            ax.plot(fixed_inputs["t_B1"], timeprof, linewidth=0.9, label=f"Slice: {i+1}\ncenter of mass: {fixed_inputs['t_B1'][comInt].item():.3f}", color=color)
            ax.plot(fixed_inputs["t_B1"][comInt], timeprof[comInt], "o", color=color)
        plt.legend()
        if path is not None:
            fig.savefig(os.path.join(os.path.dirname(path), f"timeprof_{b}.png"), dpi=300)
