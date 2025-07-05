from utils_bloch import blochsim_CK_timeprofile
import matplotlib.cm as cm
import torch
import matplotlib.pyplot as plt
import os


def plot_timeprof(B1, G, fixed_inputs, slice_centers, half_width, path=None):
    cmap = cm.get_cmap("inferno", len(slice_centers[0]) + 2)
    colors = [cmap(i + 1) for i in range(len(slice_centers[0]))]
    mxy, mz = blochsim_CK_timeprofile(B1, G, **fixed_inputs)
    for b, centers in enumerate(slice_centers):
        fig, ax = plt.subplots(figsize=(12, 6))
        for i, c in enumerate(centers):
            color = colors[i]
            timeprof = torch.mean(torch.abs(mxy[b, c - half_width : c + half_width, :]), dim=0)
            com = timeprof[-1] - torch.sum(timeprof)
            com = int(com.item())
            ax.plot(fixed_inputs["t_B1"], timeprof, linewidth=0.9, label=f"Slice: {i+1}\ncenter of mass: {fixed_inputs['t_B1'][com].item():.3f}", color=color)
            ax.plot(fixed_inputs["t_B1"][com], timeprof[com], "o", color=color)
        plt.legend()
        if path is not None:
            fig.savefig(os.path.join(os.path.dirname(path), f"timeprof_{b}.png"), dpi=300)
