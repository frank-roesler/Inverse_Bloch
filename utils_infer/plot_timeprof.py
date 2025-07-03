from utils_bloch import blochsim_CK_timeprofile
import matplotlib.cm as cm
import torch
import matplotlib.pyplot as plt
import os


def plot_timeprof(gamma, B1, G, fixed_inputs, slice_centers, path=None, fig=None, ax=None):
    mxy, mz = blochsim_CK_timeprofile(gamma, B1, G, fixed_inputs["pos"], fixed_inputs["sens"], fixed_inputs["B0_list"], fixed_inputs["M0"], fixed_inputs["dt_num"])
    cmap = cm.get_cmap("inferno", len(slice_centers[0]) + 2)
    colors = [cmap(i + 1) for i in range(len(slice_centers[0]))]
    if ax == None or fig == None:
        fig, ax = plt.subplots(figsize=(12, 6))
    for i, c in enumerate(slice_centers[0]):
        color = colors[i]
        timeprof = torch.abs(mxy[0, c, :])

        # t_B1_centers = 0.5 * (t_B1[:-1] + t_B1[1:])
        com = timeprof[-1] - torch.sum(timeprof)
        com = int(com.item())

        ax.plot(fixed_inputs["t_B1"], timeprof, linewidth=0.9, label=f"Slice: {i+1}\ncenter of mass: {fixed_inputs['t_B1'][com].item():.2f}", color=color)
        # plt.plot(fixed_inputs['t_B1_centers'], timeprof_diff, linewidth=0.8, label=f"Slice: {i+1}\ncenter of mass: {fixed_inputs['t_B1'][com].item():.2f}", color=color)
        ax.plot(fixed_inputs["t_B1"][com], timeprof[com], "o", color=color)
    plt.legend()
    if path is not None:
        fig.savefig(os.path.join(os.path.dirname(path), f"timeprof.png"), dpi=300)
