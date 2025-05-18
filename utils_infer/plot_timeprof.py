from utils_bloch import blochsim_CK_timeprofile
import matplotlib.cm as cm
import torch
import matplotlib.pyplot as plt


def plot_timeprof(gamma, B1, G, pos, sens, B0_list, M0, dt, t_B1, slice_centers):
    mxy, mz = blochsim_CK_timeprofile(gamma, B1, G, pos, sens, B0_list, M0, dt)
    cmap = cm.get_cmap("inferno", len(slice_centers) + 2)
    colors = [cmap(i + 1) for i in range(len(slice_centers))]
    fig, ax = plt.subplots(figsize=(12, 6))
    for i, c in enumerate(slice_centers):
        color = colors[i]
        timeprof = torch.abs(mxy[0, c, :])

        # t_B1_centers = 0.5 * (t_B1[:-1] + t_B1[1:])
        timeprof_diff = torch.diff(timeprof)
        timeprof_diff = torch.max(timeprof_diff, 0 * timeprof_diff)
        com = torch.sum(torch.arange(len(timeprof_diff)) * timeprof_diff) / torch.sum(timeprof_diff)
        com = int(com.item())

        plt.plot(t_B1, timeprof, linewidth=0.9, label=f"Slice: {i+1}\ncenter of mass: {t_B1[com].item():.2f}", color=color)
        # plt.plot(t_B1_centers, timeprof_diff, linewidth=0.8, label=f"Slice: {i+1}\ncenter of mass: {t_B1[com].item():.2f}", color=color)
        plt.plot(t_B1[com], timeprof[com], "o", color=color)
    plt.legend()
    plt.show()
