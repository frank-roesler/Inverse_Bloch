from utils_train.nets import get_model
from utils_train.utils import *
from utils_bloch.blochsim_CK import blochsim_CK
from utils_bloch.blochsim_batch import blochsim_CK_batch
from utils_bloch.setup import get_targets, get_smooth_targets
from params import *


device = get_device()
# target_z, target_xy = get_targets(theta=flip_angle)
target_z, target_xy = get_smooth_targets(theta=flip_angle, smoothness=2.0, function=torch.sigmoid, n_targets=n_slices)

gam = 267522.1199722082
gam_hz_mt = gam / (2 * np.pi)
# freq_offsets_Hz = torch.linspace(-297.3 * 4.7, 0.0, 5)
freq_offsets_Hz = torch.Tensor([-297.3 * 4.7 / 2])
B0_freq_offsets_mT = freq_offsets_Hz / gam_hz_mt
B0_vals = []
for ff in range(len(freq_offsets_Hz)):
    B0_vals.append(B0 + B0_freq_offsets_mT[ff])
B0_list = torch.stack(B0_vals, dim=0).to(torch.float32)

B0, B0_list, M0, sens, t_B1, pos, target_z, target_xy = move_to((B0, B0_list, M0, sens, t_B1, pos, target_z, target_xy), device)

model = get_model(modelname, **model_args)
model, optimizer, scheduler, losses = init_training(model, lr, device=device)

if pre_train_inputs:
    B1 = torch.from_numpy(inputs["rfmb"]).to(torch.complex64).detach().requires_grad_(False).to(device)
    G = torch.from_numpy(inputs["Gs"]).to(torch.float32).detach().requires_grad_(False).to(device)
    model = pre_train(target_pulse=B1, target_gradient=G, model=model, lr=1e-4, thr=1e-5, device=device)

infoscreen = InfoScreen(output_every=plot_loss_frequency)

trainLogger = TrainLogger(start_logging=start_logging)


def f(x, threshold=1):
    return 1 / (1 + 10 * x**2 / (threshold**2 + x))


for epoch in range(epochs + 1):
    pulse, gradient = model(t_B1)
    mxy, mz = blochsim_CK(B1=pulse, G=gradient, pos=pos, sens=sens, B0=B0 + B0_freq_offsets_mT[0], M0=M0, dt=dt)
    # mxy, mz = blochsim_CK_batch(B1=pulse, G=gradient, pos=pos, sens=sens, B0_list=B0_list, M0=M0, dt=dt)

    loss = torch.tensor([0.0], device=device)
    for ff in range(len(freq_offsets_Hz)):
        (loss_mxy, loss_mz, boundary_vals_pulse, gradient_height_loss, pulse_height_loss, gradient_diff_loss, phase_loss) = loss_fn(
            mz, mxy, target_z, target_xy, pulse, gradient
        )
        # ) = loss_fn(mz[ff, :], mxy[ff, :], target_z, target_xy, pulse, gradient)
        loss += loss_mxy + loss_mz + gradient_height_loss + gradient_diff_loss + pulse_height_loss + boundary_vals_pulse + phase_loss

    lossItem = loss.item()
    losses.append(lossItem)
    optimizer.zero_grad()
    loss.backward()

    total_grad_norm = 0.0
    for param in model.parameters():
        if param.grad is not None:
            param_norm = param.grad.data.norm(2)
            total_grad_norm += param_norm.item() ** 2
    total_grad_norm = total_grad_norm**0.5
    print(f"Total Gradient Norm: {total_grad_norm}")
    factor = f(total_grad_norm, 100)
    for param in model.parameters():
        if param.grad is not None:
            param.grad.data.mul_(factor)
    print(f"Gradient norm scaling down by {factor}")

    optimizer.step()
    scheduler.step(lossItem)

    new_optimum = trainLogger.log_epoch(
        epoch,
        loss,
        boundary_vals_pulse,
        losses,
        model,
        optimizer,
        pulse,
        gradient,
        (pos, dt, dx, Nz, sens, B0, tAx, fAx, t_B1, M0, inputs),
        flip_angle,
        loss_metric,
        {"target_z": target_z, "target_xy": target_xy},
        {"tAx": tAx, "fAx": fAx, "t_B1": t_B1},
    )
    infoscreen.plot_info(epoch, losses, pos, t_B1, target_z, target_xy, mz, mxy, pulse, gradient, new_optimum)
    # infoscreen.plot_info(
    #     epoch, losses, pos, t_B1, target_z, target_xy, mz[0, :], mxy[0, :], pulse, gradient, new_optimum
    # )
    infoscreen.print_info(epoch, lossItem, optimizer)
