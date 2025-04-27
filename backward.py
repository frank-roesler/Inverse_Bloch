from utils_train.nets import get_model
from utils_train.utils import *
from utils_bloch.blochsim_CK import blochsim_CK
from params import *


device = get_device()
target_z, target_xy = get_test_targets()

gam = 267522.1199722082
gam_hz_mt = gam / (2 * np.pi)
freq_offset = -297.3 * 4.7 / gam_hz_mt / 2  # center freq offset

B0, M0, sens, t_B1, pos, target_z, target_xy = move_to((B0, M0, sens, t_B1, pos, target_z, target_xy), device)

model = get_model(modelname, **model_args)
model, optimizer, scheduler, losses = init_training(model, lr, device=device)

if pre_train_inputs:
    B1 = torch.from_numpy(inputs["rfmb"]).to(torch.complex64).detach().requires_grad_(False).to(device)
    G = torch.from_numpy(inputs["Gs"]).to(torch.float32).detach().requires_grad_(False).to(device)
    model = pre_train(target_pulse=B1, target_gradient=G, model=model, lr=1e-4, thr=1e-5, device=device)

infoscreen = InfoScreen(output_every=plot_loss_frequency)
trainLogger = TrainLogger(save_every=logging_frequency)

for epoch in range(epochs + 1):
    pulse, gradient = model(t_B1)
    mxy, mz = blochsim_CK(B1=pulse, G=gradient, pos=pos, sens=sens, B0=B0 + freq_offset, M0=M0, dt=dt)

    (
        L2_loss_mxy,
        L2_loss_mz,
        boundary_vals_pulse,
        gradient_height_loss,
        pulse_height_loss,
        gradient_diff_loss,
        phase_loss,
    ) = loss_fn(mz, mxy, target_z, target_xy, pulse, gradient)
    loss = (
        L2_loss_mxy
        + L2_loss_mz
        + gradient_height_loss
        + gradient_diff_loss
        + pulse_height_loss
        + boundary_vals_pulse
        + phase_loss
    )

    losses.append(loss.item())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    scheduler.step(loss.item())

    infoscreen.plot_info(epoch, losses, pos, t_B1, target_z, target_xy, mz, mxy, pulse, gradient)
    infoscreen.print_info(epoch, loss, optimizer.param_groups[0]["lr"])
    trainLogger.log_epoch(
        epoch,
        loss,
        boundary_vals_pulse,
        losses,
        model,
        optimizer,
        pulse,
        gradient,
        inputs,
        {"target_z": target_z, "target_xy": target_xy},
        {"tAx": tAx, "fAx": fAx, "t_B1": t_B1},
    )
