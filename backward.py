from utils_train.nets import get_model
from utils_train.utils import *
from utils_bloch.blochsim_CK import blochsim_CK
from utils_bloch.blochsim_batch import blochsim_CK_batch
from utils_bloch.setup import get_targets, get_smooth_targets
from params import *


device = get_device()
target_z, target_xy, slice_centers, slice_half_width = get_smooth_targets(
    theta=flip_angle, smoothness=2.0, function=torch.sigmoid, n_targets=n_slices
)

B0, B0_list, M0, sens, t_B1, pos, target_z, target_xy = move_to((B0, B0_list, M0, sens, t_B1, pos, target_z, target_xy), device)

model = get_model(modelname, **model_args)
model_old = get_model(modelname, **model_args)
model, optimizer, scheduler, losses = init_training(model, lr, device=device)


if pre_train_inputs:
    B1, G, axes, targets = load_data("C:/Users/frank/Dropbox/090525_Mixed_4Slices/train_log.pt")
    model = pre_train(target_pulse=B1, target_gradient=G, model=model, lr={"pulse": 1e-4, "gradient": 2e-4}, thr=1e-5, device=device)

infoscreen = InfoScreen(output_every=plot_loss_frequency)
trainLogger = TrainLogger(start_logging=start_logging)

torch.autograd.set_detect_anomaly(True)
for epoch in range(epochs + 1):
    pulse, gradient = model(t_B1)
    mxy, mz, mxy_t_integrated = blochsim_CK_batch(
        B1=pulse,
        G=gradient,
        pos=pos,
        sens=sens,
        dx=dx,
        B0_list=B0_list,
        M0=M0,
        slice_centers=slice_centers,
        slice_half_width=slice_half_width,
        dt=dt,
    )

    (
        loss_mxy,
        loss_mz,
        boundary_vals_pulse,
        gradient_height_loss,
        pulse_height_loss,
        gradient_diff_loss,
        phase_loss,
        center_of_mass_loss,
    ) = loss_fn(
        mz,
        mxy,
        target_z,
        target_xy,
        pulse,
        gradient,
        mxy_t_integrated,
        1000 * dt,
        t_B1=t_B1,
        scanner_params=scanner_params,
        loss_weights=loss_weights,
        metric=loss_metric,
        verbose=True,
    )
    loss = loss_mxy + loss_mz + gradient_height_loss + gradient_diff_loss + pulse_height_loss + boundary_vals_pulse + phase_loss

    lossItem = loss.item()
    losses.append(lossItem)
    optimizer.zero_grad()
    loss.backward()
    if suppress_loss_peaks:
        # model = regularize_model_gradients(model)
        if epoch > 100 and losses[-1] > 2 * losses[-2]:
            model.load_state_dict(model_old.state_dict())
            print("EXPLOSION!!! MODEL RESET")
        else:
            model_old.load_state_dict(model.state_dict())
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
    infoscreen.plot_info(
        epoch,
        losses,
        pos,
        t_B1,
        target_z,
        target_xy,
        mz,
        mxy,
        mxy_t_integrated,
        pulse,
        gradient,
        new_optimum,
    )
    infoscreen.print_info(epoch, lossItem, optimizer)
