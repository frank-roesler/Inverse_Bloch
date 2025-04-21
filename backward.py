from utils_train.nets import get_model
from utils_train.utils import *
from utils_bloch.blochsim_CK import blochsim_CK
from params import *

device = get_device()
target_z, target_xy, _, _ = get_test_targets()
# target_xy, _, target_z = get_smooth_targets(smoothness=1)

B0, sens, t_B1, inputs["pos"], target_z, target_xy = move_to(
    (B0, sens, t_B1, inputs["pos"], target_z, target_xy), device
)

model = get_model("MLP", **model_args)
model, optimizer, scheduler, losses = init_training(model, lr, device=device)

if pre_train_inputs:
    B1 = torch.from_numpy(inputs["rfmb"]).to(torch.complex64).detach().requires_grad_(False).to(device)
    G = torch.from_numpy(inputs["Gs"]).to(torch.float32).detach().requires_grad_(False).to(device)
    model = pre_train(target_pulse=B1, target_gradient=G, model=model, lr=1e-4, thr=1e-5, device=device)

infoscreen = InfoScreen(output_every=plot_loss_frequency)
trainLogger = TrainLogger(save_every=logging_frequency)

for epoch in range(epochs + 1):
    pulse, gradient = model(t_B1)
    mxy, mz = blochsim_CK(B1=pulse, G=gradient, sens=sens, B0=B0, **inputs)

    L2_loss, boundary_vals_pulse, gradient_height_loss, pulse_height_loss, gradient_diff_loss = loss_fn(
        mz, mxy, target_z, target_xy, pulse, gradient
    )
    loss = L2_loss + gradient_height_loss + gradient_diff_loss + pulse_height_loss

    losses.append(loss.item())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    scheduler.step(loss.item())

    infoscreen.plot_info(epoch, losses, fAx, t_B1, target_z, target_xy, mz, mxy, pulse, gradient)
    infoscreen.print_info(epoch, L2_loss, optimizer.param_groups[0]["lr"])
    trainLogger.log_epoch(
        epoch,
        L2_loss,
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
