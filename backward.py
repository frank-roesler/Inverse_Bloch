from utils_train.nets import *
from utils_train.utils import *
from utils_bloch.blochsim_CK import blochsim_CK
from params import *
from buildTarget import buildTarget

device = get_device()
_, _, target_z, target_xy = get_test_targets()
# targAbsMxy, targPhMxy, targMz = buildTarget(mxy, mz, B1, dt, bwTB, sharpnessTB, phSlopReduction)
B0, sens, t_B1, inputs["pos"], target_z, target_xy = move_to(
    (B0, sens, t_B1, inputs["pos"], target_z, target_xy), device
)

# model = MLP(output_dim=3, hidden_dim=64, num_layers=8).float()
model = FourierSeries(
    n_coeffs=n_coeffs, tmin=t_B1[0].item(), tmax=t_B1[-1].item()
).float()
model, optimizer, losses = init_training(model, lr, device=device)

if pre_train_inputs:
    B1 = (
        torch.from_numpy(inputs["rfmb"])
        .to(torch.complex64)
        .detach()
        .requires_grad_(False)
        .to(device)
    )
    G = (
        torch.from_numpy(inputs["Gs"])
        .to(torch.float32)
        .detach()
        .requires_grad_(False)
        .to(device)
    )
    model = pre_train(
        target_pulse=B1,
        target_gradient=G,
        model=model,
        lr=1e-4,
        thr=0.0006,
        device=device,
    )

infoscreen = InfoScreen(output_every=plot_loss_frequency)
trainLogger = TrainLogger(save_every=logging_frequency)
for epoch in range(epochs + 1):
    pulse_gradient = model(t_B1)
    pulse = pulse_gradient[:, 0:1] + 1j * pulse_gradient[:, 1:2]
    gradient = gradient_scale * pulse_gradient[:, 2:]

    mxy, mz = blochsim_CK(B1=pulse, G=gradient, sens=sens, B0=B0, **inputs)
    L2Loss, DLoss = loss_fn(mz, mxy, target_z, target_xy, pulse, gradient)
    loss = L2Loss + DLoss  # DLoss can be omitted when training FourierSeries

    losses.append(loss.item())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    infoscreen.plot_info(
        epoch, losses, fAx, t_B1, target_z, target_xy, mz, mxy, pulse, gradient
    )
    infoscreen.print_info(epoch, L2Loss, DLoss)
    trainLogger.log_epoch(
        epoch, L2Loss, DLoss, losses, model, optimizer, pulse, gradient
    )
