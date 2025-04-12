from utils_train.nets import *
from blochsim_CK import blochsim_CK
from params import *
from utils_train.utils import *


device = (
    torch.device("mps")
    if torch.backends.mps.is_available()
    else torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
)
print("Device:", device)

inputs, dt, Nz, sens, B0, tAx, fAx, t_B1 = get_fixed_inputs(module=torch, device=device)

sens = sens.detach().requires_grad_(False)
B0 = B0.detach().requires_grad_(False)
tAx = tAx.detach().requires_grad_(False)
fAx = fAx.detach().requires_grad_(False)
t_B1 = t_B1.detach().requires_grad_(False)

target_z, target_xy = get_targets()
target_z = target_z.to(device)
target_xy = target_xy.to(device)
targets_z = target_z.detach().requires_grad_(False)
target_xy = target_xy.detach().requires_grad_(False)

model = FourierMLP(output_dim=3, hidden_dim=8, num_layers=2).float()
model, optimizer, losses = init_training(model, lr, device=device)

model = pre_train(
    target_pulse=torch.from_numpy(inputs["rfmb"]).to(torch.complex64).detach().requires_grad_(False),
    target_gradient=torch.from_numpy(inputs["Gs"]).to(torch.float32).detach().requires_grad_(False),
    model=model,
)

infoscreen = InfoScreen(output_every=1)
trainLogger = TrainLogger(save_every=10)
for epoch in range(epochs + 1):
    pulse_gradient = model(t_B1)
    pulse = pulse_gradient[:, 0:1] + 1j * pulse_gradient[:, 1:2]
    gradient = gradient_scale * pulse_gradient[:, 2:]

    mxy, mz = blochsim_CK(B1=pulse, G=gradient, sens=sens, B0=B0, **inputs)
    L2Loss, DLoss = loss_fn(mz, mxy, target_z, target_xy, pulse, gradient)
    loss = L2Loss + DLoss

    losses.append(loss.item())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    infoscreen.plot_info(epoch, losses, fAx, t_B1, target_z, target_xy, mz, mxy, pulse, gradient)
    infoscreen.print_info(epoch, L2Loss, DLoss)
    trainLogger.log_epoch(epoch, L2Loss, DLoss, losses, model, optimizer, pulse, gradient)
