from utils_train.nets import *
from blochsim_CK import blochsim_CK
from params import *
from utils_train.utils import *


device = (
    torch.device("mps")
    if torch.backends.mps.is_available()
    else torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
)
print(device)

inputs, dt, Nz, sens, B0, tAx, fAx, t_B1 = get_fixed_inputs(module=torch, device=device)

tMin = t_B1[0].item()
tMax = t_B1[-1].item()
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

model = MLPWithBoundary(output_dim=3, hidden_dim=32, num_layers=2, left_boundary=tMin, right_boundary=tMax).float()
model, optimizer, losses, best_loss, saved, save_timer = init_training(model, lr, device=device)

infoscreen = InfoScreen(output_every=1)

for epoch in range(epochs + 1):
    # Compute frequency profile:
    pulse_gradient = model(t_B1.detach())
    pulse = pulse_gradient[:, 0:1] + 1j * pulse_gradient[:, 1:2]
    gradient = gradient_scale * pulse_gradient[:, 2:]

    print(pulse)
    print(gradient)

    mxy, mz = blochsim_CK(B1=pulse, G=gradient, sens=sens, B0=B0, **inputs)
    loss = loss_fn(mz, mxy, target_z, target_xy)
    print(loss)
    print("-" * 100)
    print(mz)
    print(mxy)

    losses.append(loss.item())
    optimizer.zero_grad()
    print("optimizer.zero_grad done")
    loss.backward()
    print("loss.backward done")
    optimizer.step()
    print("optimizer.step done")

    infoscreen.plot_info(epoch, losses, fAx, t_B1, target_z, target_xy, mz, mxy, pulse, gradient)
    infoscreen.print_info(epoch, loss)
