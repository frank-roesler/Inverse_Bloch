from utils_bloch.setup import *


# TRAINING PARAMETERS:
epochs = 1000
lr = 1e-3
gradient_scale = 200.0  # relative size of gradient to RF pulse
plot_loss_frequency = 10
logging_frequency = 10  # save every n epochs
pre_train_inputs = False  # pre-train on given RF-pulse & gradient

# BLOCH PARAMETERS:
inputs, dt, Nz, sens, B0, tAx, fAx, t_B1 = get_fixed_inputs(module=torch, device=device)
