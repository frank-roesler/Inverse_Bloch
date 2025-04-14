from utils_bloch.setup import *


# TRAINING PARAMETERS:
epochs = 10000
lr = 1e-3
gradient_scale = 200.0  # relative size of gradient to RF pulse
plot_loss_frequency = 1  # plot every n epochs
logging_frequency = 100  # save every n epochs
pre_train_inputs = False  # pre-train on given RF-pulse & gradient

# BLOCH PARAMETERS:
inputs, dt, Nz, sens, B0, tAx, fAx, t_B1 = get_fixed_inputs()

# MODEL PARAMETERS:
# Fourier Series:
n_coeffs = 31

# SIREN:
omega_0 = 6

# MLP, SIREN:
hidden_dim = 64
num_layers = 8

# RBFN:
num_centers = 10

# FourierMLP:
num_fourier_features = 21
