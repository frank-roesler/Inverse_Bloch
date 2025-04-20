from utils_bloch.setup import *


# TRAINING PARAMETERS:
epochs = 10000
lr = 5e-5
gradient_scale = 200.0  # relative size of gradient to RF pulse
plot_loss_frequency = 10  # plot every n epochs
logging_frequency = 1  # save every n epochs
pre_train_inputs = False  # pre-train on given RF-pulse & gradient

# BLOCH PARAMETERS:
inputs, dt, Nz, sens, B0, tAx, fAx, t_B1 = get_fixed_inputs()

# MODEL PARAMETERS:
# Fourier Series:
n_coeffs = 20

# SIREN:
omega_0 = 2.5

# MLP, SIREN:
hidden_dim = 128
num_layers = 16

# RBFN:
num_centers = 10

# FourierMLP:
num_fourier_features = 21
