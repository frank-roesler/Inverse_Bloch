from utils_bloch.setup import get_fixed_inputs
import numpy as np


# TRAINING PARAMETERS:
epochs = 10000
lr = 5e-4
plot_loss_frequency = 10  # plot every n steps
start_logging = 100  # start logging after n steps
pre_train_inputs = False  # pre-train on given RF-pulse & gradient

# BLOCH PARAMETERS:
flip_angle = 17.0 / 45.0 * np.pi
pos, dt, dx, Nz, sens, B0, tAx, fAx, t_B1, M0, inputs = get_fixed_inputs()

# MODEL PARAMETERS:
modelname = "ModulatedFourier"  # MLP, SIREN, RBFN, FourierMLP, FourierSeries, ModulatedFourier
model_args = {
    "n_coeffs": 60,  # Fourier Series
    "omega_0": 2.5,  # SIREN
    "bandwidth": 50,  # ModulatedFourier
    "hidden_dim": 128,  # MLP, SIREN, ModulatedFourier
    "num_layers": 32,  # MLP, SIREN
    "num_centers": 10,  # RBFN
    "center_spacing": 1,  # RBFN
    "num_fourier_features": 51,  # FourierMLP
    "frequency_scale": 100.0,  # FourierMLP
    "gradient_scale": 100.0,  # relative size of gradient to RF pulse
    # "tmin": t_B1[0].item(),
    # "tmax": t_B1[-1].item(),
}
