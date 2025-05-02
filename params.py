from utils_bloch.setup import get_fixed_inputs
import numpy as np


# TRAINING PARAMETERS:
epochs = 50000
lr = 1e-4
plot_loss_frequency = 10  # plot every n steps
start_logging = 100  # start logging after n steps
pre_train_inputs = False  # pre-train on given RF-pulse & gradient

# BLOCH PARAMETERS:
flip_angle = 0.68 * np.pi/2
pos, dt, dx, Nz, sens, B0, tAx, fAx, t_B1, M0, inputs = get_fixed_inputs()

# MODEL PARAMETERS:
modelname = "MixedModel"  # MLP, SIREN, RBFN, FourierMLP, FourierSeries
model_args = {
    "n_coeffs": 70,#60,#35,#35,  # Fourier Series
    "omega_0": 3.0,  # SIREN (2.5)
    "hidden_dim": 64,#128,#128,  # MLP (128), SIREN (64)
    "num_layers": 16,#16,  # MLP (16), SIREN (16)
    "num_centers": 30,  # RBFN
    "center_spacing": 5,  # RBFN
    "num_fourier_features": 51,  # FourierMLP
    "frequency_scale": 1000.0,  # FourierMLP
    "gradient_scale": 20,#20,#100.0,  # relative size of gradient to RF pulse (20)
    "tmin": t_B1[0].item(),
    "tmax": t_B1[-1].item(),
}
