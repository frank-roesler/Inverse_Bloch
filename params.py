from utils_bloch.setup import *


# TRAINING PARAMETERS:
epochs = 10000
lr = 2e-5
plot_loss_frequency = 10  # plot every n epochs
logging_frequency = 1  # save every n epochs
pre_train_inputs = False  # pre-train on given RF-pulse & gradient

# BLOCH PARAMETERS:
inputs, dt, dx, Nz, sens, B0, tAx, fAx, t_B1 = get_fixed_inputs()

# MODEL PARAMETERS:
modelname = "MixedModel"  # MLP, SIREN, RBFN, FourierMLP, FourierSeries
model_args = {
    "n_coeffs": 30,  # Fourier Series
    "omega_0": 2.5,  # SIREN
    "hidden_dim": 256,  # MLP, SIREN
    "num_layers": 32,  # MLP, SIREN
    "num_centers": 10,  # RBFN
    "center_spacing": 1,  # RBFN
    "num_fourier_features": 51,  # FourierMLP
    "frequency_scale": 500.0,  # FourierMLP
    "gradient_scale": 100.0,  # relative size of gradient to RF pulse
    "tmin": t_B1[0].item(),
    "tmax": t_B1[-1].item(),
}
