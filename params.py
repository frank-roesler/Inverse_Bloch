from utils_bloch.setup import get_fixed_inputs
import numpy as np


# TRAINING PARAMETERS:
epochs = 10000
lr = {"pulse": 1e-4, "gradient": 5e-4}  # learning rate
plot_loss_frequency = 1  # plot every n steps
start_logging = 100  # start logging after n steps
pre_train_inputs = False  # pre-train on given RF-pulse & gradient
loss_metric = "L2"

# BLOCH PARAMETERS:
n_slices = 4
n_b0_values = 5
flip_angle = 17 / 45 * np.pi
pos, dt, dx, Nz, sens, B0, tAx, fAx, t_B1, M0, inputs, freq_offsets_Hz, B0_list = get_fixed_inputs(
    tfactor=2.0, n_b0_values=n_b0_values
)

# MODEL PARAMETERS:
modelname = "MixedModel"  # MLP, SIREN, RBFN, FourierMLP, FourierSeries, ModulatedFourier, MixedModel
model_args = {
    "n_coeffs": 20,  # Fourier Series
    "omega_0": 40,  # SIREN
    "bandwidth": 101,  # ModulatedFourier
    "hidden_dim": 32,  # MLP, SIREN, ModulatedFourier
    "num_layers": 16,  # MLP, SIREN
    "num_centers": 10,  # RBFN
    "center_spacing": 1,  # RBFN
    "num_fourier_features": 51,  # FourierMLP
    "frequency_scale": 100.0,  # FourierMLP
    "gradient_scale": 100.0,  # relative size of gradient to RF pulse
    "positive_gradient": False,
    "tmin": t_B1[0].item(),
    "tmax": t_B1[-1].item(),
}

# PARAMETERS OF THE SCANNER:
# (will appear in as constraints in Loss function)
scanner_params = {
    "max_gradient": 50,
    "max_diff_gradient": 200,
    "max_pulse_amplitude": 0.023,
}  # mT/m  # mT/m/ms  # mT
