from utils_bloch.setup import get_fixed_inputs
import numpy as np


# TRAINING PARAMETERS:
epochs = 15000
lr = {"pulse": 1e-4, "gradient": 1e-4}  # learning rate
plot_loss_frequency = 10  # plot every n steps
start_logging = 100  # start logging after n steps
pre_train_inputs = False  # pre-train on given RF-pulse & gradient
suppress_loss_peaks = True
loss_metric = "L2"
loss_weights = {
    "loss_mxy": 1.0,
    "loss_mz": 1.0,
    "boundary_vals_pulse": 100.0,
    "gradient_height_loss": 0.1,
    "pulse_height_loss": 100.0,
    "gradient_diff_loss": 1.0,
    "phase_loss": 100.0,
    "center_of_mass_loss": 0.1,
    "phase_center_loss": 0.01,
}

# BLOCH PARAMETERS:
n_slices = 2
n_b0_values = 5
flip_angle = 17 / 45 * np.pi
pos, dt, dx, Nz, sens, B0, tAx, fAx, t_B1, M0, inputs, freq_offsets_Hz, B0_list = get_fixed_inputs(tfactor=2.0, n_b0_values=n_b0_values)

# MODEL PARAMETERS:
modelname = "MixedModel"  # MLP, SIREN, RBFN, FourierMLP, FourierSeries, ModulatedFourier, MixedModel
model_args = {
    "n_coeffs": 30,  # Fourier Series
    "omega_0": 40,  # SIREN
    "bandwidth": 101,  # ModulatedFourier
    "hidden_dim": 32,  # MLP, SIREN, ModulatedFourier
    "num_layers": 16,  # MLP, SIREN
    "num_centers": 10,  # RBFN
    "center_spacing": 1,  # RBFN
    "num_fourier_features": 51,  # FourierMLP
    "frequency_scale": 100.0,  # FourierMLP
    "gradient_scale": 20.0,  # relative size of gradient to RF pulse
    "positive_gradient": False,
    "tmin": t_B1[0].item(),
    "tmax": t_B1[-1].item(),
}

# PARAMETERS OF THE SCANNER:
# (will appear in as constraints in Loss function)
scanner_params = {
    "max_gradient": 50,  # mT/m
    "max_diff_gradient": 200,  # mT/m/ms
    "max_pulse_amplitude": 0.023,  # mT
}
