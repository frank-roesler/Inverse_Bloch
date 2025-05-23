from utils_bloch.setup import get_fixed_inputs
import numpy as np


# TRAINING PARAMETERS:
start_epoch = 0
epochs = 10000
resume_from_path = "/Users/frankrosler/Desktop/PhD/Python/Inverse_Bloch/results/160525_Mixed_smooth_1Slice/train_log.pt"
lr = {"pulse": 2e-4, "gradient": 1e-4}  # learning rate
plot_loss_frequency = 1  # plot every n steps
start_logging = 200  # start logging after n steps
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
    "phase_loss": 10.0,
}

# BLOCH PARAMETERS:
n_slices = 1
n_b0_values = 1
flip_angle = 17 / 45 * np.pi
fixed_inputs = get_fixed_inputs(tfactor=2.0, n_b0_values=n_b0_values)

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
    "gradient_scale": 10.0,  # relative size of gradient to RF pulse
    "positive_gradient": False,
    "tmin": fixed_inputs["t_B1"][0].item(),
    "tmax": fixed_inputs["t_B1"][-1].item(),
}

# PARAMETERS OF THE SCANNER:
# (will appear in as constraints in Loss function)
scanner_params = {
    "max_gradient": 50,  # mT/m
    "max_diff_gradient": 200,  # mT/m/ms
    "max_pulse_amplitude": 0.023,  # mT
}
