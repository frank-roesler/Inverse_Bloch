from utils_bloch.setup import get_fixed_inputs
import numpy as np


# TRAINING PARAMETERS:
start_epoch = 0
epochs = 20000
resume_from_path = None  # "results/train_log.pt"  # path to resume training from
lr = {"pulse": 3e-5, "gradient": 3e-5}  # learning rate
plot_loss_frequency = 100  # plot every n steps
start_logging = 1000  # start logging after n steps
pre_train_inputs = False  # pre-train on given RF-pulse & gradient
suppress_loss_peaks = False  # detect peaks in loss function and reduce lr
loss_metric = "L1"
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
n_b0_values = 3
flip_angle = 0.5 * np.pi
fixed_inputs = get_fixed_inputs(tfactor=4.0, n_b0_values=n_b0_values, Nz=256, Nt=64)

# MODEL PARAMETERS:
modelname = "MixedModel"  # MLP, SIREN, RBFN, FourierMLP, FourierSeries, ModulatedFourier, MixedModel
model_args = {
    "n_coeffs": 30,  # Fourier Series
    "omega_0": 45,  # SIREN
    "bandwidth": 101,  # ModulatedFourier
    "hidden_dim": 32,  # MLP, SIREN, ModulatedFourier
    "num_layers": 16,  # MLP, SIREN
    "num_centers": 10,  # RBFN
    "center_spacing": 1,  # RBFN
    "num_fourier_features": 51,  # FourierMLP
    "frequency_scale": 100.0,  # FourierMLP
    "tvector": fixed_inputs["t_B1"][:, 0],  # NoModel
    "gradient_scale": 20.0,  # relative size of gradient to RF pulse
    "positive_gradient": False,
    "tmin": fixed_inputs["t_B1"][0].item(),
    "tmax": fixed_inputs["t_B1"][-1].item(),
}

# PARAMETERS OF THE SCANNER:
# (will appear in as constraints in Loss function)
scanner_params = {
    "max_gradient": 50,  # mT/m
    "max_diff_gradient": 200,  # mT/m/ms
    "max_pulse_amplitude": 0.02,  # mT
}
