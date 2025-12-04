from utils_bloch.setup import get_fixed_inputs
import numpy as np


# TRAINING PARAMETERS:
start_epoch = 0
target_smoothness = 4.0
shift_targets = True
epochs = 40000
resume_from_path = None  # "results/2025-07-12_15-06/train_log.pt"  # path to resume training from
lr = {"pulse": 1e-4, "gradient": 1e-4}  # learning rate
plot_loss_frequency = 100  # plot every n steps
start_logging = 2000  # start logging after n steps
pre_train_inputs = False  # pre-train on given RF-pulse & gradient
suppress_loss_peaks = False  # detect peaks in loss function and reduce lr
loss_metric = "L1"
loss_weights = {
    "loss_mxy": 1.0,
    "loss_mz": 1.0,
    "boundary_vals_pulse": 100.0,
    "gradient_height_loss": 0.1,
    "pulse_height_loss": 1000.0,
    "gradient_diff_loss": 1.0,
    # "phase_ddiff": 0.0,
    "phase_diff_var": 1000.0,
    # "phase_B0_diff": 1e-9,
}

# BLOCH PARAMETERS:
n_slices = 2
n_b0_values = 5
flip_angle = 0.5 * np.pi
tfactor = 2  # pulse time is 0.64ms * tfactor
Nz = 128  # number of mesh points in pos axis
Nt = 64  # number of mesh points per 0.64ms time interval
pos_spacing = "nonlinear"  # "nonlinear" places more mesh points in the center
fixed_inputs = get_fixed_inputs(tfactor=tfactor, n_b0_values=n_b0_values, Nz=Nz, Nt=Nt, pos_spacing=pos_spacing, n_slices=n_slices)

# MODEL PARAMETERS:
modelname = "MixedModel"  # MLP, SIREN, RBFN, FourierMLP, FourierSeries, ModulatedFourier, MixedModel
model_args = {
    "n_coeffs": 50,  # Fourier Series
    "omega_0": 43,  # SIREN
    "bandwidth": 101,  # ModulatedFourier
    "hidden_dim": 32,  # MLP, SIREN, ModulatedFourier
    "num_layers": 16,  # MLP, SIREN
    "num_centers": 10,  # RBFN
    "center_spacing": 1,  # RBFN
    "num_fourier_features": 51,  # FourierMLP
    "frequency_scale": 100.0,  # FourierMLP
    "tvector": fixed_inputs["t_B1"][:, 0],  # NoModel
    "gradient_scale": 20.0,  # relative size of gradient to RF pulse
    "positive_gradient": True,
    "tmin": fixed_inputs["t_B1"][0].item(),
    "tmax": fixed_inputs["t_B1"][-1].item(),
}

# PARAMETERS OF THE SCANNER:
# (will appear as constraints in Loss function)
scanner_params = {
    "max_gradient": 50,  # mT/m
    "max_diff_gradient": 200,  # mT/m/ms
    "max_pulse_amplitude": 0.035,  # mT
}
