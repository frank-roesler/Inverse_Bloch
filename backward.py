from utils_train.nets import get_model
from utils_train.utils import *
from utils_bloch.setup import get_smooth_targets
from params import *


device = get_device()
# device = torch.device("cpu")

target_z, target_xy, _, _ = get_smooth_targets(
    theta=flip_angle, smoothness=target_smoothness, function=torch.sigmoid, n_targets=n_slices, pos=fixed_inputs["pos"], n_b0_values=n_b0_values, shift_targets=shift_targets
)

model = get_model(modelname, **model_args)

model, optimizer, scheduler, losses = init_training(model, lr, device=device)
if resume_from_path != None:
    pre_train_inputs = False
    (model, target_z, target_xy, optimizer, losses, fixed_inputs, flip_angle, loss_metric, scanner_params, loss_weights, model_args, start_epoch) = load_data(resume_from_path, mode="train")
    # start_epoch, losses, model, optimizer, _, _, _, _, fixed_inputs = load_data_old(resume_from_path)
    # for param_group in optimizer.param_groups:
    #     param_group["lr"] *= 2
    # loss_metric = "L1"
    # loss_weights["phase_loss"] = 1.0
    # scanner_params["max_pulse_amplitude"] = 0.04

train(
    model,
    target_z,
    target_xy,
    optimizer,
    scheduler,
    losses,
    fixed_inputs,
    flip_angle,
    loss_metric,
    scanner_params,
    loss_weights,
    n_slices,
    n_b0_values,
    tfactor,
    Nz,
    Nt,
    pos_spacing,
    shift_targets,
    start_epoch,
    epochs,
    device,
    start_logging,
    plot_loss_frequency,
    pre_train_inputs,
    suppress_loss_peaks,
)
