from utils_train.nets import get_model
from utils_train.utils import *
from utils_bloch.setup import get_smooth_targets
from params import *


device = get_device()
target_z, target_xy, _, _ = get_smooth_targets(theta=flip_angle, smoothness=2.0, function=torch.sigmoid, n_targets=n_slices)

model = get_model(modelname, **model_args)
if suppress_loss_peaks:
    model_old = get_model(modelname, **model_args)

model, optimizer, scheduler, losses = init_training(model, lr, device=device)
if resume_from_path != None:
    pre_train_inputs = False
    (model, target_z, target_xy, optimizer, losses, fixed_inputs, flip_angle, loss_metric, scanner_params, loss_weights, start_epoch) = load_data(resume_from_path, mode="train")


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
    start_epoch,
    epochs,
    device,
    start_logging,
    plot_loss_frequency,
    pre_train_inputs,
)
