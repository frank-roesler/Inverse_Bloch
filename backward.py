from utils_train.nets import get_model
from utils_train.utils import *
from utils_bloch.setup import get_smooth_targets
from config import *

config_path = "config.toml"

device = get_device()

tconfig = TrainingConfig(config_path)
bconfig = BlochConfig(config_path)
mconfig = ModelConfig(config_path)
sconfig = ScannerConfig(config_path)


target_z, target_xy, _, _ = get_smooth_targets(bconfig, tconfig, function=torch.sigmoid)
model = get_model(mconfig)
model, optimizer, scheduler, losses = init_training(model, tconfig.lr, device=device)

if tconfig.resume_from_path != None:
    pre_train_inputs = False
    (model, target_z, target_xy, optimizer, losses, tconfig, bconfig, mconfig, sconfig) = load_data(tconfig.resume_from_path, mode="train")
    # start_epoch, losses, model, optimizer, _, _, _, _, fixed_inputs = load_data_old(resume_from_path)

train(model, target_z, target_xy, optimizer, scheduler, losses, device, tconfig, bconfig, mconfig, sconfig)
