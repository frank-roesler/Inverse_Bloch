import scipy
import torch
import numpy as np
import tomllib
from constants import gamma, gam_hz_mt, larmor_mhz, water_ppm


def get_fixed_inputs(tfactor=1.0, n_b0_values=1, Nz=4096, Nt=512, pos_spacing="linear", n_slices=1):
    inputs = scipy.io.loadmat("data/smPulse_512pts.mat")
    inputs["returnallstates"] = False
    inputs["dt"] = inputs["dtmb"].item()
    dt = inputs["dt"]
    sens = torch.ones((Nz, 1), dtype=torch.complex64)
    B0 = torch.zeros((Nz, 1))
    Tmin = dt * 1e3 * 512
    T = Tmin * tfactor
    Nt = int(Nt * tfactor)
    t_B1 = torch.linspace(0, T, Nt)
    t_B1 = t_B1.float()
    dt_num = (t_B1[-1] - t_B1[0]) / (len(t_B1) - 1)

    if pos_spacing == "nonlinear":
        u = torch.linspace(-1, 1, Nz)
        pos = abs(u) ** 3
        pos[u < 0] = -pos[u < 0]
        pos += 0.5 * n_slices * u
        pos = 2 * 0.18 * pos / (torch.max(pos) - torch.min(pos))
        pos = pos.detach()
    else:
        pos = torch.linspace(-0.18, 0.18, Nz)

    dx = (pos[-1] - pos[0]) / (len(pos) - 1)
    inputs["pos"] = pos
    sens = sens.detach().requires_grad_(False)
    B0 = B0.detach().requires_grad_(False)
    M0 = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32)
    if n_b0_values == 1:
        freq_offsets_Hz = torch.linspace(-larmor_mhz * water_ppm, 0.0, 2)
        freq_offsets_Hz = torch.mean(freq_offsets_Hz, dim=0, keepdim=True)
    else:
        freq_offsets_Hz = torch.linspace(-larmor_mhz * water_ppm, 0.0, n_b0_values)
    B0_freq_offsets_mT = freq_offsets_Hz / gam_hz_mt
    B0_vals = []
    for ff in range(len(freq_offsets_Hz)):
        B0_vals.append(B0 + B0_freq_offsets_mT[ff])
    B0_list = torch.stack(B0_vals, dim=0).to(torch.float32)
    return {
        "pos": pos,  # [dm]
        "dt": dt * tfactor,
        "dt_num": dt_num.item() * 1e-3,
        "dx": dx.item(),
        "Nz": Nz,
        "sens": sens,
        "B0": B0,
        "t_B1": t_B1.unsqueeze(1),
        "t_B1_legacy": torch.arange(0, len(inputs["rfmb"])).unsqueeze(1) * dt * 1e3 * tfactor,
        "M0": M0,
        "inputs": inputs,
        "freq_offsets_Hz": freq_offsets_Hz,
        "B0_list": B0_list,
    }


class InputData:
    def to_dict(self):
        pass

    def load_toml(self, path):
        with open(path, "rb") as f:
            data_dict = tomllib.load(f)
        return data_dict


class TrainingConfig(InputData):
    def __init__(self, path):
        super().__init__()
        data_dict = self.load_toml(path)
        self.start_step = data_dict["start_step"]
        self.target_smoothness = data_dict["target_smoothness"]
        self.shift_targets = data_dict["shift_targets"]
        self.steps = data_dict["steps"]
        self.resume_from_path = data_dict["resume_from_path"]
        self.lr = data_dict["lr"]
        self.plot_loss_frequency = data_dict["plot_loss_frequency"]
        self.start_logging = data_dict["start_logging"]
        self.pre_train_inputs = data_dict["pre_train_inputs"]
        self.suppress_loss_peaks = data_dict["suppress_loss_peaks"]
        self.loss_metric = data_dict["loss_metric"]
        self.loss_weights = data_dict["loss_weights"]
        self.phase_offset_in_rad = data_dict["phase_offset_in_rad"]

    def to_dict(self):
        out_dict = {}
        out_dict["start_step"] = self.start_step
        out_dict["target_smoothness"] = self.target_smoothness
        out_dict["shift_targets"] = self.shift_targets
        out_dict["steps"] = self.steps
        out_dict["resume_from_path"] = self.resume_from_path
        out_dict["lr"] = self.lr
        out_dict["plot_loss_frequency"] = self.plot_loss_frequency
        out_dict["start_logging"] = self.start_logging
        out_dict["pre_train_inputs"] = self.pre_train_inputs
        out_dict["suppress_loss_peaks"] = self.suppress_loss_peaks
        out_dict["loss_metric"] = self.loss_metric
        out_dict["loss_weights"] = self.loss_weights
        out_dict["phase_offset_in_rad"] = self.phase_offset_in_rad
        return out_dict


class BlochConfig(InputData):
    def __init__(self, path):
        super().__init__()
        data_dict = self.load_toml(path)
        self.n_slices = data_dict["n_slices"]
        self.n_b0_values = data_dict["n_b0_values"]
        self.flip_angle = data_dict["flip_angle"]
        self.tfactor = data_dict["tfactor"]
        self.Nz = data_dict["Nz"]
        self.Nt = data_dict["Nt"]
        self.pos_spacing = data_dict["pos_spacing"]
        self.fixed_inputs = get_fixed_inputs(tfactor=self.tfactor, n_b0_values=self.n_b0_values, Nz=self.Nz, Nt=self.Nt, pos_spacing=self.pos_spacing, n_slices=self.n_slices)

    def to_dict(self):
        out_dict = {}
        out_dict["n_slices"] = self.n_slices
        out_dict["n_b0_values"] = self.n_b0_values
        out_dict["flip_angle"] = self.flip_angle
        out_dict["tfactor"] = self.tfactor
        out_dict["Nz"] = self.Nz
        out_dict["Nt"] = self.Nt
        out_dict["pos_spacing"] = self.pos_spacing
        out_dict["fixed_inputs"] = self.fixed_inputs
        return out_dict


class ModelConfig(InputData):
    def __init__(self, t_B1, path):
        super().__init__()
        data_dict = self.load_toml(path)
        self.model_name = data_dict["model_name"]
        self.model_args = data_dict["model_args"]
        self.model_args["tmin"] = t_B1[0].item()
        self.model_args["tmax"] = t_B1[-1].item()

    def to_dict(self):
        out_dict = {}
        out_dict["model_name"] = self.model_name
        out_dict["model_args"] = self.model_args
        return out_dict


class ScannerConfig(InputData):
    def __init__(self, path):
        super().__init__()
        data_dict = self.load_toml(path)
        self.scanner_params = data_dict["scanner_params"]

    def to_dict(self):
        return {"scanner_params": self.scanner_params}
