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
    @classmethod
    def from_dict(cls, data_dict):
        inst = cls.__new__(cls)
        if hasattr(cls, "attributes"):
            for k in cls.attributes:
                if k in data_dict:
                    setattr(inst, k, data_dict[k])
        return inst

    def __init__(self, path):
        data_dict = self.load_toml(path)
        inst = self.from_dict(data_dict)
        self.__dict__.update(inst.__dict__)

    def to_dict(self):
        if hasattr(self, "attributes"):
            return {k: getattr(self, k) for k in self.attributes if hasattr(self, k)}
        return {k: v for k, v in self.__dict__.items() if not k.startswith("_")}

    def load_toml(self, path):
        with open(path, "rb") as f:
            data_dict = tomllib.load(f)
        return data_dict


class TrainingConfig(InputData):
    attributes = [
        "start_step",
        "target_smoothness",
        "shift_targets",
        "steps",
        "resume_from_path",
        "lr",
        "plot_loss_frequency",
        "start_logging",
        "pre_train_inputs",
        "suppress_loss_peaks",
        "loss_metric",
        "loss_weights",
        "phase_offset_in_rad",
    ]

    def __init__(self, path):
        super().__init__(path)


class BlochConfig(InputData):
    attributes = [
        "n_slices",
        "n_b0_values",
        "flip_angle",
        "tfactor",
        "Nz",
        "Nt",
        "pos_spacing",
        "fixed_inputs",
    ]

    def __init__(self, path):
        super().__init__(path)
        self.set_fixed_inputs()

    def set_fixed_inputs(self):
        self.fixed_inputs = get_fixed_inputs(tfactor=self.tfactor, n_b0_values=self.n_b0_values, Nz=self.Nz, Nt=self.Nt, pos_spacing=self.pos_spacing, n_slices=self.n_slices)


class ModelConfig(InputData):
    attributes = ["model_name", "model_args"]

    def __init__(self, t_B1, path):
        super().__init__(path)
        self.model_args["tmin"] = t_B1[0].item()
        self.model_args["tmax"] = t_B1[-1].item()


class ScannerConfig(InputData):
    attributes = ["scanner_params"]
