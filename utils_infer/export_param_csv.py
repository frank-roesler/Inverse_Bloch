import torch
import csv
from os.path import join, dirname
from utils_bloch import blochsim_CK_batch
from utils_train.utils import move_to
from utils_bloch.setup import get_smooth_targets
import numpy as np
from constants import gamma, gam_hz_mt, larmor_mhz, water_ppm


def write_rows_from_dict(heading, input_dict, writer, exclude=[]):
    writer.writerow([heading])
    for key, value in input_dict.items():
        if key in exclude:
            continue
        writer.writerow([key, value])

    writer.writerow([])


def compute_actual_phase_offsets(data_dict, B1, G, fixed_inputs):
    from utils_infer import compute_phase_offsets

    gamma_hz_mt = fixed_inputs["gam_hz_mt"]
    t_B1 = fixed_inputs["t_B1_legacy"]
    pos = fixed_inputs["pos"]
    freq_offsets_Hz = torch.linspace(-larmor_mhz * water_ppm, 0.0, data_dict["n_b0_values"])
    B0_freq_offsets_mT = freq_offsets_Hz / gamma_hz_mt
    B0_list = []
    for ff in range(len(freq_offsets_Hz)):
        B0_list.append(fixed_inputs["B0"] + B0_freq_offsets_mT[ff])

    B0 = torch.stack(B0_list, dim=0).to(torch.float32)
    mxy, mz = blochsim_CK_batch(B1=B1, G=G, pos=pos, sens=fixed_inputs["sens"], B0_list=B0, M0=fixed_inputs["M0"], dt=fixed_inputs["dt_num"])
    target_z, target_xy, slice_centers_allB0, half_width = get_smooth_targets(
        theta=data_dict["flip_angle"],
        smoothness=4,
        function=torch.sigmoid,
        n_targets=data_dict["n_slices"],
        pos=fixed_inputs["pos"],
        n_b0_values=data_dict["n_b0_values"],
        shift_targets=data_dict["shift_targets"] if "shift_targets" in data_dict else False,
    )
    (mxy, mz, pos, target_xy, target_z, t_B1, G, B1) = move_to((mxy, mz, pos, target_xy, target_z, t_B1, G, B1), torch.device("cpu"))
    phase_offsets_all_b0 = []
    for ff in range(len(freq_offsets_Hz)):
        phase = np.unwrap(np.angle(mxy[ff, :]))
        phase_offsets_all_b0.append(compute_phase_offsets(phase, slice_centers_allB0[ff], half_width))
    return phase_offsets_all_b0


def writerow_if_present(writer, key, input_dict):
    if key in input_dict:
        writer.writerow([key, input_dict[key]])


def export_param_csv(input_path, output_path, B1, G, fixed_inputs, slopes):
    data_dict = torch.load(input_path, weights_only=False)
    with open(join(dirname(output_path), "params.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        write_rows_from_dict("Model Args", data_dict["model_args"], writer, ["tvector"])
        write_rows_from_dict("Scanner Parameters", data_dict["scanner_params"], writer)

        writer.writerow(["Training Parameters"])
        writer.writerow(["model", data_dict["model"].name])
        writerow_if_present(writer, "epoch", data_dict)
        writerow_if_present(writer, "loss_metric", data_dict)
        writerow_if_present(writer, "shift_targets", data_dict)
        write_rows_from_dict("Loss Weights", data_dict["loss_weights"], writer)

        writer.writerow(["Bloch Parameters"])
        writerow_if_present(writer, "n_slices", data_dict)
        writerow_if_present(writer, "n_b0_values", data_dict)
        writerow_if_present(writer, "flip_angle", data_dict)
        writerow_if_present(writer, "tfactor", data_dict)
        writerow_if_present(writer, "Nz", data_dict)
        writerow_if_present(writer, "Nt", data_dict)
        writerow_if_present(writer, "pos_spacing", data_dict)
        writer.writerow([])

        write_rows_from_dict("Fixed Inputs", data_dict["fixed_inputs"], writer, ["pos", "t_B1", "t_B1_legacy", "sens", "B0", "M0", "inputs", "B0_list"])

        writer.writerow(["Actual Phase Offsets"])
        phase_offsets_all_b0 = compute_actual_phase_offsets(data_dict, B1, G, fixed_inputs)
        for i, phase_offsets in enumerate(phase_offsets_all_b0):
            current_offsets = [f"{(offset.item()/2/np.pi*360+180)%360-180:.2f}" for offset in phase_offsets]
            offset_strings = "; ".join(current_offsets)
            writer.writerow([i, offset_strings])

        writer.writerow(["Gradient moments"])
        slice_selec_mom = 1000 * torch.sum(G, dim=0).item() * fixed_inputs["dt_num"]
        for i, slope in enumerate(slopes):
            refoc_moment = 1000 * slope / fixed_inputs["gam"]
            writer.writerow([i, "Slice Selection moment", f"{slice_selec_mom:.6f}"])
            writer.writerow([i, "Refocusing moment fraction", refoc_moment / slice_selec_mom])
