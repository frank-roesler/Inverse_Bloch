import torch
import csv
from os.path import join, dirname


def write_rows_from_dict(heading, input_dict, writer, exclude=[]):
    writer.writerow([heading])
    for key, value in input_dict.items():
        if key in exclude:
            continue
        writer.writerow([key, value])

    writer.writerow([])


def writerow_if_present(writer, key, input_dict):
    if key in input_dict:
        writer.writerow([key, input_dict[key]])


def export_param_csv(input_path, output_path):
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
