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


def export_param_csv(input_path, output_path):
    data_dict = torch.load(input_path, weights_only=False)
    with open(join(dirname(output_path), "params.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        write_rows_from_dict("Model Args", data_dict["model_args"], writer, ["tvector"])
        write_rows_from_dict("Scanner Parameters", data_dict["scanner_params"], writer)

        writer.writerow(["Training Parameters"])
        writer.writerow(["model", data_dict["model"].name])
        writer.writerow(["epoch", data_dict["epoch"]])
        writer.writerow(["loss_metric", data_dict["loss_metric"]])
        write_rows_from_dict("Loss Weights", data_dict["loss_weights"], writer)

        writer.writerow(["Bloch Parameters"])
        writer.writerow(["n_slices", data_dict["n_slices"]])
        writer.writerow(["n_b0_values", data_dict["n_b0_values"]])
        writer.writerow(["flip_angle", data_dict["flip_angle"]])
        writer.writerow(["tfactor", data_dict["tfactor"]])
        writer.writerow(["Nz", data_dict["Nz"]])
        writer.writerow(["Nt", data_dict["Nt"]])
        writer.writerow(["pos_spacing", data_dict["pos_spacing"]])
        writer.writerow([])

        write_rows_from_dict("Fixed Inputs", data_dict["fixed_inputs"], writer, ["pos", "t_B1", "t_B1_legacy", "sens", "B0", "M0", "inputs", "B0_list"])
