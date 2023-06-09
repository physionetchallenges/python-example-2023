#!/usr/bin/env python

# Load libraries.
import os, sys, shutil, argparse

# Parse arguments.
def get_parser():
    description = 'Remove labels from the dataset.'
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('-i', '--input_folder', type=str, required=True)
    parser.add_argument('-p', '--patient_ids', nargs='*', type=str, required=False, default=[])
    parser.add_argument('-o', '--output_folder', type=str, required=True)
    return parser

# Find folders with data files.
def find_data_folders(root_folder):
    data_folders = list()
    for x in sorted(os.listdir(root_folder)):
        data_folder = os.path.join(root_folder, x)
        if os.path.isdir(data_folder):
            data_file = os.path.join(data_folder, x + '.txt')
            if os.path.isfile(data_file):
                data_folders.append(x)
    return sorted(data_folders)

# Run script.
def run(args):
    # Use either the given patient IDs or all of the patient IDs.
    if args.patient_ids:
        patient_ids = args.patient_ids
    else:
        patient_ids = find_data_folders(args.input_folder)

    # Iterate over the patient IDs.
    for patient_id in patient_ids:
        input_path = os.path.join(args.input_folder, patient_id)
        output_path = os.path.join(args.output_folder, patient_id)
        os.makedirs(output_path, exist_ok=True)

        # Iterate over the files in each folder.
        for file_name in sorted(os.listdir(input_path)):
            file_root, file_ext = os.path.splitext(file_name)
            input_file = os.path.join(input_path, file_name)
            output_file = os.path.join(output_path, file_name)

            # If the file does have the labels, then remove the labels and copy the rest of the file.
            if file_ext == '.txt' and file_root == patient_id:
                with open(input_file, 'r') as f:
                    input_lines = f.readlines()
                output_lines = [l for l in input_lines if not (l.startswith('Outcome') or l.startswith('CPC'))]
                output_string = ''.join(output_lines)
                with open(output_file, 'w') as f:
                    f.write(output_string)

            # Otherwise, copy the file as-is.
            else:
                shutil.copy2(input_file, output_file)

if __name__=='__main__':
    run(get_parser().parse_args(sys.argv[1:]))
