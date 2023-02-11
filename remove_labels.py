#!/usr/bin/env python

# Load libraries.
import os, sys, shutil, argparse
from helper_code import *

# Parse arguments.
def get_parser():
    description = 'Remove labels from the data.'
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('-i', '--input_folder', type=str, required=True)
    parser.add_argument('-o', '--output_folder', type=str, required=True)
    parser.add_argument('-ncd', '--no_copy_data', action='store_true')    
    return parser

# Run script.
def run(args):
    # Find the folders with the patient and recording metadata and data.
    patient_ids = find_data_folders(args.input_folder)

    # Iterate over each folder.
    for patient_id in patient_ids:
        # Make output folder.
        os.makedirs(os.path.join(args.output_folder, patient_id), exist_ok=True)

        # Optionally copy the full patient and recording metadata and data files.
        if not args.no_copy_data:
            input_patient_folder = os.path.join(args.input_folder, patient_id)
            output_patient_folder = os.path.join(args.output_folder, patient_id)
            shutil.copytree(input_patient_folder, output_patient_folder, dirs_exist_ok=True)

        # Update the patient metadata file.
        input_patient_metadata_file = os.path.join(args.input_folder, patient_id, patient_id + '.txt')
        output_patient_metadata_file = os.path.join(args.output_folder, patient_id, patient_id + '.txt')
     
        with open(input_patient_metadata_file, 'r') as f:
            input_lines = f.readlines() 

        output_lines = [l for l in input_lines if not l.startswith('Outcome') and not l.startswith('CPC')]
        output_string = ''.join(output_lines)

        with open(output_patient_metadata_file, 'w') as f:
            f.write(output_string)

if __name__=='__main__':
    run(get_parser().parse_args(sys.argv[1:]))
