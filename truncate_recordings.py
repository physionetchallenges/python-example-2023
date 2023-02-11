#!/usr/bin/env python

# Load libraries.
import os, sys, shutil, argparse
from helper_code import *

# Parse arguments.
def get_parser():
    description = 'Truncate recordings for the provided hour limit.'
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('-i', '--input_folder', type=str, required=True)
    parser.add_argument('-o', '--output_folder', type=str, required=True)
    parser.add_argument('-k', '--hour_limit', type=int, required=True)
    return parser

# Run script.
def run(args):
    # Find the folders with the patient and recording metadata and data.
    patient_ids = find_data_folders(args.input_folder)

    # Iterate over each folder.
    for patient_id in patient_ids:
        # Make output folder.
        os.makedirs(os.path.join(args.output_folder, patient_id), exist_ok=True)

        # Set and copy the patient metadata file.
        input_patient_metadata_file = os.path.join(args.input_folder, patient_id, patient_id + '.txt')
        output_patient_metadata_file = os.path.join(args.output_folder, patient_id, patient_id + '.txt')
     
        shutil.copy(input_patient_metadata_file, output_patient_metadata_file)

        # Set, truncate, and copy the recording metadata file. 
        input_recording_metadata_file = os.path.join(args.input_folder, patient_id, patient_id + '.tsv')
        output_recording_metadata_file = os.path.join(args.output_folder, patient_id, patient_id + '.tsv')
    
        input_recording_metadata = load_text_file(input_recording_metadata_file)
        hours = get_hours(input_recording_metadata)
        indices = [i for i, hour in enumerate(hours) if hour <= args.hour_limit]

        input_lines = input_recording_metadata.split('\n')
        lines = [input_lines[0]] + [input_lines[i+1] for i in indices]
        output_recording_metadata = '\n'.join(lines)
        with open(output_recording_metadata_file, 'w') as f:
            f.write(output_recording_metadata)

        # Set and copy the recording data.
        recording_ids = get_recording_ids(input_recording_metadata)
        for i in indices:
            recording_id = recording_ids[i]
            input_header_file = os.path.join(args.input_folder, patient_id, recording_id + '.hea')
            input_signal_file = os.path.join(args.input_folder, patient_id, recording_id + '.mat')
            output_header_file = os.path.join(args.output_folder, patient_id, recording_id + '.hea')
            output_signal_file = os.path.join(args.output_folder, patient_id, recording_id + '.mat')

            if os.path.isfile(input_header_file):
                shutil.copy(input_header_file, output_header_file)
            if os.path.isfile(input_signal_file):
                shutil.copy(input_signal_file, output_signal_file)

if __name__=='__main__':
    run(get_parser().parse_args(sys.argv[1:]))
