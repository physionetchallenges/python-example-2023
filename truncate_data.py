#!/usr/bin/env python

# Load libraries.
import os, os.path, sys, shutil, argparse
from helper_code import *

# Parse arguments.
def get_parser():
    description = 'Truncate recordings to the provided time limit (in hours).'
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('-i', '--input_folder', type=str, required=True)
    parser.add_argument('-p', '--patient_ids', nargs='*', type=str, required=False, default=[])
    parser.add_argument('-t', '--time_limit', type=float, required=True)
    parser.add_argument('-o', '--output_folder', type=str, required=True)
    return parser

# Run script.
def run(args):
    # Convert hours to seconds.
    time_limit = 3600 * args.time_limit

    # Identify the data folders.
    if args.patient_ids:
        patient_ids = args.patient_ids
    else:
        patient_ids = find_data_folders(args.input_folder)

    # Iterate over each folder.
    for patient_id in patient_ids:
        # Set the paths.
        input_path = os.path.join(args.input_folder, patient_id)
        output_path = os.path.join(args.output_folder, patient_id)

        # Create the output folder.
        if os.path.exists(input_path):
            os.makedirs(output_path, exist_ok=True)

            # Copy the patient metadata file.
            input_patient_metadata_file = os.path.join(input_path, patient_id + '.txt')
            output_patient_metadata_file = os.path.join(output_path, patient_id + '.txt')
            shutil.copy(input_patient_metadata_file, output_patient_metadata_file)

            # Copy the WFDB header and signal files for records that end before the end time.
            for filename in os.listdir(input_path):
                if not filename.startswith('.') and filename.endswith('.hea'):
                    header_file = filename
                    input_header_file = os.path.join(input_path, header_file)
                    output_header_file = os.path.join(output_path, header_file)

                    header_text = load_text_file(input_header_file)
                    start_time = convert_hours_minutes_seconds_to_seconds(*get_start_time(header_text))
                    end_time = convert_hours_minutes_seconds_to_seconds(*get_end_time(header_text))

                    # If end time for a recording is before the time limit, then copy the recording.
                    if end_time < time_limit:
                        signal_files = set()
                        for i, l in enumerate(header_text.split('\n')):
                            arrs = [arr.strip() for arr in l.split(' ')]
                            if i > 0 and not l.startswith('#') and len(arrs) > 0 and len(arrs[0]) > 0:
                                signal_file = arrs[0]
                                signal_files.add(signal_file)

                        signal_files = sorted(signal_files)
                        input_signal_files = [os.path.join(input_path, signal_file) for signal_file in signal_files]
                        output_signal_files = [os.path.join(output_path, signal_file) for signal_file in signal_files]

                        shutil.copy2(input_header_file, output_header_file)
                        for input_signal_file, output_signal_file in zip(input_signal_files, output_signal_files):
                            shutil.copy(input_signal_file, output_signal_file)

                    # If the start time is before the time limit and the end time is after the time limit, then truncate the recording.
                    elif start_time < time_limit and end_time >= time_limit:
                        record_name = header_text.split(' ')[0]
                        raise NotImplementedError('Part (but not all) of record {} exceeds the end time.'.format(record_name)) # All of the files in the dataset end on the hour.

                    # If the start time is after the time limit, then do not copy or truncate the recording.
                    elif start_time >= time_limit:
                        pass

if __name__=='__main__':
    run(get_parser().parse_args(sys.argv[1:]))
