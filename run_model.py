#!/usr/bin/env python

# Do *not* edit this script. Changes will be discarded so that we can run the trained models consistently.

# This file contains functions for running models for the Challenge. You can run it as follows:
#
#   python run_model.py models data outputs
#
# where 'models' is a folder containing the your trained models, 'data' is a folder containing the Challenge data, and 'outputs' is a
# folder for saving your models' outputs.

import numpy as np, scipy as sp, os, sys
from helper_code import *
from team_code import load_challenge_models, run_challenge_models

# Run model.
def run_model(model_folder, data_folder, output_folder, allow_failures, verbose):
    # Load model(s).
    if verbose >= 1:
        print('Loading the Challenge models...')

    # You can use this function to perform tasks, such as loading your models, that you only need to perform once.
    models = load_challenge_models(model_folder, verbose) ### Teams: Implement this function!!!

    # Find the Challenge data.
    if verbose >= 1:
        print('Finding the Challenge data...')

    patient_ids = find_data_folders(data_folder)
    num_patients = len(patient_ids)

    if num_patients==0:
        raise Exception('No data were provided.')

    # Create a folder for the Challenge outputs if it does not already exist.
    os.makedirs(output_folder, exist_ok=True)

    # Run the team's model(s) on the Challenge data.
    if verbose >= 1:
        print('Running the Challenge models on the Challenge data...')

    # Iterate over the patients.
    for i in range(num_patients):
        if verbose >= 2:
            print('    {}/{}...'.format(i+1, num_patients))

        patient_id = patient_ids[i]

        # Allow or disallow the model(s) to fail on parts of the data; this can be helpful for debugging.
        try:
            outcome_binary, outcome_probability, cpc = run_challenge_models(models, data_folder, patient_id, verbose) ### Teams: Implement this function!!!
        except:
            if allow_failures:
                if verbose >= 2:
                    print('... failed.')
                outcome_binary, outcome_probability, cpc = float('nan'), float('nan'), float('nan')
            else:
                raise

        # Save Challenge outputs.
        os.makedirs(os.path.join(output_folder, patient_id), exist_ok=True)
        output_file = os.path.join(output_folder, patient_id, patient_id + '.txt')
        save_challenge_outputs(output_file, patient_id, outcome_binary, outcome_probability, cpc)

    if verbose >= 1:
        print('Done.')

if __name__ == '__main__':
    # Parse the arguments.
    if not (len(sys.argv) == 4 or len(sys.argv) == 5):
        raise Exception('Include the model, data, and output folders as arguments, e.g., python run_model.py model data outputs.')

    # Define the model, data, and output folders.
    model_folder = sys.argv[1]
    data_folder = sys.argv[2]
    output_folder = sys.argv[3]

    # Allow or disallow the model to fail on parts of the data; helpful for debugging.
    allow_failures = False

    # Change the level of verbosity; helpful for debugging.
    if len(sys.argv)==5 and is_integer(sys.argv[4]):
        verbose = int(sys.argv[4])
    else:
        verbose = 1

    run_model(model_folder, data_folder, output_folder, allow_failures, verbose)
