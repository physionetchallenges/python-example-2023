#!/usr/bin/env python

# Do *not* edit this script.
# These are helper functions that you can use with your code.
# Check the example code to see how to import these functions to your code.

import os, numpy as np, scipy as sp, scipy.io

### Challenge data I/O functions

# Find the folders with data files.
def find_data_folders(root_folder):
    data_folders = list()
    for x in os.listdir(root_folder):
        data_folder = os.path.join(root_folder, x)
        if os.path.isdir(data_folder):
            data_folders.append(x)
    return sorted(data_folders)

def load_challenge_data(data_folder, patient_id):
    # Define file location.
    patient_metadata_file = os.path.join(data_folder, patient_id, patient_id + '.txt')
    recording_metadata_file = os.path.join(data_folder, patient_id, patient_id + '.tsv')

    # Load non-recording data.
    patient_metadata = load_text_file(patient_metadata_file)
    recording_metadata = load_text_file(recording_metadata_file)

    # Load recordings.
    recordings = list()
    recording_ids = get_recording_ids(recording_metadata)
    for recording_id in recording_ids:
        if not is_nan(recording_id):
            recording_location = os.path.join(data_folder, patient_id, recording_id)
            recording_data, sampling_frequency, channels = load_recording(recording_location)
        else:
            recording_data = None
            sampling_frequency = None
            channels = None
        recordings.append((recording_data, sampling_frequency, channels))

    return patient_metadata, recording_metadata, recordings

# Load the WFDB data for the Challenge (but not all possible WFDB files).
def load_recording(record_name):
    # Allow either the record name or the header filename.
    root, ext = os.path.splitext(record_name)
    if ext=='':
        header_file = record_name + '.hea'
    else:
        header_file = record_name

    # Load the header file.
    if not os.path.isfile(header_file):
        raise FileNotFoundError('{} recording not found.'.format(record_name))

    with open(header_file, 'r') as f:
        header = [l.strip() for l in f.readlines() if l.strip()]

    # Parse the header file.
    record_name = None
    num_signals = None
    sampling_frequency = None
    num_samples = None
    signal_files = list()
    gains = list()
    offsets = list()
    channels = list()
    initial_values = list()
    checksums = list()

    for i, l in enumerate(header):
        arrs = [arr.strip() for arr in l.split(' ')]
        # Parse the record line.
        if i==0:
            record_name = arrs[0]
            num_signals = int(arrs[1])
            sampling_frequency = float(arrs[2])
            num_samples = int(arrs[3])
        # Parse the signal specification lines.
        else:
            signal_file = arrs[0]
            gain = float(arrs[2].split('/')[0])
            offset = int(arrs[4])
            initial_value = int(arrs[5])
            checksum = int(arrs[6])
            channel = arrs[8]
            signal_files.append(signal_file)
            gains.append(gain)
            offsets.append(offset)
            initial_values.append(initial_value)
            checksums.append(checksum)
            channels.append(channel)

    # Check that the header file only references one signal file. WFDB format  allows for multiple signal files, but we have not
    # implemented that here for simplicity.
    num_signal_files = len(set(signal_files))
    if num_signal_files!=1:
        raise NotImplementedError('The header file {}'.format(header_file) \
            + ' references {} signal files; one signal file expected.'.format(num_signal_files))

    # Load the signal file.
    head, tail = os.path.split(header_file)
    signal_file = os.path.join(head, list(signal_files)[0])
    data = np.asarray(sp.io.loadmat(signal_file)['val'])

    # Check that the dimensions of the signal data in the signal file is consistent with the dimensions for the signal data given
    # in the header file.
    num_channels = len(channels)
    if np.shape(data)!=(num_channels, num_samples):
        raise ValueError('The header file {}'.format(header_file) \
            + ' is inconsistent with the dimensions of the signal file.')

    # Check that the initial value and checksums for the signal data in the signal file are consistent with the initial value and
    # checksums for the signal data given in the header file.
    for i in range(num_channels):
        if data[i, 0]!=initial_values[i]:
            raise ValueError('The initial value in header file {}'.format(header_file) \
                + ' is inconsistent with the initial value for channel'.format(channels[i]))
        if np.sum(data[i, :])!=checksums[i]:
            raise ValueError('The checksum in header file {}'.format(header_file) \
                + ' is inconsistent with the initial value for channel'.format(channels[i]))

    # Rescale the signal data using the ADC gains and ADC offsets.
    rescaled_data = np.zeros(np.shape(data), dtype=np.float32)
    for i in range(num_channels):
        rescaled_data[i, :] = (data[i, :]-offsets[i])/gains[i]

    return rescaled_data, sampling_frequency, channels

# Reorder/reselect the channels.
def reorder_recording_channels(current_data, current_channels, reordered_channels):
    if current_channels == reordered_channels:
        return current_data
    else:
        indices = list()
        for channel in reordered_channels:
            if channel in current_channels:
                i = current_channels.index(channel)
                indices.append(i)
        num_channels = len(reordered_channels)
        num_samples = np.shape(current_data)[1]
        reordered_data = np.zeros((num_channels, num_samples))
        reordered_data[:, :] = current_data[indices, :]
        return reordered_data

### Helper Challenge data I/O functions

# Load text file as a string.
def load_text_file(filename):
    with open(filename, 'r') as f:
        data = f.read()
    return data

# Parse a value.
def cast_variable(variable, variable_type, preserve_nan=True):
    if preserve_nan and is_nan(variable):
        variable = float('nan')
    else:
        if variable_type == bool:
            variable = sanitize_boolean_value(variable)
        elif variable_type == int:
            variable = sanitize_integer_value(variable)
        elif variable_type == float:
            variable = sanitize_scalar_value(variable)
        else:
            variable = variable_type(variable)
    return variable

# Get a variable from the patient metadata.
def get_variable(text, variable_name, variable_type):
    variable = None
    for l in text.split('\n'):
        if l.startswith(variable_name):
            variable = l.split(':')[1].strip()
            variable = cast_variable(variable, variable_type)
            return variable

# Get a column from the recording metadata.
def get_column(string, column, variable_type, sep='\t'):
    variables = list()
    for i, l in enumerate(string.split('\n')):
        arrs = [arr.strip() for arr in l.split(sep) if arr.strip()]
        if i==0:
            column_index = arrs.index(column)
        elif arrs:
            variable = arrs[column_index]
            variable = cast_variable(variable, variable_type)
            variables.append(variable)
    return np.asarray(variables)

# Get the patient ID variable from the patient data.
def get_patient_id(string):
    return get_variable(string, 'Patient', str)

# Get the age variable (in years) from the patient data.
def get_age(string):
    return get_variable(string, 'Age', int)

# Get the sex variable from the patient data.
def get_sex(string):
    return get_variable(string, 'Sex', str)

# Get the ROSC variable (in minutes) from the patient data.
def get_rosc(string):
    return get_variable(string, 'ROSC', int)

# Get the OHCA variable from the patient data.
def get_ohca(string):
    return get_variable(string, 'OHCA', bool)

# Get the VFib variable from the patient data.
def get_vfib(string):
    return get_variable(string, 'VFib', bool)

# Get the TTM variable (in Celsius) from the patient data.
def get_ttm(string):
    return get_variable(string, 'TTM', int)

# Get the Outcome variable from the patient data.
def get_outcome(string):
    variable = get_variable(string, 'Outcome', str)
    if variable is None or is_nan(variable):
        raise ValueError('No outcome available. Is your code trying to load labels from the hidden data?')
    if variable == 'Good':
        variable = 0
    elif variable == 'Poor':
        variable = 1
    return variable

# Get the Outcome probability variable from the patient data.
def get_outcome_probability(string):
    variable = sanitize_scalar_value(get_variable(string, 'Outcome probability', str))
    if variable is None or is_nan(variable):
        raise ValueError('No outcome available. Is your code trying to load labels from the hidden data?')
    return variable

# Get the CPC variable from the patient data.
def get_cpc(string):
    variable = sanitize_scalar_value(get_variable(string, 'CPC', str))
    if variable is None or is_nan(variable):
        raise ValueError('No CPC score available. Is your code trying to load labels from the hidden data?')
    return variable

# Get the hour number column from the patient data.
def get_hours(string):
    return get_column(string, 'Hour', int)

# Get the time column from the patient data.
def get_times(string):
    return get_column(string, 'Time', str)

# Get the quality score column from the patient data.
def get_quality_scores(string):
    return get_column(string, 'Quality', float)

# Get the recording IDs column from the patient data.
def get_recording_ids(string):
    return get_column(string, 'Record', str)

### Challenge label and output I/O functions

# Load the Challenge labels for one file.
def load_challenge_label(string):
    if os.path.isfile(string):
        string = load_text_file(string)

    outcome = get_outcome(string)
    cpc = get_cpc(string)

    return outcome, cpc

# Load the Challenge labels for all of the files in a folder.
def load_challenge_labels(folder):
    patient_folders = find_data_folders(folder)
    num_patients = len(patient_folders)

    patient_ids = list()
    outcomes = np.zeros(num_patients, dtype=np.bool_)
    cpcs = np.zeros(num_patients, dtype=np.float64)

    for i in range(num_patients):
        patient_metadata_file = os.path.join(folder, patient_folders[i], patient_folders[i] + '.txt')
        patient_metadata = load_text_file(patient_metadata_file)

        patient_ids.append(get_patient_id(patient_metadata))
        outcomes[i] = get_outcome(patient_metadata)
        cpcs[i] = get_cpc(patient_metadata)

    return patient_ids, outcomes, cpcs

# Save the Challenge outputs for one file.
def save_challenge_outputs(filename, patient_id, outcome, outcome_probability, cpc):
    # Sanitize values, e.g., in case they are a singleton array.
    outcome = sanitize_boolean_value(outcome)
    outcome_probability = sanitize_scalar_value(outcome_probability)
    cpc = sanitize_scalar_value(cpc)

    # Format Challenge outputs.
    patient_string = 'Patient: {}'.format(patient_id)
    if outcome == 0:
        outcome = 'Good'
    elif outcome == 1:
        outcome = 'Poor'
    outcome_string = 'Outcome: {}'.format(outcome)
    outcome_probability_string = 'Outcome probability: {:.3f}'.format(float(outcome_probability))
    cpc_string = 'CPC: {:.3f}'.format(int(float(cpc)) if is_integer(cpc) else float(cpc))
    output_string = patient_string + '\n' + \
        outcome_string + '\n' + outcome_probability_string + '\n' + cpc_string + '\n'

    # Write the Challenge outputs.
    if filename is not None:
        with open(filename, 'w') as f:
            f.write(output_string)

    return output_string

# Load the Challenge outputs for one file.
def load_challenge_output(string):
    if os.path.isfile(string):
        string = load_text_file(string)

    patient_id = get_patient_id(string)
    outcome = get_outcome(string)
    outcome_probability = get_outcome_probability(string)
    cpc = get_cpc(string)

    return patient_id, outcome, outcome_probability, cpc

# Load the Challenge outputs for all of the files in folder.
def load_challenge_outputs(folder, patient_ids):
    num_patients = len(patient_ids)
    outcomes = np.zeros(num_patients, dtype=np.bool_)
    outcome_probabilities = np.zeros(num_patients, dtype=np.float64)
    cpcs = np.zeros(num_patients, dtype=np.float64)

    for i in range(num_patients):
        output_file = os.path.join(folder, patient_ids[i], patient_ids[i] + '.txt')
        patient_id, outcome, outcome_probability, cpc = load_challenge_output(output_file)
        outcomes[i] = outcome
        outcome_probabilities[i] = outcome_probability
        cpcs[i] = cpc

    return outcomes, outcome_probabilities, cpcs

### Other helper functions

# Check if a variable is a number or represents a number.
def is_number(x):
    try:
        float(x)
        return True
    except (ValueError, TypeError):
        return False

# Check if a variable is an integer or represents an integer.
def is_integer(x):
    if is_number(x):
        return float(x).is_integer()
    else:
        return False

# Check if a variable is a boolean or represents a boolean.
def is_boolean(x):
    if (is_number(x) and float(x)==0) or (remove_extra_characters(x) in ('False', 'false', 'FALSE', 'F', 'f')):
        return True
    elif (is_number(x) and float(x)==1) or (remove_extra_characters(x) in ('True', 'true', 'TRUE', 'T', 't')):
        return True
    else:
        return False

# Check if a variable is a finite number or represents a finite number.
def is_finite_number(x):
    if is_number(x):
        return np.isfinite(float(x))
    else:
        return False

# Check if a variable is a NaN (not a number) or represents a NaN.
def is_nan(x):
    if is_number(x):
        return np.isnan(float(x))
    else:
        return False

# Remove any quotes, brackets (for singleton arrays), and/or invisible characters.
def remove_extra_characters(x):
    return str(x).replace('"', '').replace("'", "").replace('[', '').replace(']', '').replace(' ', '').strip()

# Sanitize boolean values.
def sanitize_boolean_value(x):
    x = remove_extra_characters(x)
    if (is_number(x) and float(x)==0) or (remove_extra_characters(str(x)) in ('False', 'false', 'FALSE', 'F', 'f')):
        return 0
    elif (is_number(x) and float(x)==1) or (remove_extra_characters(str(x)) in ('True', 'true', 'TRUE', 'T', 't')):
        return 1
    else:
        return float('nan')

# Sanitize integer values.
def sanitize_integer_value(x):
    x = remove_extra_characters(x)
    if is_integer(x):
        return int(float(x))
    else:
        return float('nan')

# Santize scalar values.
def sanitize_scalar_value(x):
    x = remove_extra_characters(x)
    if is_number(x):
        return float(x)
    else:
        return float('nan')
