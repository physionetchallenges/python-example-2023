#!/usr/bin/env python

# Edit this script to add your team's code. Some functions are *required*, but you can edit most parts of the required functions,
# change or remove non-required functions, and add your own functions.

################################################################################
#
# Optional libraries, functions, and variables. You can change or remove them.
#
################################################################################

from helper_code import *
import numpy as np, os, sys
import mne
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import joblib

################################################################################
#
# Required functions. Edit these functions to add your code, but do not change the arguments of the functions.
#
################################################################################

# Train your model.
def train_challenge_model(data_folder, model_folder, verbose):
    # Find data files.
    if verbose >= 1:
        print('Finding the Challenge data...')

    patient_ids = find_data_folders(data_folder)
    num_patients = len(patient_ids)

    if num_patients==0:
        raise FileNotFoundError('No data was provided.')

    # Create a folder for the model if it does not already exist.
    os.makedirs(model_folder, exist_ok=True)

    # Extract the features and labels.
    if verbose >= 1:
        print('Extracting features and labels from the Challenge data...')

    features = list()
    outcomes = list()
    cpcs = list()

    for i in range(num_patients):
        if verbose >= 2:
            print('    {}/{}...'.format(i+1, num_patients))

        current_features = get_features(data_folder, patient_ids[i])
        features.append(current_features)

        # Extract labels.
        patient_metadata = load_challenge_data(data_folder, patient_ids[i])
        current_outcome = get_outcome(patient_metadata)
        outcomes.append(current_outcome)
        current_cpc = get_cpc(patient_metadata)
        cpcs.append(current_cpc)

    features = np.vstack(features)
    outcomes = np.vstack(outcomes)
    cpcs = np.vstack(cpcs)

    # Train the models.
    if verbose >= 1:
        print('Training the Challenge model on the Challenge data...')

    # Define parameters for random forest classifier and regressor.
    n_estimators   = 123  # Number of trees in the forest.
    max_leaf_nodes = 456  # Maximum number of leaf nodes in each tree.
    random_state   = 789  # Random state; set for reproducibility.

    # Impute any missing features; use the mean value by default.
    imputer = SimpleImputer().fit(features)

    # Train the models.
    features = imputer.transform(features)
    outcome_model = RandomForestClassifier(
        n_estimators=n_estimators, max_leaf_nodes=max_leaf_nodes, random_state=random_state).fit(features, outcomes.ravel())
    cpc_model = RandomForestRegressor(
        n_estimators=n_estimators, max_leaf_nodes=max_leaf_nodes, random_state=random_state).fit(features, cpcs.ravel())

    # Save the models.
    save_challenge_model(model_folder, imputer, outcome_model, cpc_model)

    if verbose >= 1:
        print('Done.')

# Load your trained models. This function is *required*. You should edit this function to add your code, but do *not* change the
# arguments of this function.
def load_challenge_models(model_folder, verbose):
    filename = os.path.join(model_folder, 'models.sav')
    return joblib.load(filename)

# Run your trained models. This function is *required*. You should edit this function to add your code, but do *not* change the
# arguments of this function.
def run_challenge_models(models, data_folder, patient_id, verbose):
    imputer = models['imputer']
    outcome_model = models['outcome_model']
    cpc_model = models['cpc_model']

    # Extract features.
    features = get_features(data_folder, patient_id)
    features = features.reshape(1, -1)

    # Impute missing data.
    features = imputer.transform(features)

    # Apply models to features.
    outcome = outcome_model.predict(features)[0]
    outcome_probability = outcome_model.predict_proba(features)[0, 1]
    cpc = cpc_model.predict(features)[0]

    # Ensure that the CPC score is between (or equal to) 1 and 5.
    cpc = np.clip(cpc, 1, 5)

    return outcome, outcome_probability, cpc

################################################################################
#
# Optional functions. You can change or remove these functions and/or add new functions.
#
################################################################################

# Save your trained model.
def save_challenge_model(model_folder, imputer, outcome_model, cpc_model):
    d = {'imputer': imputer, 'outcome_model': outcome_model, 'cpc_model': cpc_model}
    filename = os.path.join(model_folder, 'models.sav')
    joblib.dump(d, filename, protocol=0)

# Preprocess data.
def preprocess_data(data, sampling_frequency, utility_frequency):
    # Define the bandpass frequencies.
    passband = [0.1, 30.0]

    # Promote the data to double precision because these libraries expect double precision.
    data = np.asarray(data, dtype=np.float64)

    # If the utility frequency is between bandpass frequencies, then apply a notch filter.
    if utility_frequency is not None and passband[0] <= utility_frequency <= passband[1]:
        data = mne.filter.notch_filter(data, sampling_frequency, utility_frequency, n_jobs=4, verbose='error')

    # Apply a bandpass filter.
    data = mne.filter.filter_data(data, sampling_frequency, passband[0], passband[1], n_jobs=4, verbose='error')

    # Resample the data.
    if sampling_frequency % 2 == 0:
        resampling_frequency = 128
    else:
        resampling_frequency = 125
    lcm = np.lcm(int(round(sampling_frequency)), int(round(resampling_frequency)))
    up = int(round(lcm / sampling_frequency))
    down = int(round(lcm / resampling_frequency))
    resampling_frequency = sampling_frequency * up / down
    data = scipy.signal.resample_poly(data, up, down, axis=1)

    # Scale the data to the interval [-1, 1].
    min_value = np.min(data)
    max_value = np.max(data)
    if min_value != max_value:
        data = 2.0 / (max_value - min_value) * (data - 0.5 * (min_value + max_value))
    else:
        data = 0 * data

    return data, resampling_frequency

# Extract features.
def get_features(data_folder, patient_id):
    # Load patient data.
    patient_metadata = load_challenge_data(data_folder, patient_id)
    recording_ids = find_recording_files(data_folder, patient_id)
    num_recordings = len(recording_ids)

    # Extract patient features.
    patient_features = get_patient_features(patient_metadata)

    # Extract EEG features.
    eeg_channels = ['F3', 'P3', 'F4', 'P4']
    group = 'EEG'

    if num_recordings > 0:
        recording_id = recording_ids[-1]
        recording_location = os.path.join(data_folder, patient_id, '{}_{}'.format(recording_id, group))
        if os.path.exists(recording_location + '.hea'):
            data, channels, sampling_frequency = load_recording_data(recording_location)
            utility_frequency = get_utility_frequency(recording_location + '.hea')

            if all(channel in channels for channel in eeg_channels):
                data, channels = reduce_channels(data, channels, eeg_channels)
                data, sampling_frequency = preprocess_data(data, sampling_frequency, utility_frequency)
                data = np.array([data[0, :] - data[1, :], data[2, :] - data[3, :]]) # Convert to bipolar montage: F3-P3 and F4-P4
                eeg_features = get_eeg_features(data, sampling_frequency).flatten()
            else:
                eeg_features = float('nan') * np.ones(8) # 2 bipolar channels * 4 features / channel
        else:
            eeg_features = float('nan') * np.ones(8) # 2 bipolar channels * 4 features / channel
    else:
        eeg_features = float('nan') * np.ones(8) # 2 bipolar channels * 4 features / channel

    # Extract ECG features.
    ecg_channels = ['ECG', 'ECGL', 'ECGR', 'ECG1', 'ECG2']
    group = 'ECG'

    if num_recordings > 0:
        recording_id = recording_ids[0]
        recording_location = os.path.join(data_folder, patient_id, '{}_{}'.format(recording_id, group))
        if os.path.exists(recording_location + '.hea'):
            data, channels, sampling_frequency = load_recording_data(recording_location)
            utility_frequency = get_utility_frequency(recording_location + '.hea')

            data, channels = reduce_channels(data, channels, ecg_channels)
            data, sampling_frequency = preprocess_data(data, sampling_frequency, utility_frequency)
            features = get_ecg_features(data)
            ecg_features = expand_channels(features, channels, ecg_channels).flatten()
        else:
            ecg_features = float('nan') * np.ones(10) # 5 channels * 2 features / channel
    else:
        ecg_features = float('nan') * np.ones(10) # 5 channels * 2 features / channel

    # Extract features.
    return np.hstack((patient_features, eeg_features, ecg_features))

# Extract patient features from the data.
def get_patient_features(data):
    age = get_age(data)
    sex = get_sex(data)
    rosc = get_rosc(data)
    ohca = get_ohca(data)
    shockable_rhythm = get_shockable_rhythm(data)
    ttm = get_ttm(data)

    sex_features = np.zeros(2, dtype=int)
    if sex == 'Female':
        female = 1
        male   = 0
        other  = 0
    elif sex == 'Male':
        female = 0
        male   = 1
        other  = 0
    else:
        female = 0
        male   = 0
        other  = 1

    features = np.array((age, female, male, other, rosc, ohca, shockable_rhythm, ttm))

    return features

# Extract features from the EEG data.
def get_eeg_features(data, sampling_frequency):
    num_channels, num_samples = np.shape(data)

    if num_samples > 0:
        delta_psd, _ = mne.time_frequency.psd_array_welch(data, sfreq=sampling_frequency,  fmin=0.5,  fmax=8.0, verbose=False)
        theta_psd, _ = mne.time_frequency.psd_array_welch(data, sfreq=sampling_frequency,  fmin=4.0,  fmax=8.0, verbose=False)
        alpha_psd, _ = mne.time_frequency.psd_array_welch(data, sfreq=sampling_frequency,  fmin=8.0, fmax=12.0, verbose=False)
        beta_psd,  _ = mne.time_frequency.psd_array_welch(data, sfreq=sampling_frequency, fmin=12.0, fmax=30.0, verbose=False)

        delta_psd_mean = np.nanmean(delta_psd, axis=1)
        theta_psd_mean = np.nanmean(theta_psd, axis=1)
        alpha_psd_mean = np.nanmean(alpha_psd, axis=1)
        beta_psd_mean  = np.nanmean(beta_psd,  axis=1)
    else:
        delta_psd_mean = theta_psd_mean = alpha_psd_mean = beta_psd_mean = float('nan') * np.ones(num_channels)

    features = np.array((delta_psd_mean, theta_psd_mean, alpha_psd_mean, beta_psd_mean)).T

    return features

# Extract features from the ECG data.
def get_ecg_features(data):
    num_channels, num_samples = np.shape(data)

    if num_samples > 0:
        mean = np.mean(data, axis=1)
        std  = np.std(data, axis=1)
    elif num_samples == 1:
        mean = np.mean(data, axis=1)
        std  = float('nan') * np.ones(num_channels)
    else:
        mean = float('nan') * np.ones(num_channels)
        std = float('nan') * np.ones(num_channels)

    features = np.array((mean, std)).T

    return features
