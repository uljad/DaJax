import numpy as np
import pandas as pd
import jax
from jax import numpy as jnp
import os
import glob

def sort_filenames(filenames):
    def extract_number(filename):
        # Split the filename to isolate the part with the number
        number_part = filename.split('_')[-1]
        # Remove the '.csv' extension and convert to integer
        number = int(number_part.replace('.csv', ''))
        return number

    # Sort the list by applying the extract_number function to each element
    sorted_filenames = sorted(filenames, key=extract_number)
    return sorted_filenames

def get_csv_files(directory):
    """
    This function returns a list of all CSV files in the specified directory.

    :param directory: The path to the directory from which to retrieve CSV files.
    :return: A list of paths to CSV files.
    """
    # Construct the path pattern to match CSV files
    path_pattern = os.path.join(directory, '*.csv')
    
    # Use glob to find all files matching the pattern
    csv_files = glob.glob(path_pattern)

    csv_files = sort_filenames(csv_files)

    return csv_files

def normalize_array(arr):
    """
    Normalize the given array.
    return normalized array and min max values for each column
    """

    # Initialize the normalized array
    normalized_array = np.zeros_like(arr, dtype=np.float64)
    
    # Arrays to store min and max for each column
    min_vals = np.min(arr, axis=0)
    max_vals = np.max(arr, axis=0)
    
    # Check for any columns with the same min and max values to prevent division by zero
    if np.any(max_vals - min_vals == 0):
        raise ValueError("One or more columns have the same min and max values.")
    
    # Normalize each column
    for i in range(arr.shape[1]):
        normalized_array[:, i] = (arr[:, i] - min_vals[i]) / (max_vals[i] - min_vals[i])
    
    return normalized_array, (min_vals, max_vals)

def normalize_array_vals(arr, min_vals, max_vals):
    """

    Normalize the given array.
    return normalized array and min max values for each column
    """
    normalized_array = (arr - min_vals)/(max_vals - min_vals+1e-9)
    
    return normalized_array

def denormalize_array(normalized_arr, min_vals, max_vals):
    # Extract the min and max values from the normalization constants
    # Denormalize each column
    denormalized_array = normalized_arr * (max_vals - min_vals) + min_vals
    
    return denormalized_array

def get_separate_episodes(df):
    """
    Extract separate episodes from a DataFrame.
    """
    episodes = []
    start_index = 0  # Start index for each new episode

    # Iterate over the DataFrame using iterrows() to get index and row
    for index, row in df.iterrows():
        if row['Done']:  # Check if the 'done' flag is True
            # Slice the DataFrame from start_index to index+1 (inclusive of the current row)
            episode = df[start_index:index]
            episodes.append(episode)
            start_index = index + 1  # Update start_index to the row after the current 'done'

    return episodes

def extract_arrays_from_df(df):
    '''
    Make arrays from the dataframe of Actions, Observations and Dones
    '''
    # Extract columns that start with 'Action_'
    action_columns = df.filter(like='Action')
    # Convert to numpy array
    action_array = action_columns.to_numpy()

    # Extract columns that start with 'Observation_'
    observation_columns = df.filter(like='Observation')
    # Convert to numpy array
    observation_array = observation_columns.to_numpy()

    # Extract columns that start with 'Observation_'
    old_observation_columns = df.filter(like='Old')
    # Convert to numpy array
    old_observation_array = old_observation_columns.to_numpy()

    # Extract columns that start with 'Done'
    dones = df.filter(like='Done')
    # Convert to numpy array
    dones_array = dones.to_numpy().astype(int)

    # Extract columns that start with 'Reward'
    rewards = df.filter(like='Reward')
    # Convert to numpy array
    rewards_array = rewards.to_numpy()
    
    return action_array, old_observation_array, observation_array, rewards_array, dones_array

def create_action_observation_io(actions, old_observation, observation, rewards, dones):
    '''
    Concatenate actions, old_observations and observations
    '''
    assert actions.shape[0] == old_observation.shape[0] == observation.shape[0] == rewards.shape[0] == dones.shape[0]
    input_list = []
    output_list = []
    # print(actions.shape, old_observation.shape, observation.shape, rewards.shape, dones.shape)
    for i in range(actions.shape[0]):
        input_list.append(np.concatenate((actions[i], old_observation[i])))
        output_list.append(np.concatenate((observation[i], rewards[i])))

    return np.array(input_list), np.array(output_list)

def generate_dataset_from_io(key,context_seqs, target_obs_seqs, validation_ratio):

    perm_idx = jax.random.permutation(key, context_seqs.shape[0])
    context_seqs_shuffled = context_seqs[perm_idx]
    target_obs_seqs_shuffled = target_obs_seqs[perm_idx]

    num_samples = context_seqs.shape[0]
    num_validation_samples = int(num_samples * validation_ratio)

    x_train = context_seqs_shuffled[:-num_validation_samples]
    y_train = target_obs_seqs_shuffled[:-num_validation_samples]
    x_test = context_seqs_shuffled[-num_validation_samples:]
    y_test = target_obs_seqs_shuffled[-num_validation_samples:]
    return x_train, y_train, x_test, y_test

def normalize_observation(arr1, arr2, arr3, arr4):
    
    # Stack all arrays vertically
    combined = np.vstack((arr1, arr2, arr3, arr4))
    
    # Dictionary to store normalization constants
    normalization_constants = {}

    # Normalize each column in the combined array
    for i in range(4):  # Iterate through each dimension
        column_min = np.min(combined[:, i])
        column_max = np.max(combined[:, i])
        combined[:, i] = (combined[:, i] - column_min) / (column_max - column_min)
        normalization_constants[f'dim_{i}'] = (column_min, column_max)
    
    # Split the combined array back into the original arrays
    end1 = len(arr1)
    end2 = end1 + len(arr2)
    end3 = end2 + len(arr3)

    normalized_arr1 = combined[:end1]
    normalized_arr2 = combined[end1:end2]
    normalized_arr3 = combined[end2:end3]
    normalized_arr4 = combined[end3:]
    
    return normalized_arr1, normalized_arr2, normalized_arr3, normalized_arr4, normalization_constants

def normalize_reward(arr1, arr2):
    # Combine the arrays to find global min and max
    combined = np.concatenate((arr1, arr2))
    
    # Calculate global min and max for normalization
    global_min = np.min(combined)
    global_max = np.max(combined)
    
    # Calculate normalization constants
    normalization_constants = {'min': global_min, 'max': global_max}
    
    # Normalize both arrays if max and min are not equal, else set to zero
    if global_max > global_min:
        normalized_arr1 = (arr1 - global_min) / (global_max - global_min)
        normalized_arr2 = (arr2 - global_min) / (global_max - global_min)
    else:
        normalized_arr1 = np.zeros_like(arr1)
        normalized_arr2 = np.zeros_like(arr2)
    
    return normalized_arr1, normalized_arr2, normalization_constants

def create_full_dataset(key,dir_path,validation_ratio = 0.2, normalize=False):

    df = pd.read_csv(dir_path)
    episodes_list = get_separate_episodes(df)

    XR = []
    YR = []
    XS = []
    YS = []

    for ep in episodes_list:
        actions_arr, old_obs_arr, obs_arr, rewards_arr, dones_arr = extract_arrays_from_df(ep)
        input_arr, output_arr = create_action_observation_io(actions_arr, old_obs_arr, obs_arr, rewards_arr, dones_arr)
        x_tr, y_tr, x_s, y_s = generate_dataset_from_io(key, input_arr, output_arr, validation_ratio)
        XR.append(x_tr)
        YR.append(y_tr)
        XS.append(x_s)
        YS.append(y_s)    

    x_test = np.concatenate(XS)
    y_test = np.concatenate(YS)
    x_train = np.concatenate(XR)
    y_train = np.concatenate(YR)

    return x_train, y_train, x_test, y_test

def normalize_final_dataset(x_train, y_train, x_test, y_test):
    x_tr_obs = x_train[:, 1:]
    x_te_obs = x_test[:, 1:]
    y_tr_obs = y_train[:, :-1]
    y_te_obs = y_test[:, :-1]
    y_tr_reward = y_train[:, -1]
    y_te_reward = y_test[:, -1]

    x_tr_obs_norm, x_te_obs_norm, y_tr_obs_norm, y_te_obs_norm, normalization_constants_obs= normalize_observation(x_tr_obs, x_te_obs, y_tr_obs, y_te_obs)
    y_tr_reward_norm, y_te_reward_norm, normalization_constants_reward = normalize_reward(y_tr_reward, y_te_reward)

    x_train[:, 1:] = x_tr_obs_norm
    x_test[:, 1:] = x_te_obs_norm
    y_train[:, :-1] = y_tr_obs_norm
    y_test[:, :-1] = y_te_obs_norm
    y_train[:, -1] = y_tr_reward_norm
    y_test[:, -1] = y_te_reward_norm

    return x_train, y_train, x_test, y_test, normalization_constants_obs, normalization_constants_reward