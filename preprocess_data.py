"""
    Two class data preprocessing:
        In order to handle more classes. Additional functionality and balancing of the data must become dynamic to the
        size of the dataset.
"""

import numpy as np
from sklearn import preprocessing


def main() -> None:
    # Extract the data from the CSV
    raw_csv_data = np.loadtxt('Audiobooks_data.csv', delimiter=',')

    # This skips the first column of the data
    unscaled_inputs_all = raw_csv_data[:, 1:-1]
    # This takes the last column for the targets
    targets_all = raw_csv_data[:, -1]

    ##
    # Shuffle the data
    shuffled_indices = np.arange(unscaled_inputs_all.shape[0])
    np.random.shuffle(shuffled_indices)

    # This skips the first column of the data
    unscaled_inputs_all = unscaled_inputs_all[shuffled_indices]
    # This takes the last column for the targets
    targets_all = targets_all[shuffled_indices]

    ##
    # Balance the dataset
    num_one_targets = int(np.sum(targets_all))
    zero_targets_counter = 0
    indices_to_remove = []

    for target in range(targets_all.shape[0]):
        if targets_all[target] == 0:
            zero_targets_counter += 1
            if zero_targets_counter > num_one_targets:
                indices_to_remove.append(target)

    unscaled_inputs_equal_priors = np.delete(unscaled_inputs_all, indices_to_remove, axis=0)
    targets_equal_priors = np.delete(targets_all, indices_to_remove, axis=0)

    ##
    # Standardize the Inputs
    scaled_inputs = preprocessing.scale(unscaled_inputs_equal_priors)

    shuffled_indices = np.arange(scaled_inputs.shape[0])
    np.random.shuffle(shuffled_indices)

    shuffled_inputs = scaled_inputs[shuffled_indices]
    shuffled_targets = targets_equal_priors[shuffled_indices]

    ##
    # Split the data into train, validation and test
    samples_count = shuffled_inputs.shape[0]

    train_samples_count = int(0.8 * samples_count)
    validation_samples_count = int(0.1 * samples_count)
    test_samples_count = samples_count - train_samples_count - validation_samples_count

    train_inputs = shuffled_inputs[:train_samples_count]
    train_targets = shuffled_targets[:train_samples_count]

    validation_inputs = shuffled_inputs[train_samples_count:train_samples_count + validation_samples_count]
    validation_targets = shuffled_targets[train_samples_count:train_samples_count + validation_samples_count]

    test_inputs = shuffled_inputs[train_samples_count + validation_samples_count:]
    test_targets = shuffled_targets[train_samples_count + validation_samples_count:]

    print(np.sum(train_targets), train_samples_count, np.sum(train_targets)/train_samples_count)
    print(np.sum(validation_targets), validation_samples_count, np.sum(validation_targets)/validation_samples_count)
    print(np.sum(test_targets), test_samples_count, np.sum(test_targets)/test_samples_count)

    # Save the three datasets in *.npz
    np.savez('Audiobooks_data_train', inputs=train_inputs, targets=train_targets)
    np.savez('Audiobooks_data_validation', inputs=validation_inputs, targets=validation_targets)
    np.savez('Audiobooks_data_test', inputs=test_inputs, targets=test_targets)


if __name__ == '__main__':
    main()
