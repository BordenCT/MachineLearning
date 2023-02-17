import numpy as np
import tensorflow as tf


def get_inputs(filename: str = None) -> np.float:
    """
    :param filename: Which file to retrieve the inputs data from.
    :return: Column of inputs values
    """
    npz = np.load(filename)
    return npz['Inputs'].astype(np.float)


def get_targets(filename: str = None) -> np.int:
    """
    :param filename: Which file to retrieve the targets data from.
    :return: Column of targets values
    """
    npz = np.load(filename)
    return npz['Targets'].astype(np.int)

def build_model(train_inputs,
                train_targets,
                validation_inputs,
                validation_targets,
                test_inputs,
                test_targets) -> None:

    # Number of Predictors
    input_size = 10
    # Number of possible outcomes.
    output_size = 2
    # Depth of network
    hidden_layer_size = 50
    # Batch Size
    batch_size = 100
    # Number of epochs
    max_epochs = 100

    model = tf.keras.Sequential([tf.keras.layers.Dense(hidden_layer_size, activation='relu'),
                                 tf.keras.layers.Dense(hidden_layer_size, activation='relu'),
                                 tf.keras.layers.Dense(output_size, activation='softmax')])

    model.compile(optimizer='adam', loss='sparce_categorical_crossentropy', metrics=['accuracy'])

    model.fit(train_inputs, train_targets,
              batch_size=batch_size,
              epochs=max_epochs,
              validation_data=(validation_inputs, validation_targets),
              verbose=2)


def main() -> None:
    train_file = 'Audiobooks_data_train.npz'
    validation_file = 'Audiobooks_data_validation.npz'
    test_file = 'Audiobooks_data_test.npz'

    train_inputs = get_inputs(train_file)
    train_targets = get_targets(train_file)

    validation_inputs = get_inputs(validation_file)
    validation_targets = get_targets(validation_file)

    test_inputs = get_inputs(test_file)
    test_targets = get_targets(test_file)

    build_model(train_inputs, train_targets, validation_inputs, validation_targets, test_inputs, test_targets)


if __name__ == '__main__':
    main()