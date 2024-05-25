"""
We want to create a Random Sin Generator :)
"""
import torch
import numpy as np
import matplotlib.pyplot as plt


def sin_generator(amp=None, phase=None):
    """ Generate a random sine wave.
    """
    if amp is None:
        amp = 0.1 + (5 - 0.1) * torch.rand(1).item()
    if phase is None:
        phase = 0.0 + (np.pi - 0.0) * torch.rand(1).item()

    def _gen(sample):
        return amp * torch.sin(sample - phase)

    return _gen


def generate_x(sample_size, min_value=-5.0, max_value=5.0):
    """ Generate random X values to be used for training or testing
    """
    return torch.unsqueeze(torch.linspace(min_value, max_value, sample_size), 1)


def sample_meta_data(num_samples_per_task=1000, num_tasks=2,
                     num_train_samples_per_task=800, num_test_samples_per_task=200):
    """ Generate a meta-data for training and testing.
    """
    train_set = []
    test_set = []

    # train data
    for task in range(num_tasks):
        sin_gen = sin_generator()
        gen_x = generate_x(num_samples_per_task)
        gen_y = sin_gen(gen_x)

        # Split into train and test sets
        indices = range(0, num_samples_per_task)
        train_indices = indices[:num_train_samples_per_task]
        test_indices = indices[num_train_samples_per_task:num_train_samples_per_task + num_test_samples_per_task]

        x_train, y_train = gen_x[train_indices], gen_y[train_indices]
        x_test, y_test = gen_x[test_indices], gen_y[test_indices]

        train_set.append((x_train, y_train))
        test_set.append((x_test, y_test))

    return train_set, test_set


if __name__ == '__main__':
    sample_meta_data()
