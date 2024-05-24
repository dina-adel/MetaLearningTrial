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


if __name__ == '__main__':
    for _ in range(3):
        func = sin_generator()
        x = generate_x(sample_size=1000)
        y = func(x)
        plt.plot(x, y)

    plt.show()
