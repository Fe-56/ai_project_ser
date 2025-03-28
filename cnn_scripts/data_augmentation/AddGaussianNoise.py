import random
import torch
import numpy as np


class AddGaussianNoise(object):
    def __init__(self, noise_factor_range=(0.01, 0.5)):
        """
        Args:
            noise_factor_range (tuple): Range (min, max) from which to sample the noise factor.
        """
        self.noise_factor_range = noise_factor_range

    def __call__(self, tensor):
        # Convert to numpy for consistent handling
        signal = tensor.numpy()

        # Compute signal's standard deviation
        signal_std = signal.std()

        # Sample a random noise factor from specified range
        noise_factor = random.uniform(
            self.noise_factor_range[0], self.noise_factor_range[1])

        # Generate Gaussian noise based on signal's std
        noise = np.random.normal(0, signal_std, signal.shape)

        # Apply the noise factor to the noise
        augmented_signal = signal + noise * noise_factor

        # Convert back to tensor
        return torch.from_numpy(augmented_signal).type_as(tensor)

    def __repr__(self):
        return f'{self.__class__.__name__}(noise_factor_range={self.noise_factor_range})'
