import random
import torch


class AddGaussianNoiseStochastic(object):
    def __init__(self, mean=0.0, std_range=(0.0, 0.1), p=0.5):
        """
        Args:
            mean (float): Mean of the Gaussian noise.
            std_range (tuple): Range (min, max) from which to sample the noise standard deviation.
            p (float): Probability of applying the noise.
        """
        self.mean = mean
        self.std_range = std_range
        self.p = p

    def __call__(self, tensor):
        # Apply noise with probability p
        if random.random() < self.p:
            std = random.uniform(*self.std_range)
            noise = torch.randn(tensor.size()) * std + self.mean
            return tensor + noise
        else:
            return tensor

    def __repr__(self):
        return f'{self.__class__.__name__}(mean={self.mean}, std_range={self.std_range}, p={self.p})'
