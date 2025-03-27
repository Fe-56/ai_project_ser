import random
import torch

class AddGaussianNoise(object):
    def __init__(self, mean=0.0, std_range=(0.0, 0.1)):
        """
        Args:
            mean (float): Mean of the Gaussian noise.
            std_range (tuple): Range (min, max) from which to sample the noise std.
        """
        self.mean = mean
        self.std_range = std_range

    def __call__(self, tensor):
        # Sample a random standard deviation for this call
        std = random.uniform(self.std_range[0], self.std_range[1])
        noise = torch.randn(tensor.size()) * std + self.mean
        return tensor + noise

    def __repr__(self):
        return f'{self.__class__.__name__}(mean={self.mean}, std_range={self.std_range})'