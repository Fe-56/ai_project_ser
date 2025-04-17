import random
import numpy as np
import librosa

# Insipiration taken from https://www.youtube.com/watch?v=umAXGVzVvwQ


class OfflineDataAugmentation:
    def add_gaussian_noise(signal, noise_factor_range=(0, 0.5)):
        # Sample a random noise factor from specified range
        noise_factor = random.uniform(
            noise_factor_range[0], noise_factor_range[1])

        # Generate Gaussian noise based on signal's std
        noise = np.random.normal(0, signal.std(), signal.shape)

        augmented_signal = signal + noise * noise_factor
        return augmented_signal

    def time_stretch(signal, stretch_rate_range=(0.8, 1.2)):
        # Sample a random noise factor from specified range
        stretch_rate = random.uniform(
            stretch_rate_range[0], stretch_rate_range[1])

        return librosa.effects.time_stretch(signal, rate=stretch_rate)

    def pitch_shift(signal, sr, n_steps_range=(-2, 2)):
        # Sample a random noise factor from specified range
        n_steps = random.uniform(
            n_steps_range[0], n_steps_range[1])

        return librosa.effects.pitch_shift(signal, sr=sr, n_steps=n_steps)
