import math
import random


class RNG:
    # Wrapper around random.Random for the server, keeps seed handling in one place.
    def __init__(self, seed=None):
        self.random = random.Random(seed)

    def uniform(self, low=0.0, high=1.0):
        return self.random.uniform(low, high)

    def uniform_int(self, low, high):
        return self.random.randint(low, high)
    
    def normal(self, mean=0.0, std=1.0):
        return self.random.gauss(mean, std)
    
    def expo(self, lam):
        # Inverse transform sampling for exponential distribution.
        u = self.uniform(0.0, 1.0)
        return (-math.log(1 - u) / lam)
