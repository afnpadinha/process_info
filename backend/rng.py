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

    def gamma(self, shape, scale):
        # random.Random uses gammavariate(alpha, beta)
        return self.random.gammavariate(shape, scale)

    def laplace(self, mu, b):
        """
        Generate Laplace noise using the difference of two Exponentials.
        L(mu, b) is equivalent to mu + Exponential(1/b) - Exponential(1/b).
        """
        # The scale of the exponential is the 'b' parameter of Laplace
        e1 = self.expo(1/b)
        e2 = self.expo(1/b)
        return mu + e1 - e2
