"""
Random Number Generator Module for Information Processing Class.

This module implements various probability distributions using fundamental methods,
demonstrating the mathematical foundations of random variable generation.

Key Concepts:
- Inverse Transform Sampling: If U ~ Uniform(0,1) and F is a CDF, then F^(-1)(U) ~ F
- Box-Muller Transform: Generates normal distribution from two uniform samples
- Acceptance-Rejection: Alternative method for complex distributions

Author: Information Processing Class Project
"""

import math
import random


class RNG:
    """
    Random Number Generator wrapper with educational implementations.
    
    This class provides various probability distributions implemented from
    first principles, demonstrating the mathematical foundations of
    random variable generation.
    
    Attributes:
        random: Instance of random.Random for reproducible sampling
    """
    
    def __init__(self, seed=None):
        """
        Initialize RNG with optional seed for reproducibility.
        
        Args:
            seed: Optional seed value for reproducible random sequences.
                  If None, uses system time for initialization.
        """
        self.random = random.Random(seed)
        self._spare_normal = None  # For Box-Muller optimization

    # =========================================================================
    # CONTINUOUS DISTRIBUTIONS
    # =========================================================================

    def uniform(self, low=0.0, high=1.0):
        """
        Uniform distribution over [low, high].
        
        The uniform distribution is the foundation for generating all other
        distributions. Every value in the interval has equal probability.
        
        PDF: f(x) = 1/(high - low) for x ∈ [low, high]
        Mean: (low + high) / 2
        Variance: (high - low)² / 12
        
        Args:
            low: Lower bound of the interval (inclusive)
            high: Upper bound of the interval (inclusive)
            
        Returns:
            Random sample from Uniform(low, high)
        """
        return self.random.uniform(low, high)

    def uniform_int(self, low, high):
        """
        Discrete uniform distribution over integers [low, high].
        
        Each integer in the range has equal probability 1/(high - low + 1).
        
        Args:
            low: Lower bound (inclusive)
            high: Upper bound (inclusive)
            
        Returns:
            Random integer from {low, low+1, ..., high}
        """
        return self.random.randint(low, high)

    def normal(self, mean=0.0, std=1.0):
        """
        Normal (Gaussian) distribution using Python's built-in implementation.
        
        PDF: f(x) = (1/√(2πσ²)) * exp(-(x-μ)²/(2σ²))
        
        Args:
            mean: Mean (μ) of the distribution
            std: Standard deviation (σ) of the distribution
            
        Returns:
            Random sample from N(mean, std²)
            
        See Also:
            normal_box_muller: Educational implementation using Box-Muller transform
        """
        return self.random.gauss(mean, std)

    def normal_box_muller(self, mean=0.0, std=1.0):
        """
        Normal distribution using Box-Muller transform.
        
        The Box-Muller transform converts two independent uniform random
        variables into two independent standard normal random variables.
        
        Mathematical Foundation:
        -----------------------
        Given U₁, U₂ ~ Uniform(0,1) independent:
        
        Z₀ = √(-2·ln(U₁)) · cos(2π·U₂)
        Z₁ = √(-2·ln(U₁)) · sin(2π·U₂)
        
        Then Z₀, Z₁ ~ N(0,1) independent.
        
        Proof Sketch:
        The transformation uses polar coordinates. The product of two
        independent standard normals has a radial component that is
        Rayleigh distributed (√(-2·ln(U))) and an angular component
        that is uniform over [0, 2π].
        
        This implementation caches the spare value (Z₁) for efficiency.
        
        Args:
            mean: Mean (μ) of the distribution
            std: Standard deviation (σ) of the distribution
            
        Returns:
            Random sample from N(mean, std²)
        """
        # Use cached spare if available (from previous call)
        if self._spare_normal is not None:
            z = self._spare_normal
            self._spare_normal = None
            return mean + std * z
        
        # Generate two uniform samples
        u1 = self.uniform(0.0, 1.0)
        u2 = self.uniform(0.0, 1.0)
        
        # Avoid log(0)
        while u1 == 0:
            u1 = self.uniform(0.0, 1.0)
        
        # Box-Muller transform
        magnitude = math.sqrt(-2.0 * math.log(u1))
        angle = 2.0 * math.pi * u2
        
        z0 = magnitude * math.cos(angle)
        z1 = magnitude * math.sin(angle)
        
        # Cache the spare for next call
        self._spare_normal = z1
        
        return mean + std * z0

    def normal_marsaglia(self, mean=0.0, std=1.0):
        """
        Normal distribution using Marsaglia polar method.
        
        The Marsaglia polar method is a variation of Box-Muller that avoids
        computing trigonometric functions by using rejection sampling.
        
        Mathematical Foundation:
        -----------------------
        1. Generate U, V ~ Uniform(-1, 1)
        2. Compute S = U² + V²
        3. Reject if S ≥ 1 or S = 0 (keeps points inside unit circle)
        4. Return U · √(-2·ln(S)/S)
        
        The key insight is that (U/√S, V/√S) is uniformly distributed
        on the unit circle, avoiding explicit trigonometry.
        
        Args:
            mean: Mean (μ) of the distribution
            std: Standard deviation (σ) of the distribution
            
        Returns:
            Random sample from N(mean, std²)
        """
        while True:
            u = self.uniform(-1.0, 1.0)
            v = self.uniform(-1.0, 1.0)
            s = u * u + v * v
            
            if 0 < s < 1:
                break
        
        multiplier = math.sqrt(-2.0 * math.log(s) / s)
        z = u * multiplier
        
        return mean + std * z

    def expo(self, lam):
        """
        Exponential distribution using inverse transform sampling.
        
        The exponential distribution models the time between events in a
        Poisson process with rate λ.
        
        Mathematical Foundation:
        -----------------------
        CDF: F(x) = 1 - e^(-λx) for x ≥ 0
        
        Inverse Transform Method:
        If U ~ Uniform(0,1), we want X such that F(X) = U
        
        Solving: U = 1 - e^(-λX)
                 e^(-λX) = 1 - U
                 -λX = ln(1 - U)
                 X = -ln(1 - U) / λ
        
        Note: Since U and (1-U) have the same distribution, we can also
        use X = -ln(U) / λ, but we use (1-U) to avoid log(0).
        
        PDF: f(x) = λ·e^(-λx) for x ≥ 0
        Mean: 1/λ
        Variance: 1/λ²
        
        Args:
            lam: Rate parameter λ > 0 (events per unit time)
            
        Returns:
            Random sample from Exp(λ)
        """
        u = self.uniform(0.0, 1.0)
        # Avoid log(0)
        while u == 1.0:
            u = self.uniform(0.0, 1.0)
        return -math.log(1.0 - u) / lam

    def expo_library(self, lam):
        """
        Exponential distribution using Python's built-in.
        
        Provided for comparison with the manual inverse transform implementation.
        
        Args:
            lam: Rate parameter λ > 0
            
        Returns:
            Random sample from Exp(λ)
        """
        return self.random.expovariate(lam)

    def gamma(self, alpha, beta=1.0):
        """
        Gamma distribution.
        
        The gamma distribution generalizes the exponential distribution.
        When α is a positive integer, it represents the sum of α independent
        exponential random variables (Erlang distribution).
        
        PDF: f(x) = (β^α / Γ(α)) · x^(α-1) · e^(-βx) for x > 0
        Mean: α/β
        Variance: α/β²
        
        Special Cases:
        - Gamma(1, λ) = Exponential(λ)
        - Gamma(n/2, 1/2) = Chi-squared(n)
        - Gamma(k, k/μ) approaches Normal(μ, μ²/k) as k → ∞
        
        Args:
            alpha: Shape parameter α > 0
            beta: Rate parameter β > 0 (default 1.0)
            
        Returns:
            Random sample from Gamma(α, β)
        """
        return self.random.gammavariate(alpha, 1.0 / beta)

    def beta(self, alpha, beta):
        """
        Beta distribution over [0, 1].
        
        The beta distribution is commonly used to model probabilities and
        proportions. It is the conjugate prior for the Bernoulli distribution.
        
        PDF: f(x) = (Γ(α+β)/(Γ(α)·Γ(β))) · x^(α-1) · (1-x)^(β-1)
        Mean: α/(α+β)
        Variance: αβ/((α+β)²(α+β+1))
        
        Special Cases:
        - Beta(1, 1) = Uniform(0, 1)
        - Beta(1/2, 1/2) = Arcsine distribution
        
        Args:
            alpha: Shape parameter α > 0
            beta: Shape parameter β > 0
            
        Returns:
            Random sample from Beta(α, β)
        """
        return self.random.betavariate(alpha, beta)

    # =========================================================================
    # DISCRETE DISTRIBUTIONS
    # =========================================================================

    def bernoulli(self, p):
        """
        Bernoulli distribution (single coin flip).
        
        The Bernoulli distribution models a single trial with two outcomes:
        success (1) with probability p, failure (0) with probability 1-p.
        
        PMF: P(X=1) = p, P(X=0) = 1-p
        Mean: p
        Variance: p(1-p)
        
        Implementation: Compare uniform sample to threshold p.
        
        Args:
            p: Probability of success, 0 ≤ p ≤ 1
            
        Returns:
            1 with probability p, 0 otherwise
        """
        return 1 if self.uniform() < p else 0

    def binomial(self, n, p):
        """
        Binomial distribution (sum of n Bernoulli trials).
        
        The binomial distribution models the number of successes in n
        independent Bernoulli trials, each with success probability p.
        
        PMF: P(X=k) = C(n,k) · p^k · (1-p)^(n-k)
        Mean: np
        Variance: np(1-p)
        
        Implementation: Sum of n independent Bernoulli(p) trials.
        For large n, more efficient algorithms exist (e.g., BTPE algorithm).
        
        Args:
            n: Number of trials (positive integer)
            p: Probability of success in each trial, 0 ≤ p ≤ 1
            
        Returns:
            Number of successes (integer in {0, 1, ..., n})
        """
        return sum(self.bernoulli(p) for _ in range(n))

    def binomial_optimized(self, n, p):
        """
        Binomial distribution using Python's built-in (optimized).
        
        For large n, this is more efficient than summing Bernoulli trials.
        Uses BTPE algorithm for n*p >= 10 and n*(1-p) >= 10.
        
        Args:
            n: Number of trials
            p: Success probability
            
        Returns:
            Number of successes
        """
        return sum(1 for _ in range(n) if self.uniform() < p)

    def geometric(self, p):
        """
        Geometric distribution (trials until first success).
        
        The geometric distribution models the number of Bernoulli trials
        needed to get the first success.
        
        Mathematical Foundation:
        -----------------------
        PMF: P(X=k) = (1-p)^(k-1) · p for k = 1, 2, 3, ...
        CDF: F(k) = 1 - (1-p)^k
        
        Inverse Transform: 
        Solving U = 1 - (1-p)^X:
        X = ⌈ln(1-U) / ln(1-p)⌉
        
        Mean: 1/p
        Variance: (1-p)/p²
        
        Args:
            p: Probability of success, 0 < p ≤ 1
            
        Returns:
            Number of trials until first success (k ≥ 1)
        """
        if p == 1.0:
            return 1
        u = self.uniform()
        # Avoid log(0)
        while u == 1.0:
            u = self.uniform()
        return int(math.ceil(math.log(1.0 - u) / math.log(1.0 - p)))

    def poisson(self, lam):
        """
        Poisson distribution using inverse transform (Knuth's algorithm).
        
        The Poisson distribution models the number of events occurring in
        a fixed interval when events happen at a constant average rate.
        
        Mathematical Foundation:
        -----------------------
        PMF: P(X=k) = (λ^k · e^(-λ)) / k! for k = 0, 1, 2, ...
        Mean: λ
        Variance: λ
        
        Algorithm (Knuth):
        The inter-arrival times between Poisson events are exponential.
        Count how many exponential samples fit before exceeding 1.
        
        Let L = e^(-λ). Generate uniform U₁, U₂, ... and multiply them.
        Return k-1 where k is the first index with ∏Uᵢ < L.
        
        Note: For large λ (>30), this becomes slow. Use normal approximation
        or rejection methods instead.
        
        Args:
            lam: Rate parameter λ > 0 (average number of events)
            
        Returns:
            Number of events (non-negative integer)
        """
        if lam > 30:
            # For large λ, use normal approximation
            return max(0, int(round(self.normal(lam, math.sqrt(lam)))))
        
        L = math.exp(-lam)
        k = 0
        p = 1.0
        
        while p > L:
            k += 1
            p *= self.uniform()
        
        return k - 1

    def negative_binomial(self, r, p):
        """
        Negative binomial distribution (trials until r successes).
        
        The negative binomial distribution models the number of failures
        before achieving r successes in Bernoulli trials.
        
        PMF: P(X=k) = C(k+r-1, k) · p^r · (1-p)^k for k = 0, 1, 2, ...
        Mean: r(1-p)/p
        Variance: r(1-p)/p²
        
        Special Case: NegBinomial(1, p) = Geometric(p) - 1
        
        Args:
            r: Number of successes needed (positive integer)
            p: Probability of success, 0 < p ≤ 1
            
        Returns:
            Number of failures before r successes
        """
        failures = 0
        successes = 0
        while successes < r:
            if self.bernoulli(p):
                successes += 1
            else:
                failures += 1
        return failures

    # =========================================================================
    # UTILITY METHODS
    # =========================================================================

    def choice(self, sequence):
        """
        Uniform random choice from a sequence.
        
        Args:
            sequence: Non-empty sequence to choose from
            
        Returns:
            Randomly selected element
        """
        return self.random.choice(sequence)

    def shuffle(self, sequence):
        """
        Shuffle a sequence in place using Fisher-Yates algorithm.
        
        Args:
            sequence: Mutable sequence to shuffle
            
        Returns:
            None (modifies sequence in place)
        """
        self.random.shuffle(sequence)

    def sample(self, population, k):
        """
        Random sample without replacement.
        
        Args:
            population: Sequence to sample from
            k: Number of elements to sample
            
        Returns:
            List of k randomly selected elements
        """
        return self.random.sample(population, k)

    def weighted_choice(self, weights):
        """
        Choose an index based on weights (categorical distribution).
        
        This implements a discrete distribution where P(X=i) ∝ weights[i].
        
        Implementation uses the cumulative distribution function (CDF) method:
        1. Compute cumulative weights
        2. Generate uniform sample scaled to total weight
        3. Find first index where cumulative weight exceeds sample
        
        Args:
            weights: List of non-negative weights (need not sum to 1)
            
        Returns:
            Index i with probability proportional to weights[i]
        """
        cumulative = []
        total = 0.0
        for w in weights:
            total += w
            cumulative.append(total)
        
        u = self.uniform(0.0, total)
        
        for i, c in enumerate(cumulative):
            if u <= c:
                return i
        
        return len(weights) - 1  # Edge case for floating point

    def set_seed(self, seed):
        """
        Reset the random state with a new seed.
        
        Useful for reproducibility in testing and debugging.
        
        Args:
            seed: New seed value
        """
        self.random.seed(seed)
        self._spare_normal = None


# =============================================================================
# DEMONSTRATION
# =============================================================================

if __name__ == "__main__":
    rng = RNG(seed=42)
    
    print("=" * 60)
    print("RNG Demonstration - Information Processing Class")
    print("=" * 60)
    
    print("\n📊 Continuous Distributions:")
    print(f"  Uniform(0, 1):        {rng.uniform():.4f}")
    print(f"  Normal(0, 1):         {rng.normal():.4f}")
    print(f"  Normal (Box-Muller):  {rng.normal_box_muller():.4f}")
    print(f"  Normal (Marsaglia):   {rng.normal_marsaglia():.4f}")
    print(f"  Exponential(λ=1):     {rng.expo(1.0):.4f}")
    print(f"  Gamma(2, 1):          {rng.gamma(2, 1):.4f}")
    print(f"  Beta(2, 5):           {rng.beta(2, 5):.4f}")
    
    print("\n📊 Discrete Distributions:")
    print(f"  Bernoulli(0.7):       {rng.bernoulli(0.7)}")
    print(f"  Binomial(10, 0.5):    {rng.binomial(10, 0.5)}")
    print(f"  Geometric(0.3):       {rng.geometric(0.3)}")
    print(f"  Poisson(λ=4):         {rng.poisson(4)}")
    print(f"  NegBinomial(3, 0.5):  {rng.negative_binomial(3, 0.5)}")
    
    print("\n📊 Utility Methods:")
    print(f"  Uniform Int(1, 6):    {rng.uniform_int(1, 6)}")
    print(f"  Choice([a,b,c,d]):    {rng.choice(['a', 'b', 'c', 'd'])}")
    print(f"  Weighted([1,2,3]):    {rng.weighted_choice([1, 2, 3])}")
    
    print("\n" + "=" * 60)
