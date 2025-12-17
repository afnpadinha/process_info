"""
Spin Engine Module for Roulette Simulation.

This module simulates the physics of a roulette wheel spin using
random variables from different probability distributions.

Educational Discussion:
----------------------
The roulette simulation demonstrates several key concepts in random
variable generation and transformation:

1. **Normal Distribution** (ω - angular velocity):
   Real spinning wheels have variability in initial velocity.
   Using N(μ, σ²) models this natural variation.

2. **Exponential Distribution** (t_spin - spin duration):
   The exponential distribution is memoryless, modeling the time
   until an event (wheel stopping) in a Poisson process.

3. **Composition of Random Variables**:
   The product ω·t gives the travel angle, which has a complex
   distribution (product of normal and exponential).

Statistical Note on the Original Implementation:
-----------------------------------------------
Adding a uniform phase U ~ Uniform(0, 360) to ANY angle and taking
mod 360 results in a uniform distribution, regardless of the input!

Proof: If θ is any random variable and U ~ Uniform(0, 360) is
independent, then (θ + U) mod 360 ~ Uniform(0, 360).

This means the physics simulation (omega * tspin) becomes irrelevant!
This is actually desirable for a FAIR roulette (each slot equally likely),
but defeats the purpose of the physics simulation for educational purposes.

This module provides TWO modes:
- "fair": Uses direct uniform sampling (each slot equally likely)
- "physics": Uses physics simulation without uniform phase (educational)

Author: Information Processing Class Project
"""

import math
from .rng import RNG
from . import config


class SpinEngine:
    """
    Roulette wheel spin simulator.
    
    Simulates the physics of a spinning roulette wheel using various
    probability distributions to model physical parameters.
    
    Attributes:
        rng: RNG instance for random number generation
        config: Configuration module with wheel parameters
        mode: "fair" for uniform slots, "physics" for physics simulation
    """
    
    def __init__(self, rng, config, mode="fair"):
        """
        Initialize the spin engine.
        
        Args:
            rng: RNG instance (seeded upstream for reproducibility)
            config: Configuration with parameters (OMEGA_MEAN, OMEGA_STD, etc.)
            mode: "fair" or "physics" (default: "fair")
        """
        self.rng = rng
        self.config = config
        self.mode = mode

    def simulate(self):
        """
        Simulate a roulette wheel spin and return the landing slot.
        
        For "fair" mode: Returns uniformly random slot.
        For "physics" mode: Uses physics simulation.
        
        Returns:
            Integer slot index (0 to NUM_SLOT - 1)
        """
        if self.mode == "fair":
            return self._simulate_fair()
        else:
            return self._simulate_physics()
    
    def _simulate_fair(self):
        """
        Fair simulation: each slot has equal probability.
        
        This directly samples from a discrete uniform distribution
        over slot indices. This is the standard for a fair roulette.
        
        Distribution: Discrete Uniform(0, NUM_SLOT - 1)
        P(slot = k) = 1/NUM_SLOT for all k
        
        Returns:
            Integer slot index
        """
        return self.rng.uniform_int(0, self.config.NUM_SLOT - 1)
    
    def _simulate_physics(self):
        """
        Physics-based simulation for educational purposes.
        
        Models the wheel spin using:
        - ω (omega): Angular velocity ~ Normal(μ, σ²)
          Represents variability in how hard the croupier spins
        
        - t: Spin duration ~ Exponential(λ)
          Models the time until the wheel stops (memoryless property)
        
        The landing angle is: θ = (ω · t) mod 360
        
        This creates a NON-uniform distribution over angles, which is
        educationally interesting but not a fair roulette!
        
        Mathematical Analysis:
        ---------------------
        When ω ~ N(μ, σ²) and t ~ Exp(λ), the product ω·t has a
        complex distribution. The resulting angle distribution depends
        on the parameters.
        
        For fairness analysis, you could:
        1. Run Monte Carlo simulation to check slot frequencies
        2. Apply chi-squared test to verify uniformity
        
        Returns:
            Integer slot index
        """
        # Random Continuous Variable 1: Angular velocity (degrees per second)
        # Normal distribution models natural variation in spin force
        omega = self.rng.normal(
            self.config.OMEGA_MEAN,
            self.config.OMEGA_STD
        )
        
        # Random Continuous Variable 2: Spin duration (seconds)
        # Exponential distribution models time until stopping
        tspin = self.rng.expo(self.config.TSPIN_LAMBDA)
        
        # Compute travel angle
        # This is a product of two random variables: ω·t
        travel_angle = omega * tspin
        
        # Take modulo 360 to get position on wheel
        landing_angle = travel_angle % 360
        
        # Handle negative angles (if omega was negative)
        if landing_angle < 0:
            landing_angle += 360
        
        # Map angle to slot index
        slot_width = 360 / self.config.NUM_SLOT
        slot = int(landing_angle / slot_width)
        
        # Clamp to valid range (edge case for angle = 360)
        slot = min(slot, self.config.NUM_SLOT - 1)
        
        return slot
    
    def simulate_with_details(self):
        """
        Simulate and return detailed information (for educational purposes).
        
        Returns:
            Dictionary with all random variables and intermediate values
        """
        if self.mode == "fair":
            slot = self._simulate_fair()
            return {
                'mode': 'fair',
                'slot': slot,
                'distribution': 'Discrete Uniform(0, NUM_SLOT-1)'
            }
        
        # Physics mode with full details
        omega = self.rng.normal(self.config.OMEGA_MEAN, self.config.OMEGA_STD)
        tspin = self.rng.expo(self.config.TSPIN_LAMBDA)
        travel_angle = omega * tspin
        landing_angle = travel_angle % 360
        if landing_angle < 0:
            landing_angle += 360
        
        slot_width = 360 / self.config.NUM_SLOT
        slot = int(landing_angle / slot_width)
        slot = min(slot, self.config.NUM_SLOT - 1)
        
        return {
            'mode': 'physics',
            'omega': omega,
            'omega_distribution': f'Normal(μ={self.config.OMEGA_MEAN}, σ={self.config.OMEGA_STD})',
            'tspin': tspin,
            'tspin_distribution': f'Exponential(λ={self.config.TSPIN_LAMBDA})',
            'travel_angle': travel_angle,
            'landing_angle': landing_angle,
            'slot': slot
        }
    
    def analyze_fairness(self, n_simulations=10000):
        """
        Analyze the fairness of the simulation by running Monte Carlo.
        
        For a fair roulette, each slot should appear with probability 1/NUM_SLOT.
        This method runs many simulations and checks the distribution.
        
        Args:
            n_simulations: Number of spins to simulate
            
        Returns:
            Dictionary with analysis results
        """
        from collections import Counter
        
        slots = [self.simulate() for _ in range(n_simulations)]
        counts = Counter(slots)
        
        expected = n_simulations / self.config.NUM_SLOT
        
        # Chi-squared statistic
        chi_sq = sum((counts.get(i, 0) - expected)**2 / expected 
                     for i in range(self.config.NUM_SLOT))
        
        # Degrees of freedom
        df = self.config.NUM_SLOT - 1
        
        # Find most and least common slots
        most_common = counts.most_common(3)
        least_common = counts.most_common()[-3:]
        
        return {
            'mode': self.mode,
            'n_simulations': n_simulations,
            'expected_per_slot': expected,
            'chi_squared': chi_sq,
            'degrees_of_freedom': df,
            'most_common_slots': most_common,
            'least_common_slots': least_common,
            'max_deviation': max(abs(counts.get(i, 0) - expected) for i in range(self.config.NUM_SLOT)),
            'is_likely_fair': chi_sq < 1.5 * df  # Rough heuristic
        }


def demo_spin_engine():
    """Demonstrate the spin engine with both modes."""
    print("=" * 60)
    print("SPIN ENGINE DEMONSTRATION")
    print("=" * 60)
    
    rng = RNG(seed=42)
    
    # Fair mode
    print("\n📊 FAIR MODE (Direct Uniform Sampling)")
    print("-" * 40)
    engine_fair = SpinEngine(rng, config, mode="fair")
    for i in range(5):
        result = engine_fair.simulate_with_details()
        print(f"  Spin {i+1}: Slot {result['slot']}")
    
    analysis = engine_fair.analyze_fairness(10000)
    print(f"\n  Fairness Analysis (10,000 spins):")
    print(f"    χ² statistic: {analysis['chi_squared']:.2f}")
    print(f"    Expected per slot: {analysis['expected_per_slot']:.1f}")
    print(f"    Max deviation: {analysis['max_deviation']:.1f}")
    print(f"    Likely fair: {'Yes ✅' if analysis['is_likely_fair'] else 'No ❌'}")
    
    # Physics mode
    print("\n📊 PHYSICS MODE (Educational Simulation)")
    print("-" * 40)
    rng.set_seed(42)
    engine_physics = SpinEngine(rng, config, mode="physics")
    for i in range(5):
        result = engine_physics.simulate_with_details()
        print(f"  Spin {i+1}:")
        print(f"    ω = {result['omega']:.2f} deg/s ~ {result['omega_distribution']}")
        print(f"    t = {result['tspin']:.2f} s ~ {result['tspin_distribution']}")
        print(f"    θ = {result['landing_angle']:.1f}° → Slot {result['slot']}")
    
    analysis = engine_physics.analyze_fairness(10000)
    print(f"\n  Fairness Analysis (10,000 spins):")
    print(f"    χ² statistic: {analysis['chi_squared']:.2f}")
    print(f"    Expected per slot: {analysis['expected_per_slot']:.1f}")
    print(f"    Max deviation: {analysis['max_deviation']:.1f}")
    print(f"    Likely fair: {'Yes ✅' if analysis['is_likely_fair'] else 'No ❌'}")
    print(f"    Most common: {analysis['most_common_slots']}")
    print(f"    Least common: {analysis['least_common_slots']}")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    demo_spin_engine()
