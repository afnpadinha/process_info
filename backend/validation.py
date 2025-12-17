"""
Statistical Validation Module for RNG.

This module provides tools to validate that our random number generators
produce correctly distributed samples using statistical hypothesis testing.

Key Concepts:
- Kolmogorov-Smirnov Test: Compares empirical CDF to theoretical CDF
- Chi-Squared Test: Compares observed frequencies to expected frequencies
- Visual Validation: Histograms with theoretical overlay

A p-value > 0.05 typically indicates the samples match the expected distribution.

Author: Information Processing Class Project
"""

import math
from collections import Counter
from .rng import RNG


def kolmogorov_smirnov_statistic(samples, cdf_func):
    """
    Compute the Kolmogorov-Smirnov test statistic.
    
    The KS test measures the maximum distance between the empirical
    cumulative distribution function (ECDF) and the theoretical CDF.
    
    D_n = sup|F_n(x) - F(x)|
    
    where F_n is the empirical CDF and F is the theoretical CDF.
    
    Args:
        samples: List of samples to test
        cdf_func: Theoretical CDF function F(x) -> probability
        
    Returns:
        Tuple of (D statistic, approximate p-value)
    """
    n = len(samples)
    sorted_samples = sorted(samples)
    
    d_max = 0.0
    
    for i, x in enumerate(sorted_samples):
        # ECDF at x (proportion of samples ≤ x)
        ecdf = (i + 1) / n
        # ECDF just before x
        ecdf_minus = i / n
        # Theoretical CDF
        tcdf = cdf_func(x)
        
        # Max difference (both at x and just before x)
        d1 = abs(ecdf - tcdf)
        d2 = abs(ecdf_minus - tcdf)
        d_max = max(d_max, d1, d2)
    
    # Approximate p-value using asymptotic distribution
    # For large n: P(D > d) ≈ 2 * sum_{k=1}^∞ (-1)^(k-1) * exp(-2k²n·d²)
    # We use a simple approximation
    sqrt_n = math.sqrt(n)
    lambda_ks = (sqrt_n + 0.12 + 0.11 / sqrt_n) * d_max
    
    # Compute p-value using first few terms of the series
    p_value = 0.0
    for k in range(1, 100):
        term = 2 * ((-1) ** (k - 1)) * math.exp(-2 * k * k * lambda_ks * lambda_ks)
        p_value += term
        if abs(term) < 1e-10:
            break
    
    p_value = max(0.0, min(1.0, p_value))
    
    return d_max, p_value


def chi_squared_test(observed, expected):
    """
    Perform chi-squared goodness-of-fit test.
    
    Tests whether observed frequencies differ significantly from expected.
    
    χ² = Σ (O_i - E_i)² / E_i
    
    Args:
        observed: List of observed counts
        expected: List of expected counts
        
    Returns:
        Tuple of (chi-squared statistic, degrees of freedom, approximate p-value)
    """
    assert len(observed) == len(expected), "Observed and expected must have same length"
    
    chi_sq = 0.0
    for o, e in zip(observed, expected):
        if e > 0:
            chi_sq += (o - e) ** 2 / e
    
    df = len(observed) - 1  # degrees of freedom
    
    # Approximate p-value using Wilson-Hilferty transformation
    # For χ² with df degrees of freedom
    if df <= 0:
        return chi_sq, df, 0.0
    
    # Normal approximation for p-value
    z = (chi_sq / df) ** (1/3) - (1 - 2 / (9 * df))
    z = z / math.sqrt(2 / (9 * df))
    
    # Standard normal CDF approximation
    p_value = 0.5 * (1 + math.erf(-z / math.sqrt(2)))
    
    return chi_sq, df, p_value


# =============================================================================
# CDF FUNCTIONS FOR COMMON DISTRIBUTIONS
# =============================================================================

def uniform_cdf(low, high):
    """CDF for Uniform(low, high) distribution."""
    def cdf(x):
        if x < low:
            return 0.0
        elif x > high:
            return 1.0
        else:
            return (x - low) / (high - low)
    return cdf


def exponential_cdf(lam):
    """CDF for Exponential(λ) distribution."""
    def cdf(x):
        if x < 0:
            return 0.0
        return 1.0 - math.exp(-lam * x)
    return cdf


def normal_cdf(mean, std):
    """CDF for Normal(μ, σ²) distribution."""
    def cdf(x):
        z = (x - mean) / std
        return 0.5 * (1 + math.erf(z / math.sqrt(2)))
    return cdf


# =============================================================================
# VALIDATION FUNCTIONS
# =============================================================================

def validate_uniform(rng, n_samples=10000, low=0.0, high=1.0):
    """
    Validate uniform distribution using KS test.
    
    Args:
        rng: RNG instance to test
        n_samples: Number of samples to generate
        low, high: Distribution parameters
        
    Returns:
        Dict with test results
    """
    samples = [rng.uniform(low, high) for _ in range(n_samples)]
    d_stat, p_value = kolmogorov_smirnov_statistic(samples, uniform_cdf(low, high))
    
    return {
        'distribution': f'Uniform({low}, {high})',
        'n_samples': n_samples,
        'ks_statistic': d_stat,
        'p_value': p_value,
        'passed': p_value > 0.05,
        'sample_mean': sum(samples) / n_samples,
        'expected_mean': (low + high) / 2,
        'sample_std': math.sqrt(sum((x - sum(samples)/n_samples)**2 for x in samples) / n_samples),
        'expected_std': (high - low) / math.sqrt(12)
    }


def validate_exponential(rng, n_samples=10000, lam=1.0, method='manual'):
    """
    Validate exponential distribution using KS test.
    
    Args:
        rng: RNG instance to test
        n_samples: Number of samples to generate
        lam: Rate parameter
        method: 'manual' for inverse transform or 'library' for built-in
        
    Returns:
        Dict with test results
    """
    if method == 'manual':
        samples = [rng.expo(lam) for _ in range(n_samples)]
    else:
        samples = [rng.expo_library(lam) for _ in range(n_samples)]
    
    d_stat, p_value = kolmogorov_smirnov_statistic(samples, exponential_cdf(lam))
    
    return {
        'distribution': f'Exponential(λ={lam}) [{method}]',
        'n_samples': n_samples,
        'ks_statistic': d_stat,
        'p_value': p_value,
        'passed': p_value > 0.05,
        'sample_mean': sum(samples) / n_samples,
        'expected_mean': 1 / lam,
        'sample_std': math.sqrt(sum((x - sum(samples)/n_samples)**2 for x in samples) / n_samples),
        'expected_std': 1 / lam
    }


def validate_normal(rng, n_samples=10000, mean=0.0, std=1.0, method='library'):
    """
    Validate normal distribution using KS test.
    
    Args:
        rng: RNG instance to test
        n_samples: Number of samples to generate
        mean, std: Distribution parameters
        method: 'library', 'box_muller', or 'marsaglia'
        
    Returns:
        Dict with test results
    """
    if method == 'library':
        samples = [rng.normal(mean, std) for _ in range(n_samples)]
    elif method == 'box_muller':
        samples = [rng.normal_box_muller(mean, std) for _ in range(n_samples)]
    else:
        samples = [rng.normal_marsaglia(mean, std) for _ in range(n_samples)]
    
    d_stat, p_value = kolmogorov_smirnov_statistic(samples, normal_cdf(mean, std))
    
    sample_mean = sum(samples) / n_samples
    return {
        'distribution': f'Normal(μ={mean}, σ={std}) [{method}]',
        'n_samples': n_samples,
        'ks_statistic': d_stat,
        'p_value': p_value,
        'passed': p_value > 0.05,
        'sample_mean': sample_mean,
        'expected_mean': mean,
        'sample_std': math.sqrt(sum((x - sample_mean)**2 for x in samples) / n_samples),
        'expected_std': std
    }


def validate_bernoulli(rng, n_samples=10000, p=0.5):
    """
    Validate Bernoulli distribution using chi-squared test.
    
    Args:
        rng: RNG instance to test
        n_samples: Number of samples
        p: Success probability
        
    Returns:
        Dict with test results
    """
    samples = [rng.bernoulli(p) for _ in range(n_samples)]
    
    counts = Counter(samples)
    observed = [counts.get(0, 0), counts.get(1, 0)]
    expected = [n_samples * (1 - p), n_samples * p]
    
    chi_sq, df, p_value = chi_squared_test(observed, expected)
    
    return {
        'distribution': f'Bernoulli(p={p})',
        'n_samples': n_samples,
        'chi_squared': chi_sq,
        'degrees_of_freedom': df,
        'p_value': p_value,
        'passed': p_value > 0.05,
        'observed_counts': dict(counts),
        'expected_counts': {0: expected[0], 1: expected[1]},
        'sample_mean': sum(samples) / n_samples,
        'expected_mean': p
    }


def validate_binomial(rng, n_samples=10000, n=10, p=0.5):
    """
    Validate Binomial distribution using chi-squared test.
    
    Args:
        rng: RNG instance to test
        n_samples: Number of samples
        n: Number of trials
        p: Success probability
        
    Returns:
        Dict with test results
    """
    samples = [rng.binomial(n, p) for _ in range(n_samples)]
    
    counts = Counter(samples)
    
    # Expected probabilities
    observed = []
    expected = []
    for k in range(n + 1):
        obs = counts.get(k, 0)
        # Binomial coefficient and probability
        coef = math.comb(n, k)
        exp_prob = coef * (p ** k) * ((1 - p) ** (n - k))
        exp_count = n_samples * exp_prob
        
        observed.append(obs)
        expected.append(exp_count)
    
    chi_sq, df, p_value = chi_squared_test(observed, expected)
    
    return {
        'distribution': f'Binomial(n={n}, p={p})',
        'n_samples': n_samples,
        'chi_squared': chi_sq,
        'degrees_of_freedom': df,
        'p_value': p_value,
        'passed': p_value > 0.05,
        'sample_mean': sum(samples) / n_samples,
        'expected_mean': n * p,
        'sample_var': sum((x - sum(samples)/n_samples)**2 for x in samples) / n_samples,
        'expected_var': n * p * (1 - p)
    }


def validate_poisson(rng, n_samples=10000, lam=5.0):
    """
    Validate Poisson distribution using chi-squared test.
    
    Args:
        rng: RNG instance to test
        n_samples: Number of samples
        lam: Rate parameter
        
    Returns:
        Dict with test results
    """
    samples = [rng.poisson(lam) for _ in range(n_samples)]
    
    counts = Counter(samples)
    max_k = max(samples) + 1
    
    # Group expected counts that are too small
    observed = []
    expected = []
    
    obs_tail = 0
    exp_tail = 0
    
    for k in range(max_k + 5):
        obs = counts.get(k, 0)
        # Poisson probability
        exp_prob = (lam ** k) * math.exp(-lam) / math.factorial(k)
        exp_count = n_samples * exp_prob
        
        if exp_count >= 5:
            observed.append(obs + obs_tail)
            expected.append(exp_count + exp_tail)
            obs_tail = 0
            exp_tail = 0
        else:
            obs_tail += obs
            exp_tail += exp_count
    
    # Add remaining tail
    if obs_tail > 0 or exp_tail > 0:
        if observed:
            observed[-1] += obs_tail
            expected[-1] += exp_tail
        else:
            observed.append(obs_tail)
            expected.append(exp_tail)
    
    chi_sq, df, p_value = chi_squared_test(observed, expected)
    
    return {
        'distribution': f'Poisson(λ={lam})',
        'n_samples': n_samples,
        'chi_squared': chi_sq,
        'degrees_of_freedom': df,
        'p_value': p_value,
        'passed': p_value > 0.05,
        'sample_mean': sum(samples) / n_samples,
        'expected_mean': lam,
        'sample_var': sum((x - sum(samples)/n_samples)**2 for x in samples) / n_samples,
        'expected_var': lam
    }


def validate_geometric(rng, n_samples=10000, p=0.3):
    """
    Validate Geometric distribution using chi-squared test.
    
    Args:
        rng: RNG instance to test
        n_samples: Number of samples
        p: Success probability
        
    Returns:
        Dict with test results
    """
    samples = [rng.geometric(p) for _ in range(n_samples)]
    
    counts = Counter(samples)
    max_k = max(samples)
    
    # Group expected counts that are too small
    observed = []
    expected = []
    
    obs_tail = 0
    exp_tail = 0
    
    for k in range(1, max_k + 5):
        obs = counts.get(k, 0)
        # Geometric probability
        exp_prob = ((1 - p) ** (k - 1)) * p
        exp_count = n_samples * exp_prob
        
        if exp_count >= 5:
            observed.append(obs + obs_tail)
            expected.append(exp_count + exp_tail)
            obs_tail = 0
            exp_tail = 0
        else:
            obs_tail += obs
            exp_tail += exp_count
    
    # Add remaining tail
    if obs_tail > 0 or exp_tail > 0:
        if observed:
            observed[-1] += obs_tail
            expected[-1] += exp_tail
        else:
            observed.append(obs_tail)
            expected.append(exp_tail)
    
    chi_sq, df, p_value = chi_squared_test(observed, expected)
    
    sample_mean = sum(samples) / n_samples
    return {
        'distribution': f'Geometric(p={p})',
        'n_samples': n_samples,
        'chi_squared': chi_sq,
        'degrees_of_freedom': df,
        'p_value': p_value,
        'passed': p_value > 0.05,
        'sample_mean': sample_mean,
        'expected_mean': 1 / p,
        'sample_var': sum((x - sample_mean)**2 for x in samples) / n_samples,
        'expected_var': (1 - p) / (p ** 2)
    }


def run_all_validations(seed=42, n_samples=10000):
    """
    Run all validation tests and print results.
    
    Args:
        seed: Random seed for reproducibility
        n_samples: Number of samples per test
        
    Returns:
        List of all test results
    """
    rng = RNG(seed)
    results = []
    
    print("=" * 70)
    print("STATISTICAL VALIDATION OF RNG DISTRIBUTIONS")
    print(f"Samples per test: {n_samples}")
    print("=" * 70)
    
    # Continuous distributions
    print("\n📊 CONTINUOUS DISTRIBUTIONS (Kolmogorov-Smirnov Test)")
    print("-" * 70)
    
    tests = [
        validate_uniform(rng, n_samples),
        validate_exponential(rng, n_samples, lam=1.0, method='manual'),
        validate_exponential(rng, n_samples, lam=2.0, method='library'),
        validate_normal(rng, n_samples, method='library'),
        validate_normal(rng, n_samples, method='box_muller'),
        validate_normal(rng, n_samples, method='marsaglia'),
    ]
    
    for result in tests:
        status = "✅ PASS" if result['passed'] else "❌ FAIL"
        print(f"{result['distribution']}")
        print(f"  KS Statistic: {result['ks_statistic']:.4f}, p-value: {result['p_value']:.4f} {status}")
        print(f"  Sample Mean: {result['sample_mean']:.4f} (expected: {result['expected_mean']:.4f})")
        print(f"  Sample Std:  {result['sample_std']:.4f} (expected: {result['expected_std']:.4f})")
        results.append(result)
    
    # Discrete distributions
    print("\n📊 DISCRETE DISTRIBUTIONS (Chi-Squared Test)")
    print("-" * 70)
    
    tests = [
        validate_bernoulli(rng, n_samples, p=0.7),
        validate_binomial(rng, n_samples, n=10, p=0.5),
        validate_poisson(rng, n_samples, lam=5.0),
        validate_geometric(rng, n_samples, p=0.3),
    ]
    
    for result in tests:
        status = "✅ PASS" if result['passed'] else "❌ FAIL"
        print(f"{result['distribution']}")
        print(f"  χ² Statistic: {result['chi_squared']:.4f}, df: {result['degrees_of_freedom']}, p-value: {result['p_value']:.4f} {status}")
        print(f"  Sample Mean: {result['sample_mean']:.4f} (expected: {result['expected_mean']:.4f})")
        results.append(result)
    
    # Summary
    print("\n" + "=" * 70)
    passed = sum(1 for r in results if r['passed'])
    total = len(results)
    print(f"SUMMARY: {passed}/{total} tests passed")
    print("=" * 70)
    
    return results


if __name__ == "__main__":
    run_all_validations()
