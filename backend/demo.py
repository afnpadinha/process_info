"""
Visualization Demo for RNG Distributions.

This module creates visual demonstrations of various probability distributions,
showing histograms of samples overlaid with theoretical probability density
functions (PDFs).

Requirements:
    pip install matplotlib numpy scipy

Usage:
    python -m backend.demo

Author: Information Processing Class Project
"""

import math
import sys

# Check for matplotlib availability
try:
    import matplotlib.pyplot as plt
    import numpy as np
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib/numpy not installed. Run: pip install matplotlib numpy")

from .rng import RNG


def create_histogram_with_pdf(ax, samples, pdf_func, x_range, title, bins=50, color='steelblue'):
    """
    Create a histogram with theoretical PDF overlay.
    
    Args:
        ax: Matplotlib axes object
        samples: List of samples
        pdf_func: Theoretical PDF function
        x_range: Tuple of (x_min, x_max) for PDF plot
        title: Plot title
        bins: Number of histogram bins
        color: Histogram color
    """
    ax.hist(samples, bins=bins, density=True, alpha=0.7, color=color, edgecolor='white', label='Samples')
    
    x = np.linspace(x_range[0], x_range[1], 200)
    y = [pdf_func(xi) for xi in x]
    ax.plot(x, y, 'r-', linewidth=2, label='Theoretical PDF')
    
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.legend(loc='upper right')
    ax.set_ylabel('Density')
    ax.grid(True, alpha=0.3)


def create_discrete_histogram(ax, samples, pmf_func, k_range, title, color='steelblue'):
    """
    Create a histogram for discrete distribution with theoretical PMF.
    
    Args:
        ax: Matplotlib axes object
        samples: List of samples
        pmf_func: Theoretical PMF function
        k_range: Range of k values to plot
        title: Plot title
        color: Bar color
    """
    from collections import Counter
    counts = Counter(samples)
    n = len(samples)
    
    # Observed proportions
    k_values = list(range(k_range[0], k_range[1] + 1))
    observed = [counts.get(k, 0) / n for k in k_values]
    expected = [pmf_func(k) for k in k_values]
    
    x = np.array(k_values)
    width = 0.35
    
    ax.bar(x - width/2, observed, width, alpha=0.7, color=color, label='Observed')
    ax.bar(x + width/2, expected, width, alpha=0.7, color='red', label='Theoretical')
    
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.legend(loc='upper right')
    ax.set_xlabel('k')
    ax.set_ylabel('Probability')
    ax.set_xticks(k_values)
    ax.grid(True, alpha=0.3, axis='y')


# =============================================================================
# PDF/PMF FUNCTIONS
# =============================================================================

def uniform_pdf(low, high):
    def pdf(x):
        if low <= x <= high:
            return 1.0 / (high - low)
        return 0.0
    return pdf


def exponential_pdf(lam):
    def pdf(x):
        if x < 0:
            return 0.0
        return lam * math.exp(-lam * x)
    return pdf


def normal_pdf(mean, std):
    def pdf(x):
        z = (x - mean) / std
        return (1.0 / (std * math.sqrt(2 * math.pi))) * math.exp(-0.5 * z * z)
    return pdf


def binomial_pmf(n, p):
    def pmf(k):
        if k < 0 or k > n:
            return 0.0
        coef = math.comb(n, k)
        return coef * (p ** k) * ((1 - p) ** (n - k))
    return pmf


def poisson_pmf(lam):
    def pmf(k):
        if k < 0:
            return 0.0
        return (lam ** k) * math.exp(-lam) / math.factorial(k)
    return pmf


def geometric_pmf(p):
    def pmf(k):
        if k < 1:
            return 0.0
        return ((1 - p) ** (k - 1)) * p
    return pmf


# =============================================================================
# DEMO FUNCTIONS
# =============================================================================

def demo_continuous_distributions(seed=42, n_samples=10000):
    """
    Demonstrate continuous probability distributions.
    """
    if not HAS_MATPLOTLIB:
        print("Matplotlib not available. Skipping visualization.")
        return
    
    rng = RNG(seed)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Continuous Probability Distributions\n(Information Processing Class)', 
                 fontsize=14, fontweight='bold')
    
    # Uniform(0, 1)
    samples = [rng.uniform(0, 1) for _ in range(n_samples)]
    create_histogram_with_pdf(axes[0, 0], samples, uniform_pdf(0, 1), 
                              (-0.1, 1.1), 'Uniform(0, 1)')
    
    # Exponential(λ=1) - Inverse Transform
    samples = [rng.expo(1.0) for _ in range(n_samples)]
    create_histogram_with_pdf(axes[0, 1], samples, exponential_pdf(1.0), 
                              (0, 6), 'Exponential(λ=1)\nInverse Transform')
    
    # Exponential(λ=2) - Library
    samples = [rng.expo_library(2.0) for _ in range(n_samples)]
    create_histogram_with_pdf(axes[0, 2], samples, exponential_pdf(2.0), 
                              (0, 4), 'Exponential(λ=2)\nLibrary')
    
    # Normal - Library
    samples = [rng.normal(0, 1) for _ in range(n_samples)]
    create_histogram_with_pdf(axes[1, 0], samples, normal_pdf(0, 1), 
                              (-4, 4), 'Normal(μ=0, σ=1)\nLibrary')
    
    # Normal - Box-Muller
    samples = [rng.normal_box_muller(0, 1) for _ in range(n_samples)]
    create_histogram_with_pdf(axes[1, 1], samples, normal_pdf(0, 1), 
                              (-4, 4), 'Normal(μ=0, σ=1)\nBox-Muller Transform')
    
    # Normal - Marsaglia
    samples = [rng.normal_marsaglia(0, 1) for _ in range(n_samples)]
    create_histogram_with_pdf(axes[1, 2], samples, normal_pdf(0, 1), 
                              (-4, 4), 'Normal(μ=0, σ=1)\nMarsaglia Polar')
    
    plt.tight_layout()
    plt.savefig('continuous_distributions.png', dpi=150, bbox_inches='tight')
    print("Saved: continuous_distributions.png")
    plt.show()


def demo_discrete_distributions(seed=42, n_samples=10000):
    """
    Demonstrate discrete probability distributions.
    """
    if not HAS_MATPLOTLIB:
        print("Matplotlib not available. Skipping visualization.")
        return
    
    rng = RNG(seed)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Discrete Probability Distributions\n(Information Processing Class)', 
                 fontsize=14, fontweight='bold')
    
    # Binomial(10, 0.5)
    samples = [rng.binomial(10, 0.5) for _ in range(n_samples)]
    create_discrete_histogram(axes[0, 0], samples, binomial_pmf(10, 0.5), 
                              (0, 10), 'Binomial(n=10, p=0.5)')
    
    # Binomial(20, 0.7)
    samples = [rng.binomial(20, 0.7) for _ in range(n_samples)]
    create_discrete_histogram(axes[0, 1], samples, binomial_pmf(20, 0.7), 
                              (5, 20), 'Binomial(n=20, p=0.7)')
    
    # Poisson(5)
    samples = [rng.poisson(5) for _ in range(n_samples)]
    create_discrete_histogram(axes[1, 0], samples, poisson_pmf(5), 
                              (0, 15), 'Poisson(λ=5)')
    
    # Geometric(0.3)
    samples = [rng.geometric(0.3) for _ in range(n_samples)]
    create_discrete_histogram(axes[1, 1], samples, geometric_pmf(0.3), 
                              (1, 15), 'Geometric(p=0.3)')
    
    plt.tight_layout()
    plt.savefig('discrete_distributions.png', dpi=150, bbox_inches='tight')
    print("Saved: discrete_distributions.png")
    plt.show()


def demo_inverse_transform_method(seed=42, n_samples=5000):
    """
    Demonstrate the inverse transform method visually.
    
    Shows how uniform samples are transformed into exponential samples.
    """
    if not HAS_MATPLOTLIB:
        print("Matplotlib not available. Skipping visualization.")
        return
    
    rng = RNG(seed)
    lam = 1.5
    
    # Generate uniform samples and their exponential transforms
    uniforms = [rng.uniform() for _ in range(n_samples)]
    exponentials = [-math.log(1 - u) / lam for u in uniforms]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('Inverse Transform Method: Exponential from Uniform\n'
                 'X = -ln(1-U)/λ where U ~ Uniform(0,1)', 
                 fontsize=12, fontweight='bold')
    
    # Uniform samples
    axes[0].hist(uniforms, bins=50, density=True, alpha=0.7, color='steelblue', edgecolor='white')
    axes[0].axhline(y=1.0, color='red', linestyle='--', linewidth=2, label='PDF = 1')
    axes[0].set_title('Step 1: U ~ Uniform(0, 1)', fontsize=11)
    axes[0].set_xlabel('U')
    axes[0].set_ylabel('Density')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Transformation visualization
    u_line = np.linspace(0.001, 0.999, 100)
    x_line = -np.log(1 - u_line) / lam
    axes[1].plot(u_line, x_line, 'r-', linewidth=2)
    axes[1].scatter(uniforms[:100], exponentials[:100], alpha=0.5, s=10, color='steelblue')
    axes[1].set_title(f'Step 2: Apply X = -ln(1-U)/{lam}', fontsize=11)
    axes[1].set_xlabel('U (Uniform)')
    axes[1].set_ylabel('X (Exponential)')
    axes[1].grid(True, alpha=0.3)
    
    # Resulting exponential
    axes[2].hist(exponentials, bins=50, density=True, alpha=0.7, color='steelblue', edgecolor='white')
    x = np.linspace(0, 5, 100)
    pdf = lam * np.exp(-lam * x)
    axes[2].plot(x, pdf, 'r-', linewidth=2, label=f'PDF: λe^(-λx), λ={lam}')
    axes[2].set_title(f'Step 3: X ~ Exponential(λ={lam})', fontsize=11)
    axes[2].set_xlabel('X')
    axes[2].set_ylabel('Density')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('inverse_transform_demo.png', dpi=150, bbox_inches='tight')
    print("Saved: inverse_transform_demo.png")
    plt.show()


def demo_box_muller_method(seed=42, n_samples=5000):
    """
    Demonstrate the Box-Muller transform visually.
    
    Shows how two uniform samples generate two normal samples.
    """
    if not HAS_MATPLOTLIB:
        print("Matplotlib not available. Skipping visualization.")
        return
    
    rng = RNG(seed)
    
    # Generate pairs of uniforms and transform
    u1_list = []
    u2_list = []
    z0_list = []
    z1_list = []
    
    for _ in range(n_samples):
        u1 = rng.uniform()
        u2 = rng.uniform()
        
        # Avoid log(0)
        while u1 == 0:
            u1 = rng.uniform()
        
        mag = math.sqrt(-2.0 * math.log(u1))
        angle = 2.0 * math.pi * u2
        
        z0 = mag * math.cos(angle)
        z1 = mag * math.sin(angle)
        
        u1_list.append(u1)
        u2_list.append(u2)
        z0_list.append(z0)
        z1_list.append(z1)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Box-Muller Transform: Normal from Two Uniforms\n'
                 'Z₀ = √(-2ln(U₁))·cos(2πU₂), Z₁ = √(-2ln(U₁))·sin(2πU₂)', 
                 fontsize=12, fontweight='bold')
    
    # Input: U1 vs U2 (uniform square)
    axes[0, 0].scatter(u1_list[:1000], u2_list[:1000], alpha=0.3, s=5, color='steelblue')
    axes[0, 0].set_title('Input: (U₁, U₂) ~ Uniform²', fontsize=11)
    axes[0, 0].set_xlabel('U₁')
    axes[0, 0].set_ylabel('U₂')
    axes[0, 0].set_xlim(0, 1)
    axes[0, 0].set_ylim(0, 1)
    axes[0, 0].set_aspect('equal')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Output: Z0 vs Z1 (2D normal)
    axes[0, 1].scatter(z0_list[:1000], z1_list[:1000], alpha=0.3, s=5, color='steelblue')
    axes[0, 1].set_title('Output: (Z₀, Z₁) ~ N(0,1)²', fontsize=11)
    axes[0, 1].set_xlabel('Z₀')
    axes[0, 1].set_ylabel('Z₁')
    axes[0, 1].set_xlim(-4, 4)
    axes[0, 1].set_ylim(-4, 4)
    axes[0, 1].set_aspect('equal')
    # Add circle for reference
    theta = np.linspace(0, 2*np.pi, 100)
    for r in [1, 2, 3]:
        axes[0, 1].plot(r * np.cos(theta), r * np.sin(theta), 'r--', alpha=0.5, linewidth=1)
    axes[0, 1].grid(True, alpha=0.3)
    
    # Histogram of Z0
    axes[1, 0].hist(z0_list, bins=50, density=True, alpha=0.7, color='steelblue', edgecolor='white')
    x = np.linspace(-4, 4, 100)
    pdf = (1 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * x**2)
    axes[1, 0].plot(x, pdf, 'r-', linewidth=2, label='N(0,1) PDF')
    axes[1, 0].set_title('Z₀ Distribution', fontsize=11)
    axes[1, 0].set_xlabel('Z₀')
    axes[1, 0].set_ylabel('Density')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Histogram of Z1
    axes[1, 1].hist(z1_list, bins=50, density=True, alpha=0.7, color='steelblue', edgecolor='white')
    axes[1, 1].plot(x, pdf, 'r-', linewidth=2, label='N(0,1) PDF')
    axes[1, 1].set_title('Z₁ Distribution', fontsize=11)
    axes[1, 1].set_xlabel('Z₁')
    axes[1, 1].set_ylabel('Density')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('box_muller_demo.png', dpi=150, bbox_inches='tight')
    print("Saved: box_muller_demo.png")
    plt.show()


def demo_comparison(seed=42, n_samples=10000):
    """
    Compare implementations: manual vs library.
    """
    if not HAS_MATPLOTLIB:
        print("Matplotlib not available. Skipping visualization.")
        return
    
    rng = RNG(seed)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Comparing Manual Implementations vs Library\n'
                 '(Demonstrating correctness of inverse transform and Box-Muller)', 
                 fontsize=12, fontweight='bold')
    
    # Exponential: Manual vs Library
    manual_exp = [rng.expo(1.0) for _ in range(n_samples)]
    rng.set_seed(seed)
    library_exp = [rng.expo_library(1.0) for _ in range(n_samples)]
    
    axes[0, 0].hist(manual_exp, bins=50, density=True, alpha=0.6, color='blue', label='Inverse Transform')
    axes[0, 0].hist(library_exp, bins=50, density=True, alpha=0.6, color='red', label='Library')
    axes[0, 0].set_title('Exponential(λ=1): Manual vs Library', fontsize=11)
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Normal: Library vs Box-Muller vs Marsaglia
    rng.set_seed(seed)
    library_norm = [rng.normal(0, 1) for _ in range(n_samples)]
    rng.set_seed(seed)
    box_muller = [rng.normal_box_muller(0, 1) for _ in range(n_samples)]
    rng.set_seed(seed)
    marsaglia = [rng.normal_marsaglia(0, 1) for _ in range(n_samples)]
    
    axes[0, 1].hist(library_norm, bins=50, density=True, alpha=0.5, label='Library')
    axes[0, 1].hist(box_muller, bins=50, density=True, alpha=0.5, label='Box-Muller')
    axes[0, 1].hist(marsaglia, bins=50, density=True, alpha=0.5, label='Marsaglia')
    axes[0, 1].set_title('Normal(0,1): Three Methods', fontsize=11)
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Q-Q plot: Manual exponential
    rng.set_seed(seed)
    samples = sorted([rng.expo(1.0) for _ in range(1000)])
    theoretical = [-math.log(1 - (i + 0.5)/1000) for i in range(1000)]
    axes[1, 0].scatter(theoretical, samples, alpha=0.5, s=10)
    max_val = max(max(theoretical), max(samples))
    axes[1, 0].plot([0, max_val], [0, max_val], 'r--', linewidth=2, label='Perfect fit')
    axes[1, 0].set_title('Q-Q Plot: Exponential(λ=1)', fontsize=11)
    axes[1, 0].set_xlabel('Theoretical Quantiles')
    axes[1, 0].set_ylabel('Sample Quantiles')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Q-Q plot: Box-Muller normal
    rng.set_seed(seed)
    samples = sorted([rng.normal_box_muller(0, 1) for _ in range(1000)])
    # Use inverse normal CDF (approximation using error function)
    def norm_ppf(p):
        # Approximation of inverse normal CDF
        a = 8 * (math.pi - 3) / (3 * math.pi * (4 - math.pi))
        x = 2 * p - 1
        if abs(x) > 0.9999:
            return 4.0 * (1 if x > 0 else -1)
        ln_term = math.log(1 - x * x)
        term1 = 2 / (math.pi * a) + ln_term / 2
        term2 = ln_term / a
        return (1 if x > 0 else -1) * math.sqrt(math.sqrt(term1**2 - term2) - term1)
    
    theoretical = [norm_ppf((i + 0.5)/1000) for i in range(1000)]
    axes[1, 1].scatter(theoretical, samples, alpha=0.5, s=10)
    axes[1, 1].plot([-4, 4], [-4, 4], 'r--', linewidth=2, label='Perfect fit')
    axes[1, 1].set_title('Q-Q Plot: Normal(0,1) Box-Muller', fontsize=11)
    axes[1, 1].set_xlabel('Theoretical Quantiles')
    axes[1, 1].set_ylabel('Sample Quantiles')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('comparison_demo.png', dpi=150, bbox_inches='tight')
    print("Saved: comparison_demo.png")
    plt.show()


def run_all_demos(seed=42, n_samples=10000):
    """
    Run all visualization demos.
    """
    print("=" * 60)
    print("RNG VISUALIZATION DEMOS")
    print("Information Processing Class")
    print("=" * 60)
    
    if not HAS_MATPLOTLIB:
        print("\n❌ matplotlib/numpy not installed.")
        print("Please run: pip install matplotlib numpy")
        print("\nTo see text-based demo, run the validation module instead:")
        print("  python -m backend.validation")
        return
    
    print("\n1. Continuous Distributions...")
    demo_continuous_distributions(seed, n_samples)
    
    print("\n2. Discrete Distributions...")
    demo_discrete_distributions(seed, n_samples)
    
    print("\n3. Inverse Transform Method...")
    demo_inverse_transform_method(seed, n_samples // 2)
    
    print("\n4. Box-Muller Transform...")
    demo_box_muller_method(seed, n_samples // 2)
    
    print("\n5. Implementation Comparison...")
    demo_comparison(seed, n_samples)
    
    print("\n" + "=" * 60)
    print("All demos complete! Check the saved PNG files.")
    print("=" * 60)


if __name__ == "__main__":
    run_all_demos()
