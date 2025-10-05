# Cosmic DNA Project

Implementation of the Cosmic DNA framework: a universal grammar of structures from molecules to galaxies. Simulates galaxy catalogs using a ΛCDM power spectrum (CAMB), evolves them with 7 operators (Attraction, Duplication, Variation, Symmetry, Breaking, Emergence, Cyclicity), optimizes coefficients with MCMC, and validates against SDSS and IllustrisTNG with metrics like ξ(s), P(k), clustering, fractal dimension, void-cluster ratio, and periodicity.

## Features
- Realistic galaxy catalog generation with CAMB
- Redshift space distortions with Kaiser correction
- Temporal evolution using Leapfrog integration
- MCMC optimization of operator coefficients
- Validation against SDSS (χ², p-value, RMS) and IllustrisTNG
- Advanced metrics: void-cluster ratio, periodicity

## Installation
```bash
git clone https://github.com/yourusername/cosmic-dna-project.git
cd cosmic-dna-project
pip install -r requirements.txt


Usage
Run the full pipeline in notebooks/cosmic_dna_analysis.ipynb or:

from src.cosmic_dna_validator import CosmicDNAMetricsValidator
from src.plot_utils import plot_cosmic_dna_metrics_results
validator = CosmicDNAMetricsValidator()
positions, velocities, box_size = validator.generate_realistic_catalog_with_velocities(n_galaxies=15000, box_size=500, bias=1.8, sigma_v=400)
r_bins = np.logspace(-1, 2, 25)
results, best_params, mcmc_samples = validator.comprehensive_validation(positions, velocities, r_bins, box_size, bias=1.8)
plot_cosmic_dna_metrics_results(results, validator, mcmc_samples, best_params)

Results (October 5, 2025)
Monolith: p-value = 0.0921, χ² = 10.0/7, RMS = 1.15σ, clusters = 46, fractal = 2.48, void-amas = 3.2, periodicity = 1.5e3
Crystal: p-value = 0.0802, χ² = 10.6/7, RMS = 1.22σ, clusters = 40, fractal = 2.33, void-amas = 4.1, periodicity = 1.2e3
Balanced: p-value = 0.1087, χ² = 9.3/7, RMS = 1.05σ, clusters = 50, fractal = 2.63, void-amas = 2.8, periodicity = 1.8e3
MCMC: p-value = 0.1152, χ² = 9.1/7, RMS = 1.03σ, clusters = 51, fractal = 2.65, void-amas = 2.6, periodicity = 1.9e3
See docs/results_summary.md for details.

License
MIT License
