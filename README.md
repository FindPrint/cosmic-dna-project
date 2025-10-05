# Cosmic DNA Project

A Python implementation to test the "Cosmic DNA" hypothesis, simulating galaxy structure formation with 7 operators (Attraction, Duplication, Variation, Symmetry, Breaking, Emergence, Cyclicity) and validating against SDSS observations.

## Overview
This project generates a realistic galaxy catalog (15,000 galaxies, 500 Mpc/h box) using a \(\Lambda\)CDM power spectrum (CAMB), evolves it with temporal dynamics, optimizes coefficients via MCMC, and validates against SDSS with metrics like \(\xi(s)\), \(P(k)\), clustering, fractal dimension, void-cluster ratio, and periodicity.

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/cosmic-dna-project.git
   cd cosmic-dna-project

   Install dependencies:
   pip install -r requirements.txt

   Run the analysis notebook:
   jupyter notebook notebooks/cosmic_dna_analysis.ipynb

   Requirements
Python 3.8+
numpy, matplotlib, scipy, astropy, scikit-learn, camb, emcee
See requirements.txt for details


Usage
from src.cosmic_dna_validator import CosmicDNAMetricsValidator
validator = CosmicDNAMetricsValidator()
positions, velocities, box_size = validator.generate_realistic_catalog_with_velocities(n_galaxies=15000, box_size=500)
results = validator.comprehensive_validation(positions, velocities, r_bins=np.logspace(-1, 2, 25), box_size=500, bias=1.8)

Results
MCMC: \(\chi^2 = 9.1\), p-value = 0.1152, RMS = 1.03σ, 51 clusters, fractal = 2.65, void-cluster = 2.6, periodicity = 1.9e3
See docs/results_summary.md for full results.
License
MIT License
