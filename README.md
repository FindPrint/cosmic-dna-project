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
