# tests/test_validator.py

import unittest
import numpy as np
from src.cosmic_dna_validator import CosmicDNAMetricsValidator

class TestCosmicDNAValidator(unittest.TestCase):
    def setUp(self):
        self.validator = CosmicDNAMetricsValidator()
        self.positions, self.velocities, self.box_size = self.validator.generate_realistic_catalog_with_velocities(
            n_galaxies=1000, box_size=500, bias=1.8, sigma_v=400, random_seed=42
        )
        self.r_bins = np.logspace(-1, 2, 25)

    def test_correlation_function(self):
        r, xi, _ = self.validator.compute_correlation_function(self.positions, self.r_bins, self.box_size, 'real')
        self.assertEqual(len(r), len(self.r_bins) - 1)
        self.assertTrue(np.all(xi >= -1.0))

    def test_redshift_distortions(self):
        redshift_pos = self.validator.apply_redshift_space_distortions(self.positions, self.velocities)
        self.assertEqual(redshift_pos.shape, self.positions.shape)

    def test_clustering(self):
        metrics = self.validator.compute_clustering_metrics(self.positions)
        self.assertTrue(metrics['n_clusters'] >= 0)

    def test_mcmc_optimization(self):
        best_params, samples = self.validator.optimize_coefficients_mcmc(
            self.positions, self.velocities, self.r_bins, self.box_size, n_walkers=8, n_steps=100, burn_in=20
        )
        self.assertEqual(len(best_params), 7)
        self.assertTrue(np.all(best_params > 0))

if __name__ == '__main__':
    unittest.main()
