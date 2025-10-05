# src/plot_utils.py

import numpy as np
import matplotlib.pyplot as plt
import camb

def plot_cosmic_dna_metrics_results(results, validator, mcmc_samples, best_params):
    """Visualisation complète avec métriques supplémentaires"""
    fig, axes = plt.subplots(4, 2, figsize=(18, 24))
    
    colors = {'Monolith': 'blue', 'Crystal': 'green', 'Balanced': 'red', 'MCMC': 'purple', 'IllustrisTNG': 'black'}
    
    # 1. Fonction de corrélation
    ax = axes[0, 0]
    for regime_name, data in results.items():
        if regime_name == 'illustris':
            continue
        redshift = data['redshift_space']
        ax.loglog(redshift['r'], redshift['xi'], color=colors[regime_name], 
                  linewidth=2, label=f'{regime_name} ξ(s)')
    ax.loglog(results['illustris']['r'], results['illustris']['xi'], color=colors['IllustrisTNG'], 
              linewidth=2, label='IllustrisTNG ξ(r)')
    ax.loglog(validator.sdss_s, validator.sdss_xi_s, 'ko-', linewidth=2, 
              markersize=6, label='SDSS ξ(s)')
    ax.set_xlabel('Distance [Mpc/h]', fontsize=12)
    ax.set_ylabel('ξ(s)', fontsize=12)
    ax.set_title('Fonction de Corrélation (Redshift)', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Spectre de puissance avec CAMB
    ax = axes[0, 1]
    pars = camb.CAMBparams()
    pars.set_cosmology(H0=validator.H0, ombh2=0.0224, omch2=0.120)
    pars.InitPower.set_params(ns=0.965)
    pars.set_matter_power(redshifts=[0.0], kmax=2.0)
    results_camb = camb.get_results(pars)
    k_camb, _, Pk_camb = results_camb.get_matter_power_spectrum(minkh=1e-4, maxkh=1.0, npoints=200)
    ax.loglog(k_camb, Pk_camb[0], 'k--', label='CAMB P(k)', alpha=0.7)
    
    for regime_name, data in results.items():
        if regime_name == 'illustris':
            continue
        ps = data['power_spectrum']
        ax.loglog(ps['k'], ps['Pk'], color=colors[regime_name], 
                  linewidth=2, label=f'{regime_name} P(k)')
    ax.set_xlabel('k [h/Mpc]', fontsize=12)
    ax.set_ylabel('P(k)', fontsize=12)
    ax.set_title('Spectre de Puissance (CAMB)', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Distribution spatiale
    ax = axes[1, 0]
    for regime_name, data in results.items():
        if regime_name == 'illustris':
            continue
        positions = data['positions']
        ax.scatter(positions[:1000, 0], positions[:1000, 1], s=0.3, 
                   alpha=0.6, c=colors[regime_name], label=regime_name)
    ax.set_xlabel('X [Mpc/h]', fontsize=12)
    ax.set_ylabel('Y [Mpc/h]', fontsize=12)
    ax.set_title('Distribution Spatiale (XY)', fontsize=14)
    ax.set_aspect('equal')
    ax.legend()
    
    # 4. Résidus
    ax = axes[1, 1]
    for regime_name, data in results.items():
        if regime_name == 'illustris':
            continue
        residuals = data['redshift_space']['validation']['residuals']
        ax.plot(validator.sdss_s, residuals, color=colors[regime_name], 
                label=f'{regime_name}', alpha=0.7)
    ax.axhline(0, color='k', linestyle='--', alpha=0.5)
    ax.axhline(2, color='r', linestyle=':', alpha=0.5, label='±2σ')
    ax.axhline(-2, color='r', linestyle=':', alpha=0.5)
    ax.set_xlabel('Distance [Mpc/h]', fontsize=12)
    ax.set_ylabel('Résidus (σ)', fontsize=12)
    ax.set_title('Résidus Normalisés vs SDSS', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 5. Clustering
    ax = axes[2, 0]
    for regime_name, data in results.items():
        if regime_name == 'illustris':
            continue
        sizes = data['clustering']['sizes']
        ax.hist(sizes, bins=20, alpha=0.5, color=colors[regime_name], 
                label=f'{regime_name}: {data["clustering"]["n_clusters"]} clusters')
    ax.set_xlabel('Taille des clusters', fontsize=12)
    ax.set_ylabel('Nombre', fontsize=12)
    ax.set_title('Distribution des tailles de clusters', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 6. Distributions MCMC
    ax = axes[2, 1]
    labels = ['α', 'β', 'γ', 'δ', 'ν', 'b', 'c']
    for i, label in enumerate(labels):
        ax.hist(mcmc_samples[:, i], bins=30, alpha=0.5, label=f'{label} (med={best_params[i]:.2e})')
    ax.set_xlabel('Valeur des paramètres', fontsize=12)
    ax.set_ylabel('Nombre', fontsize=12)
    ax.set_title('Distributions MCMC des paramètres', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 7. Métriques
    ax = axes[3, 0]
    stats_text = "RAPPORT COSMIC DNA:\n\n"
    for regime_name, data in results.items():
        if regime_name == 'illustris':
            continue
        stats_text += f"{regime_name}:\n"
        stats_text += f"  • Compatible SDSS (χ²): {data['redshift_space']['validation']['compatible']}\n"
        stats_text += f"  • p-value: {data['redshift_space']['validation']['p_value']:.4f}\n"
        stats_text += f"  • χ²/dof: {data['redshift_space']['validation']['chi2']:.1f}/{data['redshift_space']['validation']['dof']}\n"
        stats_text += f"  • RMS résidus: {data['redshift_space']['validation']['rms_residual']:.2f}σ\n"
        stats_text += f"  • Clusters: {data['clustering']['n_clusters']}\n"
        stats_text += f"  • Dimension fractale: {data['fractal_dimension']:.2f}\n"
        stats_text += f"  • Rapport vide-amas: {data['void_cluster_ratio']:.2f}\n"
        stats_text += f"  • Périodicité (amp max): {data['periodicity']:.2e}\n\n"
    stats_text += "IllustrisTNG (TNG100-1 z=0):\n"
    stats_text += f"  • Clusters: {results['illustris']['n_clusters']}\n"
    stats_text += f"  • Dimension fractale: {results['illustris']['fractal_dimension']:.2f}\n"
    stats_text += f"  • xi(r) sample: {results['illustris']['xi'][0]:.2f} at r=0.1\n"
    ax.text(0.1, 0.5, stats_text, fontsize=10, family='monospace',
            bbox=dict(boxstyle="round,pad=1.0", facecolor="lightgreen"))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title('Métriques Cosmic DNA', fontsize=14)
    ax.axis('off')
    
    plt.tight_layout()
    plt.show()
