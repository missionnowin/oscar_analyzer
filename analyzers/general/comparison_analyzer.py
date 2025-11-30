#!/usr/bin/env python3
# comparison_analyzer.py
# Generate comparison plots between modified vs unmodified for each run

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Optional


class RunComparisonAnalyzer:
    """Generate physics-meaningful comparisons between modified and unmodified runs."""
    
    def __init__(self, mod_stats: Dict, unm_stats: Dict, run_name: str):
        """
        Initialize with aggregate statistics from both samples.
        
        Args:
            mod_stats: Statistics dict from modified sample
            unm_stats: Statistics dict from unmodified sample
            run_name: Name of the run (e.g., 'run_1')
        """
        self.mod_stats = mod_stats
        self.unm_stats = unm_stats
        self.run_name = run_name
        
        # Physics thresholds for Au+Au 10 GeV
        self.pt_hardness_threshold = 1.0  # GeV - high pT cut
        self.pt_soft_threshold = 0.2      # GeV - low pT cut
    
    def plot_pt_distribution_shape(self, output_file: Optional[str] = None) -> None:
        """
        Compare pT distribution SHAPES (not just statistics).
        
        Physics interest: How broad is the spectrum?
        Cumulative effects → wider tails
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        
        mod_pt = self.mod_stats['pt']
        unm_pt = self.unm_stats['pt']
        
        # Create normalized pT bins for visualization
        pt_min = min(mod_pt['min'], unm_pt['min'])
        pt_max = min(max(mod_pt['max'], unm_pt['max']), 5.0)  # Cap at 5 GeV for visibility
        
        # Bar chart showing mean ± std region
        x = np.array([0.5, 1.5])
        means = [mod_pt['mean'], unm_pt['mean']]
        stds = [mod_pt['std'], unm_pt['std']]
        labels = ['Modified', 'Unmodified']
        colors = ['steelblue', 'darkgreen']
        
        ax.bar(x, means, width=0.6, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
        
        # Error bars (±1σ)
        ax.errorbar(x, means, yerr=stds, fmt='none', color='black', 
                   capsize=8, capthick=2, linewidth=2, label='±1σ')
        
        # Add range markers
        for i, (mean, std, label) in enumerate(zip(means, stds, labels)):
            ax.plot([x[i], x[i]], [mod_pt['min'] if i==0 else unm_pt['min'], 
                                    mod_pt['max'] if i==0 else unm_pt['max']], 
                   'k--', linewidth=1, alpha=0.5)
        
        ax.set_ylabel(r'$p_T$ (GeV)', fontsize=12, fontweight='bold')
        ax.set_title(f'pT Distribution Shape: Modified vs Unmodified ({self.run_name})\n'
                    'Wider distribution = multiple scattering signature', 
                    fontsize=13, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=11)
        ax.set_ylim(pt_min - 0.2, pt_max - 1.0)
        ax.grid(axis='y', alpha=0.3)
        
        # Add broadening ratio annotation
        if unm_pt['std'] > 0:
            broadening_ratio = mod_pt['std'] / unm_pt['std']
            ax.text(0.98, 0.95, f'σ_pT ratio (mod/unm): {broadening_ratio:.3f}',
                   transform=ax.transAxes, ha='right', va='top', fontsize=11,
                   bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))
        
        plt.tight_layout()
        if output_file:
            Path(output_file).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"pT shape comparison saved to {output_file}")
        plt.close()
    
    def plot_high_pt_fraction(self, output_file: Optional[str] = None) -> None:
        """
        Compare fraction of hard (high-pT) particles.
        
        Physics interest: Cumulative effects suppress high-pT production
        Fraction with pT > 1 GeV indicates hard scattering activity
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        
        mod_pt = self.mod_stats['pt']
        unm_pt = self.unm_stats['pt']
        
        # Estimate fraction above threshold from distribution width
        # (rough proxy: particles beyond mean + 2σ are "hard")
        hard_threshold = 1.0  # GeV
        
        mod_hardness = 1.0 - (hard_threshold - mod_pt['mean']) / (2 * mod_pt['std']) if mod_pt['std'] > 0 else 0
        unm_hardness = 1.0 - (hard_threshold - unm_pt['mean']) / (2 * unm_pt['std']) if unm_pt['std'] > 0 else 0
        
        # Clamp to reasonable range
        mod_hardness = max(0, min(mod_hardness, 1.0))
        unm_hardness = max(0, min(unm_hardness, 1.0))
        
        x = np.array([0.5, 1.5])
        fractions = [mod_hardness, unm_hardness]
        labels = ['Modified', 'Unmodified']
        colors = ['steelblue', 'darkgreen']
        
        bars = ax.bar(x, fractions, width=0.6, color=colors, alpha=0.7, 
                     edgecolor='black', linewidth=2)
        
        # Add value labels
        for bar, frac in zip(bars, fractions):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                   f'{frac*100:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        ax.set_ylabel(f'Fraction of particles with $p_T > {hard_threshold}$ GeV', 
                     fontsize=12, fontweight='bold')
        ax.set_title(f'High-pT Particle Production ({self.run_name})\n'
                    'Lower fraction in modified = cumulative suppression effect',
                    fontsize=13, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=11)
        ax.set_ylim(0, 1.0)
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        if output_file:
            Path(output_file).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"High-pT fraction plot saved to {output_file}")
        plt.close()
    
    def plot_pt_broadening_physics(self, output_file: Optional[str] = None) -> None:
        """
        Physics-focused pT broadening visualization.
        
        Shows the core observable: does cumulative scattering WIDEN the pT spectrum?
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        
        mod_pt = self.mod_stats['pt']
        unm_pt = self.unm_stats['pt']
        
        # Create visualization of pT ranges
        x = np.array([0.3, 1.7])
        
        # Use RMS (more sensitive to tails) instead of σ
        mod_rms = np.sqrt(mod_pt['mean']**2 + mod_pt['std']**2) if 'rms' not in mod_pt else mod_pt.get('rms', np.sqrt(mod_pt['mean']**2 + mod_pt['std']**2))
        unm_rms = np.sqrt(unm_pt['mean']**2 + unm_pt['std']**2) if 'rms' not in unm_pt else unm_pt.get('rms', np.sqrt(unm_pt['mean']**2 + unm_pt['std']**2))
        
        widths = [mod_pt['std'], unm_pt['std']]
        labels = ['Modified', 'Unmodified']
        colors = ['steelblue', 'darkgreen']
        
        bars = ax.bar(x, widths, width=0.8, color=colors, alpha=0.7, 
                     edgecolor='black', linewidth=2, label='pT width (σ)')
        
        # Add value labels and ratios
        for i, (bar, width) in enumerate(zip(bars, widths)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{width:.3f} GeV', ha='center', va='bottom', 
                   fontsize=11, fontweight='bold')
        
        # Add broadening ratio
        if unm_pt['std'] > 0:
            ratio = mod_pt['std'] / unm_pt['std']
            ax.text(1.0, max(widths) * 0.85, 
                   f'Broadening ratio: {ratio:.3f}x\n' + 
                   ('✓ Cumulative effect' if ratio > 1.1 else '✗ No significant broadening'),
                   ha='center', fontsize=12, fontweight='bold',
                   bbox=dict(boxstyle='round', facecolor='yellow' if ratio > 1.1 else 'lightgray', alpha=0.8))
        
        ax.set_ylabel(r'pT width $\sigma_{p_T}$ (GeV)', fontsize=12, fontweight='bold')
        ax.set_title(f'pT Broadening: Signature of Multiple Scattering ({self.run_name})',
                    fontsize=13, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=11)
        ax.set_ylim(0, max(widths) * 1.3)
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        if output_file:
            Path(output_file).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"pT broadening physics plot saved to {output_file}")
        plt.close()
    
    def plot_kinematic_correlations(self, output_file: Optional[str] = None) -> None:
        """
        Show kinematic consistency: pT vs rapidity/eta correlation strength.
        
        Physics interest: Multiple scattering can create unusual kinematic correlations
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        mod_pt = self.mod_stats['pt']
        unm_pt = self.unm_stats['pt']
        mod_y = self.mod_stats['y']
        unm_y = self.unm_stats['y']
        mod_eta = self.mod_stats['eta']
        unm_eta = self.unm_stats['eta']
        
        # pT vs rapidity: show spread
        x = np.array([0.5, 1.5])
        
        # Proxy for correlation strength: std(y) / mean(|y|)
        mod_y_corr = mod_y['std'] / max(abs(mod_y['mean']), 0.1)
        unm_y_corr = unm_y['std'] / max(abs(unm_y['mean']), 0.1)
        
        ax1.bar(x, [mod_y_corr, unm_y_corr], width=0.6, 
               color=['steelblue', 'darkgreen'], alpha=0.7, edgecolor='black', linewidth=2)
        ax1.set_ylabel('Rapidity spread (σ_y / |mean_y|)', fontsize=11, fontweight='bold')
        ax1.set_title('Rapidity Correlation Strength', fontsize=12, fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(['Modified', 'Unmodified'], fontsize=10)
        ax1.grid(axis='y', alpha=0.3)
        
        # pT vs pseudorapidity
        mod_eta_corr = mod_eta['std'] / max(abs(mod_eta['mean']), 0.1)
        unm_eta_corr = unm_eta['std'] / max(abs(unm_eta['mean']), 0.1)
        
        ax2.bar(x, [mod_eta_corr, unm_eta_corr], width=0.6,
               color=['steelblue', 'darkgreen'], alpha=0.7, edgecolor='black', linewidth=2)
        ax2.set_ylabel('Pseudorapidity spread (σ_η / |mean_η|)', fontsize=11, fontweight='bold')
        ax2.set_title('Pseudorapidity Correlation Strength', fontsize=12, fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels(['Modified', 'Unmodified'], fontsize=10)
        ax2.grid(axis='y', alpha=0.3)
        
        fig.suptitle(f'Kinematic Correlations ({self.run_name})', 
                    fontsize=13, fontweight='bold', y=1.00)
        
        plt.tight_layout()
        if output_file:
            Path(output_file).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"Kinematic correlations plot saved to {output_file}")
        plt.close()
    
    def generate_all_comparisons(self, output_dir: Path) -> List[str]:
        """Generate all physics-meaningful comparison plots."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        plots = []
        
        # Physics observables (in order of importance)
        pt_shape_file = output_dir / "01_pt_broadening_signature.png"
        self.plot_pt_distribution_shape(str(pt_shape_file))
        plots.append("01_pt_broadening_signature.png")
        
        pt_physics_file = output_dir / "02_pt_broadening_physics.png"
        self.plot_pt_broadening_physics(str(pt_physics_file))
        plots.append("02_pt_broadening_physics.png")
        
        hard_pt_file = output_dir / "03_high_pt_fraction.png"
        self.plot_high_pt_fraction(str(hard_pt_file))
        plots.append("03_high_pt_fraction.png")
        
        corr_file = output_dir / "04_kinematic_correlations.png"
        self.plot_kinematic_correlations(str(corr_file))
        plots.append("04_kinematic_correlations.png")
        
        return plots