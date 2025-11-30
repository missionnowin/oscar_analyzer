# cumulative_detector.py
# Cumulative Effect Detector for UrQMD Collisions
# Analyzes modified (cumulative) vs unmodified collision data
# Identifies signatures of cumulative scattering effects

import sys
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Optional

# Physical constants
M_NUCLEON = 0.938  # GeV (nucleon mass)
M_PION = 0.140     # GeV (for reference)

try:
    from models.cumulative_singnature import CumulativeSignature
except ImportError as e:
    print(f"Error importing modules: {e}")
    sys.exit(1)

class CumulativeEffectDetector:
    """
    Detect cumulative scattering effects in collision data
    
    CRITERIA FOR CUMULATIVE PARTICLES (e.g. Au+Au 10 GeV):
    ──────────────────────────────────────────────────
    1. Cumulative variable x > 1.0 (or x > 1.1 for safety)
       where x = (E - p_L) / m_N
       - E: particle total energy
       - p_L: longitudinal momentum (beam direction)
       - m_N: nucleon mass
    
    2. Forward direction: x_F > 0.3 (Feynman scaling variable)
    
    3. Non-relativistic regime: β < 0.95 (v/c < 0.95)
    
    Physical interpretation:
    - x > 1 means particle uses >1 nucleon's worth of momentum
    - Only possible with multi-nucleon correlation
    - Signature of cumulative scattering
    """
    
    def __init__(self, particles_modified: List, particles_unmodified: List = None):
        self.particles_modified = particles_modified
        self.particles_unmodified = particles_unmodified
        self.signatures = []
        self.comparison_stats = {}
        
        # Au+Au 10 GeV specific parameters
        self.cumulative_threshold = 1.1  # x > 1.1 for cumulative
        self.forward_cut_xf = 0.3        # x_F > 0.3
        self.cm_energy = 10.0            # GeV (√s_NN)
    
    @staticmethod
    def calculate_cumulative_variable(px, py, pz, mass=0.938):
        """
        Calculate cumulative variable x = (E - p_L) / m_N
        
        Args:
            px, py, pz: Cartesian momentum components (GeV)
            mass: Particle rest mass (default: nucleon mass 0.938 GeV)
        
        
        Returns:
            Cumulative variable x (dimensionless)
        """
        # Total energy: E = sqrt(p^2 + m^2)
        p_mag = np.sqrt(px**2 + py**2 + pz**2)
        E = np.sqrt(p_mag**2 + mass**2)
        
        # Longitudinal momentum (beam direction is z)
        p_L = pz
        
        # Cumulative variable
        x = (E - p_L) / M_NUCLEON
        
        return x
    
    @staticmethod
    def calculate_feynman_xf(pz, p_L_max=None):
        """
        Calculate Feynman scaling variable x_F = p_L / p_L_max
        
        For Au+Au 10 GeV: p_L_max ≈ sqrt(s_NN)/2 ≈ 5 GeV/nucleon
        
        Args:
            pz: Longitudinal momentum (GeV)
            p_L_max: Maximum longitudinal momentum (default for 10 GeV)
        
        Returns:
            Feynman x_F (-1 to +1, positive = forward)
        """
        if p_L_max is None:
            p_L_max = 10.0 / 2.0  # √s_NN / 2 for 10 GeV
        
        xf = pz / p_L_max
        return np.clip(xf, -1.0, 1.0)
    
    def identify_cumulative_particles(self, particles: List) -> Dict:
        """
        Identify particles satisfying cumulative criteria
        
        Returns:
            {
                'cumulative_mask': boolean array,
                'x_values': cumulative variables,
                'xf_values': Feynman variables,
                'n_cumulative': count of cumulative particles,
                'fraction': fraction of all particles
            }
        """
        if not particles:
            return {'cumulative_mask': np.array([]), 'x_values': np.array([])}
        
        # Calculate x and x_F for all particles
        x_array = np.array([
            self.calculate_cumulative_variable(p.px, p.py, p.pz, p.mass)
            for p in particles
        ])
        
        xf_array = np.array([
            self.calculate_feynman_xf(p.pz)
            for p in particles
        ])
        
        # Cumulative criteria: x > threshold AND forward direction
        cumulative_mask = (x_array > self.cumulative_threshold) & (xf_array > self.forward_cut_xf)
        
        return {
            'cumulative_mask': cumulative_mask,
            'x_values': x_array,
            'xf_values': xf_array,
            'n_cumulative': np.sum(cumulative_mask),
            'fraction': np.sum(cumulative_mask) / len(particles) if len(particles) > 0 else 0,
            'x_mean': np.mean(x_array[cumulative_mask]) if np.sum(cumulative_mask) > 0 else 0,
            'x_max': np.max(x_array) if len(x_array) > 0 else 0
        }
    
    def detect_cumulative_signal(self) -> Optional[CumulativeSignature]:
        """
        PRIMARY DETECTOR: Direct identification via cumulative variable x
        """
        cum_stats_mod = self.identify_cumulative_particles(self.particles_modified)
        n_cum_mod = cum_stats_mod['n_cumulative']
        frac_cum_mod = cum_stats_mod['fraction']
        
        if n_cum_mod == 0:
            return None
        
        if not self.particles_unmodified:
            # Single sample: presence of x > 1.1 particles indicates cumulative effects
            strength = min(frac_cum_mod / 0.1, 1.0)  # 10% cumulative → strength=1.0
            
            return CumulativeSignature(
                signature_type='cumulative_variable',
                strength=strength,
                confidence=0.95,  # High confidence - direct kinematic criterion
                affected_particles=n_cum_mod,
                description=f'Direct cumulative detection: {n_cum_mod} particles with x > {self.cumulative_threshold} '
                           f'({frac_cum_mod*100:.1f}% of sample). Max x = {cum_stats_mod["x_max"]:.2f}. '
                           f'These particles require multi-nucleon correlation.'
            )
        else:
            # Comparison mode
            cum_stats_unmod = self.identify_cumulative_particles(self.particles_unmodified)
            n_cum_unmod = cum_stats_unmod['n_cumulative']
            frac_cum_unmod = cum_stats_unmod['fraction']
            
            # Modified should have MORE cumulative particles
            if frac_cum_mod > frac_cum_unmod:
                frac_increase = (frac_cum_mod - frac_cum_unmod) / max(frac_cum_unmod, 0.01)
                strength = min(frac_increase / 2.0, 1.0)
                
                return CumulativeSignature(
                    signature_type='cumulative_variable',
                    strength=strength,
                    confidence=0.95,
                    affected_particles=n_cum_mod,
                    description=f'Cumulative effect enhancement: Modified sample has {frac_cum_mod*100:.1f}% '
                               f'cumulative particles vs {frac_cum_unmod*100:.1f}% in unmodified. '
                               f'Ratio: {frac_cum_mod/max(frac_cum_unmod, 0.001):.1f}x. '
                               f'Modified max x = {cum_stats_mod["x_max"]:.2f}, Unmodified max x = {cum_stats_unmod["x_max"]:.2f}'
                )
        
        return None
    
    def detect_pt_broadening(self, threshold: float = 0.20) -> Optional[CumulativeSignature]:
        """
        SECONDARY DETECTOR: pT broadening from multiple scattering
        
        For Au+Au 10 GeV, typical pT broadening is 15-25% for cumulative effects
        """
        if not self.particles_unmodified:
            return None  # Need comparison for reliable detection
        
        # Calculate pT for both samples
        pt_mod = np.array([np.sqrt(p.px**2 + p.py**2) for p in self.particles_modified])
        pt_unmod = np.array([np.sqrt(p.px**2 + p.py**2) for p in self.particles_unmodified])
        
        std_mod = np.std(pt_mod)
        std_unmod = np.std(pt_unmod)
        
        if std_unmod > 0:
            broadening_ratio = std_mod / std_unmod
            
            if broadening_ratio > (1.0 + threshold):
                strength = min((broadening_ratio - 1.0) / 0.3, 1.0)
                
                return CumulativeSignature(
                    signature_type='pt_broadening',
                    strength=strength,
                    confidence=0.75,  # Lower confidence than direct criterion
                    affected_particles=len(self.particles_modified),
                    description=f'pT broadening (indirect signature): {broadening_ratio:.2f}x increase. '
                               f'Unmodified σ_pT = {std_unmod:.3f} GeV, '
                               f'Modified σ_pT = {std_mod:.3f} GeV. '
                               f'Consistent with cumulative scattering.'
                )
        
        return None
    
    def detect_forward_enhancement(self) -> Optional[CumulativeSignature]:
        """
        SECONDARY DETECTOR: Enhanced forward (large x_F) particle production
        
        Cumulative particles concentrate in forward direction
        """
        xf_mod = np.array([self.calculate_feynman_xf(p.pz) for p in self.particles_modified])
        
        # Fraction in forward region (x_F > 0.3)
        frac_forward_mod = np.sum(xf_mod > 0.3) / len(xf_mod)
        
        if not self.particles_unmodified:
            if frac_forward_mod > 0.35:
                return CumulativeSignature(
                    signature_type='forward_enhancement',
                    strength=min((frac_forward_mod - 0.25) / 0.2, 1.0),
                    confidence=0.65,
                    affected_particles=int(frac_forward_mod * len(self.particles_modified)),
                    description=f'Forward particle enhancement: {frac_forward_mod*100:.1f}% with x_F > 0.3. '
                               f'Cumulative particles preferentially produce forward.'
                )
        else:
            xf_unmod = np.array([self.calculate_feynman_xf(p.pz) for p in self.particles_unmodified])
            frac_forward_unmod = np.sum(xf_unmod > 0.3) / len(xf_unmod)
            
            if frac_forward_mod > frac_forward_unmod * 1.2:
                return CumulativeSignature(
                    signature_type='forward_enhancement',
                    strength=min((frac_forward_mod - frac_forward_unmod) / 0.1, 1.0),
                    confidence=0.70,
                    affected_particles=int((frac_forward_mod - frac_forward_unmod) * len(self.particles_modified)),
                    description=f'Forward particle enhancement: Modified {frac_forward_mod*100:.1f}% '
                               f'vs Unmodified {frac_forward_unmod*100:.1f}% with x_F > 0.3.'
                )
        
        return None
    
    def detect_all_signatures(self) -> List[CumulativeSignature]:
        """Run all detectors in priority order"""
        self.signatures = []
        
        # Priority 1: Direct cumulative variable detection (highest confidence)
        sig = self.detect_cumulative_signal()
        if sig:
            self.signatures.append(sig)
        
        # Priority 2: Secondary indicators (lower confidence)
        for detector in [
            self.detect_forward_enhancement,
            self.detect_pt_broadening
        ]:
            sig = detector()
            if sig:
                self.signatures.append(sig)
        
        return self.signatures
    
    def print_report(self) -> None:
        """Print detailed cumulative effect report"""
        if not self.signatures:
            self.detect_all_signatures()
        
        print("\n" + "="*80)
        print("CUMULATIVE SCATTERING EFFECT ANALYSIS - Au+Au 10 GeV")
        print("="*80)
        
        # Show cumulative variable analysis
        print("\n📊 CUMULATIVE VARIABLE ANALYSIS (x = (E - p_L) / m_N):")
        print("-" * 80)
        
        cum_mod = self.identify_cumulative_particles(self.particles_modified)
        print(f"Modified sample:")
        print(f"  Cumulative particles (x > {self.cumulative_threshold}): {cum_mod['n_cumulative']} / {len(self.particles_modified)} "
              f"({cum_mod['fraction']*100:.1f}%)")
        print(f"  Max x observed: {cum_mod['x_max']:.2f}")
        print(f"  Mean x (cumulative particles): {cum_mod['x_mean']:.2f}")
        
        if self.particles_unmodified:
            cum_unmod = self.identify_cumulative_particles(self.particles_unmodified)
            print(f"\nUnmodified sample:")
            print(f"  Cumulative particles (x > {self.cumulative_threshold}): {cum_unmod['n_cumulative']} / {len(self.particles_unmodified)} "
                  f"({cum_unmod['fraction']*100:.1f}%)")
            print(f"  Max x observed: {cum_unmod['x_max']:.2f}")
        
        # Show detected signatures
        print(f"\n🔍 DETECTED SIGNATURES ({len(self.signatures)} total):")
        print("-" * 80)
        
        if not self.signatures:
            print("No significant cumulative effects detected.")
        else:
            for i, sig in enumerate(self.signatures, 1):
                print(f"\n{i}. {sig.signature_type.upper()}")
                print(f"   Strength: {sig.strength:.2f} | Confidence: {sig.confidence:.2f}")
                print(f"   Affected: {sig.affected_particles} particles")
                print(f"   {sig.description}")
        
        print(f"\n{'='*80}\n")
    
    def get_cumulative_likelihood(self) -> float:
        """
        Calculate overall cumulative effect likelihood
        
        Heavily weighted toward direct cumulative variable detection
        """
        if not self.signatures:
            self.detect_all_signatures()
        
        if not self.signatures:
            return 0.0
        
        # Primary signature (cumulative_variable) gets 70% weight
        primary_score = 0.0
        secondary_score = 0.0
        
        for sig in self.signatures:
            if sig.signature_type == 'cumulative_variable':
                primary_score = sig.strength * sig.confidence * 0.7
            else:
                secondary_score += sig.strength * sig.confidence * 0.15
        
        overall = min(primary_score + secondary_score, 1.0)
        return overall


class CumulativeComparisonVisualizer:
    """Visualization for cumulative effect analysis"""
    
    def __init__(self, detector: CumulativeEffectDetector):
        self.detector = detector
    
    def plot_cumulative_variable(self, output_file: Optional[str] = None) -> None:
        """Plot cumulative variable x = (E - p_L) / m_N distributions"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle('Cumulative Variable Distribution: x = (E - p_L) / m_N', 
                    fontsize=14, fontweight='bold')
        
        x_mod = np.array([
            self.detector.calculate_cumulative_variable(p.px, p.py, p.pz, p.mass)
            for p in self.detector.particles_modified
        ])
        
        # Modified distribution
        ax = axes[0]
        ax.hist(x_mod, bins=50, color='steelblue', alpha=0.7, edgecolor='black', range=(0, 3))
        ax.axvline(self.detector.cumulative_threshold, color='red', linestyle='--', 
                  linewidth=2, label=f'Cumulative threshold (x={self.detector.cumulative_threshold})')
        ax.set_xlabel('Cumulative Variable x', fontsize=11)
        ax.set_ylabel('Count', fontsize=11)
        ax.set_title('Modified (Cumulative)', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)
        
        # Unmodified or statistics
        if self.detector.particles_unmodified:
            ax = axes[1]
            x_unmod = np.array([
                self.detector.calculate_cumulative_variable(p.px, p.py, p.pz, p.mass)
                for p in self.detector.particles_unmodified
            ])
            ax.hist(x_unmod, bins=50, color='darkgreen', alpha=0.7, edgecolor='black', range=(0, 3))
            ax.axvline(self.detector.cumulative_threshold, color='red', linestyle='--',
                      linewidth=2, label=f'Cumulative threshold (x={self.detector.cumulative_threshold})')
            ax.set_xlabel('Cumulative Variable x', fontsize=11)
            ax.set_ylabel('Count', fontsize=11)
            ax.set_title('Unmodified (Baseline)', fontsize=12, fontweight='bold')
            ax.legend()
            ax.grid(alpha=0.3)
        else:
            ax = axes[1]
            ax.axis('off')
            cum_mod = self.detector.identify_cumulative_particles(self.detector.particles_modified)
            stats_text = f"""
Cumulative Particle Statistics:
━━━━━━━━━━━━━━━━━━━━━━━━━━━
Particles with x > {self.detector.cumulative_threshold}: {cum_mod['n_cumulative']}
Fraction: {cum_mod['fraction']*100:.1f}%
Mean x (cumulative): {cum_mod['x_mean']:.2f}
Max x: {cum_mod['x_max']:.2f}

Threshold: x > 1 means particle
uses >1 nucleon's worth of momentum
→ requires multi-nucleon correlation
            """
            ax.text(0.1, 0.5, stats_text, fontsize=10, family='monospace',
                   verticalalignment='center',
                   bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
        
        plt.tight_layout()
        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"Cumulative variable plot saved to {output_file}")
        plt.show()
    
    def plot_signatures(self, output_file: Optional[str] = None) -> None:
        """Plot detected signatures"""
        if not self.detector.signatures:
            self.detector.detect_all_signatures()
        
        if not self.detector.signatures:
            print("No signatures to plot")
            return
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        sig_names = [s.signature_type for s in self.detector.signatures]
        strengths = [s.strength for s in self.detector.signatures]
        confidences = [s.confidence for s in self.detector.signatures]
        
        x = np.arange(len(sig_names))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, strengths, width, label='Strength', 
                      color='steelblue', alpha=0.7, edgecolor='black')
        bars2 = ax.bar(x + width/2, confidences, width, label='Confidence',
                      color='darkgreen', alpha=0.7, edgecolor='black')
        
        ax.set_ylabel('Score (0-1)', fontsize=12)
        ax.set_title('Cumulative Effect Signatures - Au+Au 10 GeV', fontsize=13, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(sig_names, rotation=15, ha='right')
        ax.legend()
        ax.set_ylim(0, 1.1)
        ax.grid(axis='y', alpha=0.3)
        
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.2f}', ha='center', va='bottom', fontsize=9)
        
        likelihood = self.detector.get_cumulative_likelihood()
        fig.text(0.99, 0.01, f'Overall Likelihood: {likelihood:.2f}',
                ha='right', va='bottom', fontsize=11, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
        
        plt.tight_layout()
        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"Signatures plot saved to {output_file}")
        plt.show()
