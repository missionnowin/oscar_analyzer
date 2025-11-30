# angle_analyzer.py
# UrQMD Oscar Format Angle Distribution Analyzer
# Supports both Oscar1992A (f19) and OSC1997A (final_id_p_x) formats
# Reads Oscar files and generates angle distribution visualizations
# With multi-file processing and parallel execution support

import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
from typing import List, Tuple, Optional, Dict
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
from functools import partial
import time
import json

try:
    from utils.readers import OscarFormat, OscarReader
    from models.particle import Particle
except ImportError as e:
    print(f"Error importing modules: {e}")
    sys.exit(1)

class AngleAnalyzer:
    """Calculate and analyze angle distributions"""
    
    @staticmethod
    def polar_angle(particle: Particle) -> float:
        """Calculate polar angle theta (0 to pi radians)
        Angle relative to beam axis (z-axis)
        """
        pt = np.sqrt(particle.px**2 + particle.py**2)
        theta = np.arctan2(pt, particle.pz)
        return theta
    
    @staticmethod
    def azimuthal_angle(particle: Particle) -> float:
        """Calculate azimuthal angle phi (0 to 2pi radians)
        Angle in transverse plane
        """
        phi = np.arctan2(particle.py, particle.px)
        if phi < 0:
            phi += 2 * np.pi
        return phi
    
    @staticmethod
    def transverse_momentum(particle: Particle) -> float:
        """Calculate transverse momentum magnitude"""
        return np.sqrt(particle.px**2 + particle.py**2)
    
    @staticmethod
    def rapidity(particle: Particle) -> Optional[float]:
        """Calculate rapidity y = 0.5 * ln((E+pz)/(E-pz))"""
        numerator = particle.E + particle.pz
        denominator = particle.E - particle.pz
        if denominator > 0:
            return 0.5 * np.log(numerator / denominator)
        return None
    
    @staticmethod
    def pseudorapidity(particle: Particle) -> float:
        """Calculate pseudorapidity eta = -ln(tan(theta/2))"""
        pt = np.sqrt(particle.px**2 + particle.py**2)
        theta = np.arctan2(pt, particle.pz)
        
        # Avoid division by zero
        if np.sin(theta / 2) == 0:
            return 0.0
        
        eta = -np.log(np.tan(theta / 2))
        return eta


class AngleDistributionPlotter:
    """Visualize angle distributions"""
    
    def __init__(self, particles: List[Particle], figsize: Tuple = (14, 10)):
        self.particles = particles
        self.figsize = figsize
        self.analyzer = AngleAnalyzer()
        
    def plot_distributions(self, output_file: Optional[str] = None) -> None:
        """Create comprehensive angle distribution plots"""
        
        # Calculate angles for all particles
        polar_angles = [self.analyzer.polar_angle(p) for p in self.particles]
        azimuthal_angles = [self.analyzer.azimuthal_angle(p) for p in self.particles]
        pt = [self.analyzer.transverse_momentum(p) for p in self.particles]
        
        # Convert to degrees
        polar_deg = np.degrees(polar_angles)
        azimuthal_deg = np.degrees(azimuthal_angles)
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=self.figsize)
        fig.suptitle('UrQMD Angle Distribution Analysis', fontsize=16, fontweight='bold')
        
        # 1. Polar angle histogram
        ax = axes[0, 0]
        ax.hist(polar_deg, bins=36, color='steelblue', edgecolor='black', alpha=0.7)
        ax.set_xlabel('Polar Angle θ (degrees)', fontsize=11)
        ax.set_ylabel('Number of Particles', fontsize=11)
        ax.set_title('Polar Angle Distribution', fontsize=12, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        ax.set_xlim(0, 180)
        
        # Add statistics
        mean_theta = np.mean(polar_deg)
        ax.axvline(mean_theta, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_theta:.1f}°')
        ax.legend()
        
        # 2. Azimuthal angle histogram
        ax = axes[0, 1]
        ax.hist(azimuthal_deg, bins=36, color='darkgreen', edgecolor='black', alpha=0.7)
        ax.set_xlabel('Azimuthal Angle φ (degrees)', fontsize=11)
        ax.set_ylabel('Number of Particles', fontsize=11)
        ax.set_title('Azimuthal Angle Distribution', fontsize=12, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        ax.set_xlim(0, 360)
        
        # 3. 2D scatter: polar vs azimuthal
        ax = axes[1, 0]
        scatter = ax.scatter(azimuthal_deg, polar_deg, c=pt, cmap='viridis', 
                           alpha=0.6, s=30, edgecolors='black', linewidth=0.5)
        ax.set_xlabel('Azimuthal Angle φ (degrees)', fontsize=11)
        ax.set_ylabel('Polar Angle θ (degrees)', fontsize=11)
        ax.set_title('Polar vs Azimuthal Angles (colored by pT)', fontsize=12, fontweight='bold')
        ax.set_xlim(0, 360)
        ax.set_ylim(0, 180)
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Transverse Momentum (GeV)', fontsize=10)
        ax.grid(alpha=0.3)
        
        # 4. Polar angle with cos(theta) for physics insight
        ax = axes[1, 1]
        cos_theta = np.cos(np.radians(polar_deg))
        ax.hist(cos_theta, bins=40, color='coral', edgecolor='black', alpha=0.7)
        ax.set_xlabel('cos(θ)', fontsize=11)
        ax.set_ylabel('Number of Particles', fontsize=11)
        ax.set_title('cos(θ) Distribution', fontsize=12, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {output_file}")
        
        plt.show()
    
    def plot_single_distribution(self, angle_type: str = 'polar', 
                                 output_file: Optional[str] = None) -> None:
        """Plot single angle distribution (polar or azimuthal)"""
        
        if angle_type == 'polar':
            angles = [self.analyzer.polar_angle(p) for p in self.particles]
            angles_deg = np.degrees(angles)
            xlabel = 'Polar Angle θ (degrees)'
            title = 'Polar Angle Distribution'
            xlim = (0, 180)
            color = 'steelblue'
        elif angle_type == 'azimuthal':
            angles = [self.analyzer.azimuthal_angle(p) for p in self.particles]
            angles_deg = np.degrees(angles)
            xlabel = 'Azimuthal Angle φ (degrees)'
            title = 'Azimuthal Angle Distribution'
            xlim = (0, 360)
            color = 'darkgreen'
        else:  # pseudorapidity
            angles = [self.analyzer.pseudorapidity(p) for p in self.particles]
            angles_deg = np.array(angles)
            xlabel = 'Pseudorapidity η'
            title = 'Pseudorapidity Distribution'
            xlim = None
            color = 'purple'
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(angles_deg, bins=40, color=color, edgecolor='black', alpha=0.7)
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel('Number of Particles', fontsize=12)
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        
        if xlim:
            ax.set_xlim(xlim)
        
        # Add statistics
        mean_angle = np.mean(angles_deg)
        std_angle = np.std(angles_deg)
        ax.text(0.98, 0.97, f'Mean: {mean_angle:.1f}\nStd: {std_angle:.1f}',
               transform=ax.transAxes, verticalalignment='top', 
               horizontalalignment='right', bbox=dict(boxstyle='round', 
               facecolor='wheat', alpha=0.8), fontsize=11)
        
        plt.tight_layout()
        
        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {output_file}")
        
        plt.show()
    
    def plot_pt_distribution(self, output_file: Optional[str] = None) -> None:
        """Plot transverse momentum distribution"""
        pt = [self.analyzer.transverse_momentum(p) for p in self.particles]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(pt, bins=50, color='teal', edgecolor='black', alpha=0.7)
        ax.set_xlabel('Transverse Momentum pT (GeV)', fontsize=12)
        ax.set_ylabel('Number of Particles', fontsize=12)
        ax.set_title('Transverse Momentum Distribution', fontsize=13, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        ax.set_yscale('log')
        
        # Add statistics
        mean_pt = np.mean(pt)
        std_pt = np.std(pt)
        ax.text(0.98, 0.97, f'Mean: {mean_pt:.2f} GeV\nStd: {std_pt:.2f} GeV',
               transform=ax.transAxes, verticalalignment='top', 
               horizontalalignment='right', bbox=dict(boxstyle='round', 
               facecolor='wheat', alpha=0.8), fontsize=11)
        
        plt.tight_layout()
        
        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {output_file}")
        
        plt.show()


class EventStatistics:
    """Calculate event-level statistics"""
    
    def __init__(self, particles: List[Particle]):
        self.particles = particles
        self.analyzer = AngleAnalyzer()
    
    def print_summary(self) -> None:
        """Print summary statistics for event"""
        n_particles = len(self.particles)
        
        # Momenta
        px = np.array([p.px for p in self.particles])
        py = np.array([p.py for p in self.particles])
        pz = np.array([p.pz for p in self.particles])
        pt = np.sqrt(px**2 + py**2)
        
        # Angles
        polar_angles = [self.analyzer.polar_angle(p) for p in self.particles]
        azimuthal_angles = [self.analyzer.azimuthal_angle(p) for p in self.particles]
        
        print("\n" + "="*60)
        print("EVENT STATISTICS")
        print("="*60)
        print(f"Total particles: {n_particles}")
        print(f"\nMomenta (GeV):")
        print(f"  px: mean={np.mean(px):.3f}, std={np.std(px):.3f}")
        print(f"  py: mean={np.mean(py):.3f}, std={np.std(py):.3f}")
        print(f"  pz: mean={np.mean(pz):.3f}, std={np.std(pz):.3f}")
        print(f"  pT: mean={np.mean(pt):.3f}, std={np.std(pt):.3f}, max={np.max(pt):.3f}")
        print(f"\nAngles (degrees):")
        print(f"  θ: mean={np.degrees(np.mean(polar_angles)):.1f}, std={np.degrees(np.std(polar_angles)):.1f}")
        print(f"  φ: mean={np.degrees(np.mean(azimuthal_angles)):.1f}, std={np.degrees(np.std(azimuthal_angles)):.1f}")
        print("="*60 + "\n")
    
    def get_statistics_dict(self) -> Dict[str, float]:
        """Return statistics as dictionary for aggregation"""
        n_particles = len(self.particles)
        px = np.array([p.px for p in self.particles])
        py = np.array([p.py for p in self.particles])
        pz = np.array([p.pz for p in self.particles])
        pt = np.sqrt(px**2 + py**2)
        
        polar_angles = [self.analyzer.polar_angle(p) for p in self.particles]
        azimuthal_angles = [self.analyzer.azimuthal_angle(p) for p in self.particles]
        
        return {
            'n_particles': n_particles,
            'px_mean': np.mean(px),
            'px_std': np.std(px),
            'py_mean': np.mean(py),
            'py_std': np.std(py),
            'pz_mean': np.mean(pz),
            'pz_std': np.std(pz),
            'pt_mean': np.mean(pt),
            'pt_std': np.std(pt),
            'pt_max': np.max(pt),
            'theta_mean': np.degrees(np.mean(polar_angles)),
            'theta_std': np.degrees(np.std(polar_angles)),
            'phi_mean': np.degrees(np.mean(azimuthal_angles)),
            'phi_std': np.degrees(np.std(azimuthal_angles))
        }


class MultiFileProcessor:
    """Process multiple Oscar files in parallel or sequential"""
    
    def __init__(self, use_parallel: bool = True, n_workers: Optional[int] = None):
        self.use_parallel = use_parallel
        self.n_workers = n_workers or mp.cpu_count()
        self.results = {}
    
    @staticmethod
    def process_single_file(filepath: str, event_idx: int = 0) -> Dict:
        """Process a single file and return statistics
        
        This function must be at module level for multiprocessing
        """
        try:
            reader = OscarReader(filepath)
            events = reader.read_file()
            
            if not events or event_idx >= len(events):
                return {
                    'filepath': filepath,
                    'success': False,
                    'error': f'Event {event_idx} not found',
                    'n_events': len(events) if events else 0
                }
            
            particles = events[event_idx]
            stats = EventStatistics(particles)
            stats_dict = stats.get_statistics_dict()
            stats_dict['filepath'] = filepath
            stats_dict['success'] = True
            stats_dict['n_events'] = len(events)
            stats_dict['event_idx'] = event_idx
            
            return stats_dict
            
        except Exception as e:
            return {
                'filepath': filepath,
                'success': False,
                'error': str(e)
            }
    
    def process_files(self, filepaths: List[str], event_idx: int = 0) -> Dict[str, Dict]:
        """Process multiple files
        
        Args:
            filepaths: List of file paths to process
            event_idx: Event index to analyze in each file
            
        Returns:
            Dictionary with results for each file
        """
        print(f"\nProcessing {len(filepaths)} file(s)...")
        start_time = time.time()
        
        if self.use_parallel and len(filepaths) > 1:
            print(f"Using parallel processing with {self.n_workers} workers")
            with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
                process_func = partial(self.process_single_file, event_idx=event_idx)
                results = list(executor.map(process_func, filepaths))
        else:
            print("Using sequential processing")
            results = [
                self.process_single_file(fp, event_idx) 
                for fp in filepaths
            ]
        
        # Organize results by filepath
        self.results = {r['filepath']: r for r in results}
        
        elapsed = time.time() - start_time
        print(f"Completed in {elapsed:.2f} seconds\n")
        
        return self.results
    
    def print_summary(self) -> None:
        """Print summary of all processed files"""
        successful = [r for r in self.results.values() if r.get('success', False)]
        failed = [r for r in self.results.values() if not r.get('success', False)]
        
        print("\n" + "="*80)
        print("MULTI-FILE PROCESSING SUMMARY")
        print("="*80)
        print(f"Total files: {len(self.results)}")
        print(f"Successful: {len(successful)}")
        print(f"Failed: {len(failed)}")
        
        if successful:
            print("\n--- SUCCESSFUL FILES ---")
            for r in successful:
                print(f"\n{Path(r['filepath']).name}:")
                print(f"  Events: {r['n_events']}, Particles: {r['n_particles']}")
                print(f"  pT (mean/std): {r['pt_mean']:.3f} / {r['pt_std']:.3f} GeV")
                print(f"  θ (mean/std): {r['theta_mean']:.1f}° / {r['theta_std']:.1f}°")
        
        if failed:
            print("\n--- FAILED FILES ---")
            for r in failed:
                print(f"\n{Path(r['filepath']).name}:")
                print(f"  Error: {r.get('error', 'Unknown error')}")
        
        print("\n" + "="*80 + "\n")
    
    def save_results_json(self, output_file: str) -> None:
        """Save results to JSON file"""
        # Convert numpy types to native Python types for JSON serialization
        results_serializable = {}
        for filepath, data in self.results.items():
            results_serializable[filepath] = {
                k: float(v) if isinstance(v, (np.floating, np.integer)) else v
                for k, v in data.items()
            }
        
        with open(output_file, 'w') as f:
            json.dump(results_serializable, f, indent=2)
        
        print(f"Results saved to {output_file}")
    
    def plot_comparison(self, output_file: Optional[str] = None) -> None:
        """Plot comparison across multiple files"""
        successful = [r for r in self.results.values() if r.get('success', False)]
        
        if not successful:
            print("No successful results to plot")
            return
        
        filenames = [Path(r['filepath']).stem for r in successful]
        pt_means = [r['pt_mean'] for r in successful]
        pt_stds = [r['pt_std'] for r in successful]
        theta_means = [r['theta_mean'] for r in successful]
        theta_stds = [r['theta_std'] for r in successful]
        n_particles = [r['n_particles'] for r in successful]
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Multi-File Comparison', fontsize=16, fontweight='bold')
        
        # pT comparison
        ax = axes[0, 0]
        x = np.arange(len(filenames))
        ax.bar(x, pt_means, yerr=pt_stds, capsize=5, color='steelblue', alpha=0.7)
        ax.set_ylabel('Mean pT (GeV)', fontsize=11)
        ax.set_title('Transverse Momentum Comparison', fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(filenames, rotation=45, ha='right')
        ax.grid(axis='y', alpha=0.3)
        
        # θ comparison
        ax = axes[0, 1]
        ax.bar(x, theta_means, yerr=theta_stds, capsize=5, color='darkgreen', alpha=0.7)
        ax.set_ylabel('Mean θ (degrees)', fontsize=11)
        ax.set_title('Polar Angle Comparison', fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(filenames, rotation=45, ha='right')
        ax.grid(axis='y', alpha=0.3)
        
        # Particle count
        ax = axes[1, 0]
        ax.bar(x, n_particles, color='teal', alpha=0.7)
        ax.set_ylabel('Number of Particles', fontsize=11)
        ax.set_title('Event Particle Count', fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(filenames, rotation=45, ha='right')
        ax.grid(axis='y', alpha=0.3)
        
        # Statistics table
        ax = axes[1, 1]
        ax.axis('tight')
        ax.axis('off')
        
        table_data = [['File', 'Particles', 'pT (mean)', 'θ (mean)']]
        for i, fname in enumerate(filenames):
            table_data.append([
                fname,
                f"{n_particles[i]}",
                f"{pt_means[i]:.2f}",
                f"{theta_means[i]:.1f}°"
            ])
        
        table = ax.table(cellText=table_data, cellLoc='center', loc='center',
                        colWidths=[0.25, 0.25, 0.25, 0.25])
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 2)
        
        # Style header row
        for i in range(4):
            table[(0, i)].set_facecolor('#40466e')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        plt.tight_layout()
        
        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"Comparison plot saved to {output_file}")
        
        plt.show()


def main():
    parser = argparse.ArgumentParser(
        description='Analyze angle distributions from multiple UrQMD Oscar format files'
    )
    parser.add_argument('input_files', nargs='+', help='Path(s) to Oscar format file(s)')
    parser.add_argument('--format', choices=['auto', 'oscar1992a', 'osc1997a'], 
                       default='auto', help='Force specific format')
    parser.add_argument('--output', '-o', help='Output directory for results')
    parser.add_argument('--angle-type', '-a', choices=['polar', 'azimuthal', 'eta', 'all'],
                       default='all', help='Type of angle distribution to plot')
    parser.add_argument('--event', '-e', type=int, default=0,
                       help='Event number to analyze in each file')
    parser.add_argument('--single', '-s', action='store_true',
                       help='Plot single distribution instead of comprehensive view')
    parser.add_argument('--pt', action='store_true',
                       help='Plot transverse momentum distribution')
    parser.add_argument('--stats', action='store_true',
                       help='Print event statistics')
    parser.add_argument('--no-parallel', action='store_true',
                       help='Disable parallel processing')
    parser.add_argument('--workers', type=int, default=None,
                       help='Number of parallel workers (default: CPU count)')
    parser.add_argument('--compare', action='store_true',
                       help='Generate comparison plots for multiple files')
    parser.add_argument('--json', action='store_true',
                       help='Save results to JSON file')
    
    args = parser.parse_args()
    
    # Determine format
    if args.format == 'auto':
        format_type = None
    elif args.format == 'oscar1992a':
        format_type = OscarFormat.OSCAR1992A
    else:
        format_type = OscarFormat.OSC1997A
    
    # Handle single file vs multiple files
    if len(args.input_files) == 1:
        # Single file processing
        filepath = args.input_files[0]
        print(f"Reading Oscar format file: {filepath}")
        reader = OscarReader(filepath, format_type)
        detected_format = reader.format_type
        print(f"Detected format: {detected_format.value}")
        
        events = reader.read_file()
        
        if not events:
            print("No events found in file!")
            return
        
        print(f"Found {len(events)} events")
        
        # Select event
        if args.event >= len(events):
            print(f"Event {args.event} not found. Using first event.")
            event_idx = 0
        else:
            event_idx = args.event
        
        particles = events[event_idx]
        print(f"Event {event_idx}: {len(particles)} particles")
        
        # Print statistics if requested
        if args.stats:
            stats = EventStatistics(particles)
            stats.print_summary()
        
        # Create plotter
        plotter = AngleDistributionPlotter(particles)
        
        # Generate plots
        output_file = None
        if args.output:
            Path(args.output).mkdir(parents=True, exist_ok=True)
            output_file = str(Path(args.output) / f"{Path(filepath).stem}_plot.png")
        
        if args.pt:
            plotter.plot_pt_distribution(output_file)
        elif args.single:
            plotter.plot_single_distribution(args.angle_type, output_file)
        elif args.angle_type == 'all':
            plotter.plot_distributions(output_file)
        else:
            plotter.plot_single_distribution(args.angle_type, output_file)
    
    else:
        # Multi-file processing
        print(f"\n{'='*80}")
        print("MULTI-FILE PROCESSING MODE")
        print(f"{'='*80}")
        
        processor = MultiFileProcessor(
            use_parallel=not args.no_parallel,
            n_workers=args.workers
        )
        
        results = processor.process_files(args.input_files, args.event)
        processor.print_summary()
        
        # Create output directory
        if args.output:
            output_dir = Path(args.output)
            output_dir.mkdir(parents=True, exist_ok=True)
        else:
            output_dir = Path.cwd()
        
        # Save JSON results
        if args.json:
            json_file = str(output_dir / "results.json")
            processor.save_results_json(json_file)
        
        # Generate comparison plots
        if args.compare:
            comparison_file = str(output_dir / "comparison.png")
            processor.plot_comparison(comparison_file)


if __name__ == '__main__':
    main()
