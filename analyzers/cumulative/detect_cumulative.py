#!/usr/bin/env python3
# detect_cumulative.py
# Integration script: Use cumulative detector with main analyzer

"""
Usage examples:

# 1. Compare modified vs unmodified files:
python detect_cumulative.py --modified modified_event.f19 --unmodified unmodified_event.f19

# 2. Analyze single modified file for cumulative signatures:
python detect_cumulative.py --modified modified_event.f19

# 3. Full analysis with visualizations:
python detect_cumulative.py --modified mod.f19 --unmodified unmod.f19 --compare --plot all

# 4. Batch process multiple pairs:
python detect_cumulative.py --batch pairs_list.txt --plot signatures

# 5. Generate detailed report:
python detect_cumulative.py --modified mod.f19 --unmodified unmod.f19 --report results.json
"""

import argparse
from pathlib import Path
import json
from typing import List, Dict
import sys


try:
    from cumulative_detector import (
        CumulativeEffectDetector,
        CumulativeComparisonVisualizer,
        compare_files
    )
except ImportError:
    print("Error: cumulative_detector.py not found in same directory")
    sys.exit(1)

# Import from main analyzer
try:
    from utils.readers import OscarReader
except ImportError:
    print("Error: angle_analyzer.py not found in same directory")
    sys.exit(1)


class CumulativeEffectAnalysis:
    """Orchestrate cumulative effect analysis workflow"""
    
    def __init__(self, output_dir: str = "cumulative_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results = {}
    
    def analyze_pair(self, modified_file: str, unmodified_file: str = None, 
                     event_idx: int = 0) -> Dict:
        """Analyze a modified/unmodified file pair"""
        
        print(f"\n{'='*80}")
        print(f"CUMULATIVE EFFECT ANALYSIS: {Path(modified_file).name}")
        print(f"{'='*80}")
        
        # Read files
        try:
            reader_mod = OscarReader(modified_file)
            events_mod = reader_mod.read_file()
            
            if not events_mod or event_idx >= len(events_mod):
                print(f"Error: Event {event_idx} not found in {modified_file}")
                return None
            
            particles_mod = events_mod[event_idx]
            print(f"Modified: {len(events_mod)} events, analyzing event {event_idx} "
                  f"with {len(particles_mod)} particles")
            
            particles_unmod = None
            if unmodified_file:
                reader_unmod = OscarReader(unmodified_file)
                events_unmod = reader_unmod.read_file()
                
                if events_unmod and event_idx < len(events_unmod):
                    particles_unmod = events_unmod[event_idx]
                    print(f"Unmodified: {len(events_unmod)} events, analyzing event {event_idx} "
                          f"with {len(particles_unmod)} particles")
            
            # Create detector
            detector = CumulativeEffectDetector(particles_mod, particles_unmod)
            detector.detect_all_signatures()
            detector.print_report()
            
            # Store results
            result = {
                'modified_file': modified_file,
                'unmodified_file': unmodified_file,
                'event_idx': event_idx,
                'n_particles_mod': len(particles_mod),
                'n_particles_unmod': len(particles_unmod) if particles_unmod else None,
                'cumulative_likelihood': detector.get_cumulative_likelihood(),
                'signatures': []
            }
            
            for sig in detector.signatures:
                result['signatures'].append({
                    'type': sig.signature_type,
                    'strength': sig.strength,
                    'confidence': sig.confidence,
                    'affected_particles': sig.affected_particles,
                    'description': sig.description
                })
            
            return {
                'detector': detector,
                'particles_mod': particles_mod,
                'particles_unmod': particles_unmod,
                'result': result
            }
            
        except Exception as e:
            print(f"Error analyzing files: {e}")
            return None
    
    def create_visualizations(self, analysis: Dict, plot_types: List[str] = ['all']):
        """Generate comparison visualizations"""
        
        detector = analysis['detector']
        visualizer = CumulativeComparisonVisualizer(detector)
        
        if 'all' in plot_types or 'pt' in plot_types:
            pt_file = str(self.output_dir / "pt_comparison.png")
            visualizer.plot_pt_comparison(pt_file)
        
        if 'all' in plot_types or 'angles' in plot_types:
            angle_file = str(self.output_dir / "angle_comparison.png")
            visualizer.plot_angle_comparison(angle_file)
        
        if 'all' in plot_types or 'signatures' in plot_types:
            sig_file = str(self.output_dir / "signatures.png")
            visualizer.plot_signatures(sig_file)
    
    def save_results(self, analysis: Dict, output_file: str = "cumulative_analysis.json"):
        """Save analysis results to JSON"""
        
        result = analysis['result']
        output_path = self.output_dir / output_file
        
        with open(output_path, 'w') as f:
            json.dump(result, f, indent=2)
        
        print(f"\nResults saved to {output_path}")
        return output_path
    
    def process_batch(self, batch_file: str, plot_types: List[str] = None):
        """Process batch of modified/unmodified file pairs
        
        batch_file format:
        modified1.f19 unmodified1.f19
        modified2.f19 unmodified2.f19
        modified3.f19
        """
        
        print(f"\nProcessing batch from: {batch_file}")
        
        batch_results = []
        pair_num = 0
        
        with open(batch_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                
                files = line.split()
                if not files:
                    continue
                
                pair_num += 1
                modified_file = files[0]
                unmodified_file = files[1] if len(files) > 1 else None
                
                analysis = self.analyze_pair(modified_file, unmodified_file)
                
                if analysis:
                    batch_results.append(analysis)
                    
                    # Generate plots if requested
                    if plot_types:
                        self.create_visualizations(analysis, plot_types)
                    
                    # Save individual results
                    self.save_results(analysis, f"pair_{pair_num}_analysis.json")
        
        # Generate batch summary
        self.create_batch_summary(batch_results)
        
        return batch_results
    
    def create_batch_summary(self, analyses: List[Dict]):
        """Create summary of batch analysis"""
        
        print(f"\n{'='*80}")
        print("BATCH ANALYSIS SUMMARY")
        print(f"{'='*80}\n")
        
        summary = {
            'total_pairs': len(analyses),
            'pairs': []
        }
        
        for i, analysis in enumerate(analyses, 1):
            result = analysis['result']
            summary['pairs'].append({
                'pair_number': i,
                'modified_file': result['modified_file'],
                'unmodified_file': result['unmodified_file'],
                'cumulative_likelihood': result['cumulative_likelihood'],
                'n_signatures': len(result['signatures']),
                'signature_types': [s['type'] for s in result['signatures']]
            })
            
            print(f"Pair {i}:")
            print(f"  Modified: {Path(result['modified_file']).name}")
            if result['unmodified_file']:
                print(f"  Unmodified: {Path(result['unmodified_file']).name}")
            print(f"  Cumulative likelihood: {result['cumulative_likelihood']:.2f}")
            print(f"  Signatures found: {len(result['signatures'])}")
            if result['signatures']:
                for sig in result['signatures']:
                    print(f"    - {sig['type']}: strength={sig['strength']:.2f}, "
                          f"confidence={sig['confidence']:.2f}")
            print()
        
        # Save batch summary
        summary_file = self.output_dir / "batch_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"Batch summary saved to {summary_file}\n")
        print("="*80 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description='Detect cumulative scattering effects in UrQMD collision data'
    )
    
    # File inputs
    parser.add_argument('--modified', '-m', help='Path to file with cumulative effects')
    parser.add_argument('--unmodified', '-u', help='Path to baseline file (optional)')
    parser.add_argument('--batch', '-b', help='Batch file with file pairs (one per line)')
    parser.add_argument('--event', '-e', type=int, default=0,
                       help='Event index to analyze')
    
    # Output options
    parser.add_argument('--output', '-o', default='cumulative_results',
                       help='Output directory (default: cumulative_results)')
    parser.add_argument('--plot', nargs='+', 
                       choices=['all', 'pt', 'angles', 'signatures'],
                       help='Generate comparison plots')
    parser.add_argument('--report', help='Save detailed report to JSON file')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Print detailed output')
    
    args = parser.parse_args()
    
    # Create analysis instance
    analysis_engine = CumulativeEffectAnalysis(args.output)
    
    # Process batch or single pair
    if args.batch:
        # Batch processing
        plot_types = args.plot if args.plot else ['signatures']
        analyses = analysis_engine.process_batch(args.batch, plot_types)
        
        # Save combined report
        if args.report:
            report_file = Path(args.output) / args.report
            all_results = [a['result'] for a in analyses]
            with open(report_file, 'w') as f:
                json.dump(all_results, f, indent=2)
            print(f"Combined report saved to {report_file}")
    
    elif args.modified:
        # Single pair analysis
        analysis = analysis_engine.analyze_pair(args.modified, args.unmodified, args.event)
        
        if analysis:
            # Generate plots
            plot_types = args.plot if args.plot else ['signatures']
            analysis_engine.create_visualizations(analysis, plot_types)
            
            # Save report
            report_file = args.report if args.report else 'cumulative_analysis.json'
            analysis_engine.save_results(analysis, report_file)
    
    else:
        parser.print_help()
        print("\nExamples:")
        print("  # Compare two files:")
        print("  python detect_cumulative.py --modified mod.f19 --unmodified unmod.f19 --plot all")
        print("\n  # Analyze single file:")
        print("  python detect_cumulative.py --modified mod.f19 --plot signatures")
        print("\n  # Batch process:")
        print("  python detect_cumulative.py --batch pairs.txt --plot all")


if __name__ == '__main__':
    main()
