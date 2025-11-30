#!/usr/bin/env python3
# verify_data_integrity.py
# Check if modified and unmodified files are actually different

import numpy as np
import sys

try:
    from utils.readers import OscarReader
except ImportError:
    print("Error: Cannot import OscarReader")
    sys.exit(1)


def compare_file_pair(mod_file: str, unm_file: str, n_events: int = 5):
    """Compare modified vs unmodified files event-by-event."""
    
    print(f"\n{'='*80}")
    print(f"DATA INTEGRITY CHECK")
    print(f"{'='*80}\n")
    
    print(f"Modified file:   {mod_file}")
    print(f"Unmodified file: {unm_file}\n")
    
    # Read both files
    reader_mod = OscarReader(mod_file)
    reader_unm = OscarReader(unm_file)
    
    events_mod = reader_mod.read_file()
    events_unm = reader_unm.read_file()
    
    if not events_mod or not events_unm:
        print("ERROR: Could not read files!")
        return
    
    print(f"Modified file:   {len(events_mod):,d} events")
    print(f"Unmodified file: {len(events_unm):,d} events\n")
    
    # Compare first few events
    print(f"Comparing first {min(n_events, len(events_mod), len(events_unm))} events:\n")
    
    identical_count = 0
    different_count = 0
    
    for evt_idx in range(min(n_events, len(events_mod), len(events_unm))):
        evt_mod = events_mod[evt_idx]
        evt_unm = events_unm[evt_idx]
        
        n_mod = len(evt_mod)
        n_unm = len(evt_unm)
        
        # Check if events are identical
        if n_mod == n_unm:
            # Compare particle-by-particle
            identical_particles = 0
            for p_mod, p_unm in zip(evt_mod, evt_unm):
                if (p_mod.px == p_unm.px and 
                    p_mod.py == p_unm.py and 
                    p_mod.pz == p_unm.pz and
                    p_mod.particle_id == p_unm.particle_id):
                    identical_particles += 1
            
            if identical_particles == n_mod:
                print(f"Event {evt_idx}: ✗ IDENTICAL ({n_mod} particles)")
                identical_count += 1
            else:
                print(f"Event {evt_idx}: ✓ Different ({identical_particles}/{n_mod} particles match)")
                different_count += 1
        else:
            print(f"Event {evt_idx}: ✓ Different multiplicity (mod: {n_mod}, unm: {n_unm})")
            different_count += 1
        
        # Show sample particles from first event
        if evt_idx == 0:
            print(f"\n  First 3 particles (Modified):")
            for i, p in enumerate(evt_mod[:3]):
                print(f"    {i}: px={p.px:7.3f} py={p.py:7.3f} pz={p.pz:7.3f} pdgid={p.particle_id}")
            
            print(f"\n  First 3 particles (Unmodified):")
            for i, p in enumerate(evt_unm[:3]):
                print(f"    {i}: px={p.px:7.3f} py={p.py:7.3f} pz={p.pz:7.3f} pdgid={p.particle_id}")
            print()
    
    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY:")
    print(f"{'='*80}")
    print(f"Identical events:  {identical_count}")
    print(f"Different events:  {different_count}")
    
    if identical_count == n_events:
        print("\n❌ CRITICAL: Modified and unmodified files are IDENTICAL!")
        print("   Check your generator/data pipeline!")
    elif different_count > 0:
        print("\n✓ Files appear to be different (as expected)")
    
    # Statistical comparison
    print(f"\n{'='*80}")
    print("STATISTICAL COMPARISON (all events):")
    print(f"{'='*80}\n")
    
    # Calculate pT for all particles
    def calc_pt_stats(events):
        pt_all = []
        for evt in events:
            for p in evt:
                pt = np.sqrt(p.px**2 + p.py**2)
                pt_all.append(pt)
        pt_array = np.array(pt_all)
        return {
            'mean': np.mean(pt_array),
            'std': np.std(pt_array),
            'n_particles': len(pt_array)
        }
    
    stats_mod = calc_pt_stats(events_mod)
    stats_unm = calc_pt_stats(events_unm)
    
    print(f"Modified:")
    print(f"  Total particles: {stats_mod['n_particles']:,d}")
    print(f"  pT mean:  {stats_mod['mean']:.4f} GeV")
    print(f"  pT std:   {stats_mod['std']:.4f} GeV")
    
    print(f"\nUnmodified:")
    print(f"  Total particles: {stats_unm['n_particles']:,d}")
    print(f"  pT mean:  {stats_unm['mean']:.4f} GeV")
    print(f"  pT std:   {stats_unm['std']:.4f} GeV")
    
    print(f"\nDifference:")
    print(f"  ΔpT_mean: {abs(stats_mod['mean'] - stats_unm['mean']):.6f} GeV "
          f"({abs(stats_mod['mean'] - stats_unm['mean'])/stats_unm['mean']*100:.2f}%)")
    print(f"  ΔpT_std:  {abs(stats_mod['std'] - stats_unm['std']):.6f} GeV "
          f"({abs(stats_mod['std'] - stats_unm['std'])/stats_unm['std']*100:.2f}%)")
    
    if abs(stats_mod['mean'] - stats_unm['mean']) / stats_unm['mean'] < 0.001:
        print("\n  ⚠️  Statistics are IDENTICAL (< 0.1% difference)")
        print("      This suggests files are the same or generator has no effect")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Verify data integrity')
    parser.add_argument('--mod', '-m', required=True, help='Modified file')
    parser.add_argument('--unm', '-u', required=True, help='Unmodified file')
    parser.add_argument('--events', '-e', type=int, default=5,
                       help='Number of events to compare (default: 5)')
    
    args = parser.parse_args()
    
    compare_file_pair(args.mod, args.unm, args.events)


if __name__ == '__main__':
    main()
