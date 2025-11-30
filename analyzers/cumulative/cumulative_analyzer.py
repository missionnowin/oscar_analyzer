import sys
from typing import Dict, List

import numpy as np

try:
    from models.cumulative_singnature import CumulativeSignature
except ImportError as e:
    print(f"Error importing modules: {e}")
    sys.exit(1)


class CumulativeAnalyzer:
    """
    Streaming cumulative effect detector.
    Processes event batches without storing particles in memory.
    """
    
    def __init__(self, threshold_strength: float = 0.5, threshold_confidence: float = 0.7):
        """
        Initialize cumulative analyzer.
        
        Args:
            threshold_strength: Minimum strength for signature detection (0-1)
            threshold_confidence: Minimum confidence for signature detection (0-1)
        """
        self.threshold_strength = threshold_strength
        self.threshold_confidence = threshold_confidence
        
        # Accumulate statistics across batches
        self.total_events_mod = 0
        self.total_events_unm = 0
        self.total_particles_mod = 0
        self.total_particles_unm = 0
        
        # Track signatures found
        self.signatures = []
        
        # Accumulate distributions for comparison
        self.pt_dist_mod = []
        self.pt_dist_unm = []
        self.eta_dist_mod = []
        self.eta_dist_unm = []
        self.multiplicity_mod = []
        self.multiplicity_unm = []
        
    def process_batch(self, batch_mod: List, batch_unm: List) -> None:
        """
        Process a batch of events from both modified and unmodified files.
        
        Args:
            batch_mod: List of modified particle lists (one per event)
            batch_unm: List of unmodified particle lists (one per event)
        """
        if not batch_mod or not batch_unm:
            return
        
        # Process events pairwise
        for particles_mod, particles_unm in zip(batch_mod, batch_unm):
            self.total_events_mod += 1
            self.total_events_unm += 1
            
            # Count particles
            n_mod = len(particles_mod) if particles_mod else 0
            n_unm = len(particles_unm) if particles_unm else 0
            
            self.total_particles_mod += n_mod
            self.total_particles_unm += n_unm
            
            # Track multiplicity
            self.multiplicity_mod.append(n_mod)
            self.multiplicity_unm.append(n_unm)
            
            # Extract kinematics (if particles have these attributes)
            if particles_mod:
                try:
                    for p in particles_mod:
                        if hasattr(p, 'pt'):
                            self.pt_dist_mod.append(p.pt)
                        if hasattr(p, 'eta'):
                            self.eta_dist_mod.append(p.eta)
                except:
                    pass
            
            if particles_unm:
                try:
                    for p in particles_unm:
                        if hasattr(p, 'pt'):
                            self.pt_dist_unm.append(p.pt)
                        if hasattr(p, 'eta'):
                            self.eta_dist_unm.append(p.eta)
                except:
                    pass
        
        # Detect cumulative signatures from this batch
        self._detect_signatures_from_batch(batch_mod, batch_unm)
    
    def _detect_signatures_from_batch(self, batch_mod: List, batch_unm: List) -> None:
        """
        Detect cumulative effect signatures in current batch.
        Updates self.signatures list.
        """
        
        if not batch_mod or not batch_unm:
            return
        
        # Signature 1: Multiplicity Difference
        self._detect_multiplicity_signature(batch_mod, batch_unm)
        
        # Signature 2: Kinematic Differences
        self._detect_kinematic_signature(batch_mod, batch_unm)
        
        # Signature 3: Particle Type Differences
        self._detect_composition_signature(batch_mod, batch_unm)
    
    def _detect_multiplicity_signature(self, batch_mod: List, batch_unm: List) -> None:
        """Detect cumulative multiplicity differences."""
        
        multiplicity_mod = np.array([len(p) if p else 0 for p in batch_mod])
        multiplicity_unm = np.array([len(p) if p else 0 for p in batch_unm])
        
        mean_diff = np.abs(np.mean(multiplicity_mod) - np.mean(multiplicity_unm))
        std_mod = np.std(multiplicity_mod) if len(multiplicity_mod) > 1 else 1.0
        std_unm = np.std(multiplicity_unm) if len(multiplicity_unm) > 1 else 1.0
        
        # Calculate strength and confidence
        combined_std = np.sqrt(std_mod**2 + std_unm**2 + 1e-6)
        strength = min(1.0, mean_diff / (combined_std + 1e-6))
        confidence = min(1.0, (mean_diff / (combined_std + 1e-6)) * 0.8)
        
        if strength >= self.threshold_strength and confidence >= self.threshold_confidence:
            affected = int(np.sum(multiplicity_mod != multiplicity_unm))
            sig = CumulativeSignature(
                signature_type="multiplicity_change",
                strength=float(strength),
                confidence=float(confidence),
                affected_particles=affected,
                description=f"Mean multiplicity: mod={multiplicity_mod.mean():.1f}, unm={multiplicity_unm.mean():.1f}"
            )
            self.signatures.append(sig)
    
    def _detect_kinematic_signature(self, batch_mod: List, batch_unm: List) -> None:
        """Detect cumulative kinematic differences."""
        
        # Extract pT and eta if available
        pt_mod = []
        pt_unm = []
        
        for event in batch_mod:
            if event:
                for p in event:
                    if hasattr(p, 'pt'):
                        pt_mod.append(p.pt)
        
        for event in batch_unm:
            if event:
                for p in event:
                    if hasattr(p, 'pt'):
                        pt_unm.append(p.pt)
        
        if len(pt_mod) > 0 and len(pt_unm) > 0:
            pt_mod_arr = np.array(pt_mod)
            pt_unm_arr = np.array(pt_unm)
            
            mean_pt_mod = np.mean(pt_mod_arr)
            mean_pt_unm = np.mean(pt_unm_arr)
            mean_diff = np.abs(mean_pt_mod - mean_pt_unm)
            
            # Strength based on pT difference
            strength = min(1.0, mean_diff / (mean_pt_mod + mean_pt_unm + 1e-6))
            confidence = min(1.0, strength * 0.85)
            
            if strength >= self.threshold_strength and confidence >= self.threshold_confidence:
                sig = CumulativeSignature(
                    signature_type="kinematic_change",
                    strength=float(strength),
                    confidence=float(confidence),
                    affected_particles=len(pt_mod),
                    description=f"Mean pT: mod={mean_pt_mod:.2f}, unm={mean_pt_unm:.2f}"
                )
                self.signatures.append(sig)
    
    def _detect_composition_signature(self, batch_mod: List, batch_unm: List) -> None:
        """Detect cumulative particle composition changes."""
        
        # Count particle types if available
        types_mod = {}
        types_unm = {}
        
        for event in batch_mod:
            if event:
                for p in event:
                    if hasattr(p, 'pdg'):
                        pid = p.pdg
                        types_mod[pid] = types_mod.get(pid, 0) + 1
        
        for event in batch_unm:
            if event:
                for p in event:
                    if hasattr(p, 'pdg'):
                        pid = p.pdg
                        types_unm[pid] = types_unm.get(pid, 0) + 1
        
        if types_mod and types_unm:
            # Compare distributions
            all_types = set(types_mod.keys()) | set(types_unm.keys())
            differences = 0
            
            for pid in all_types:
                count_mod = types_mod.get(pid, 0)
                count_unm = types_unm.get(pid, 0)
                if count_mod != count_unm:
                    differences += abs(count_mod - count_unm)
            
            total = sum(types_mod.values()) + sum(types_unm.values())
            strength = min(1.0, differences / (total + 1e-6))
            confidence = min(1.0, strength * 0.8)
            
            if strength >= self.threshold_strength and confidence >= self.threshold_confidence:
                sig = CumulativeSignature(
                    signature_type="composition_change",
                    strength=float(strength),
                    confidence=float(confidence),
                    affected_particles=differences,
                    description=f"Particle composition change detected ({len(all_types)} types)"
                )
                self.signatures.append(sig)
    
    def get_signatures(self) -> List[CumulativeSignature]:
        """
        Get all detected signatures.
        
        Returns:
            List of CumulativeSignature objects
        """
        return self.signatures
    
    def get_statistics(self) -> Dict:
        """
        Get cumulative analysis statistics.
        
        Returns:
            Dictionary of statistics
        """
        return {
            'total_events_modified': int(self.total_events_mod),
            'total_events_unmodified': int(self.total_events_unm),
            'total_particles_modified': int(self.total_particles_mod),
            'total_particles_unmodified': int(self.total_particles_unm),
            'signatures_detected': len(self.signatures),
            'avg_multiplicity_modified': self.total_particles_mod / max(self.total_events_mod, 1),
            'avg_multiplicity_unmodified': self.total_particles_unm / max(self.total_events_unm, 1),
        }
    
    def reset(self) -> None:
        """Reset analyzer for new analysis."""
        self.total_events_mod = 0
        self.total_events_unm = 0
        self.total_particles_mod = 0
        self.total_particles_unm = 0
        self.signatures = []
        self.pt_dist_mod = []
        self.pt_dist_unm = []
        self.eta_dist_mod = []
        self.eta_dist_unm = []
        self.multiplicity_mod = []
        self.multiplicity_unm = []