import sys
from typing import Dict, List

import numpy as np

try:
    from models.cumulative_singnature import CumulativeSignature
except ImportError as e:
    print(f"Error importing modules: {e}")
    sys.exit(1)


class CumulativeAnalyzer:
    def __init__(self, threshold_strength: float = 0.05, threshold_confidence: float = 0.1):
        self.threshold_strength = threshold_strength
        self.threshold_confidence = threshold_confidence
        
        self.total_events_mod = 0
        self.total_events_unm = 0
        self.total_particles_mod = 0
        self.total_particles_unm = 0
        
        self.signatures = []
        
        self.multiplicity_mod = []
        self.multiplicity_unm = []
    
    def process_batch(self, batch_mod: List, batch_unm: List) -> None:
        if not batch_mod or not batch_unm:
            return
        
        for particles_mod, particles_unm in zip(batch_mod, batch_unm):
            self.total_events_mod += 1
            self.total_events_unm += 1
            
            n_mod = len(particles_mod) if particles_mod else 0
            n_unm = len(particles_unm) if particles_unm else 0
            
            self.total_particles_mod += n_mod
            self.total_particles_unm += n_unm
            
            self.multiplicity_mod.append(n_mod)
            self.multiplicity_unm.append(n_unm)
        
        self._detect_signatures_from_batch(batch_mod, batch_unm)
    
    def _detect_signatures_from_batch(self, batch_mod: List, batch_unm: List) -> None:
        if not batch_mod or not batch_unm:
            return
        
        self._detect_forbidden_kinematics(batch_mod, batch_unm)
    
    def _is_forbidden_kinematic(self, p) -> bool:
        """
        Check if particle is in forbidden kinematic region.
        For Au+Au @ 10 GeV, normal particles are forward-going.
        Fluctions scatter to extreme angles.
        """
        pt = (p.px**2 + p.py**2) ** 0.5
        p_total = (p.px**2 + p.py**2 + p.pz**2) ** 0.5
        
        # Very forward: high pz, very low pt (unphysical)
        if p.pz > 3.0 and pt < 0.1:
            return True
        
        # Very backward: large negative pz
        if p.pz < -3.0 and pt < 0.2:
            return True
        
        # High pT with very low total momentum (impossible)
        if pt > 2.0 and p_total < 0.5:
            return True
        
        # Extreme sideways: high pt with low pz
        if pt > 1.5 and abs(p.pz) < 0.1:
            return True
        
        return False
    
    def _detect_forbidden_kinematics(self, batch_mod: List, batch_unm: List) -> None:
        """
        Detect particles scattered to forbidden kinematic regions.
        Signature = MORE forbidden particles in modified than unmodified.
        """
        forbidden_mod = 0
        forbidden_unm = 0
        
        for event in batch_mod:
            if event:
                for p in event:
                    if self._is_forbidden_kinematic(p):
                        forbidden_mod += 1
        
        for event in batch_unm:
            if event:
                for p in event:
                    if self._is_forbidden_kinematic(p):
                        forbidden_unm += 1
        
        # Strength = excess of forbidden particles in modified
        if forbidden_unm > 0:
            excess = forbidden_mod - forbidden_unm
            strength = min(1.0, excess / max(forbidden_unm, 1))
        elif forbidden_mod > 0:
            strength = 1.0
        else:
            strength = 0.0
        
        confidence = strength * 0.9 if forbidden_mod > forbidden_unm else 0.0
        
        if strength > self.threshold_strength:
            sig = CumulativeSignature(
                signature_type="forbidden_kinematics",
                strength=float(strength),
                confidence=float(confidence),
                affected_particles=int(forbidden_mod - forbidden_unm),
                description=f"Forbidden region: mod={forbidden_mod}, unm={forbidden_unm}, excess={forbidden_mod - forbidden_unm}"
            )
            self.signatures.append(sig)
    
    def get_signatures(self) -> List[CumulativeSignature]:
        return self.signatures
    
    def get_statistics(self) -> Dict:
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
        self.total_events_mod = 0
        self.total_events_unm = 0
        self.total_particles_mod = 0
        self.total_particles_unm = 0
        self.signatures = []
        self.multiplicity_mod = []
        self.multiplicity_unm = []