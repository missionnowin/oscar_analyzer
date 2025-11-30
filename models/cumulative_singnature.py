from dataclasses import dataclass


@dataclass
class CumulativeSignature:
    """Represents detected cumulative effect signature"""
    signature_type: str
    strength: float      # 0-1, measure of effect strength
    confidence: float    # 0-1, statistical confidence
    affected_particles: int
    description: str