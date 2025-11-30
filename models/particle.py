from dataclasses import dataclass
from typing import Optional
import math


@dataclass
class Particle:
    """Represent a single particle from Oscar format"""
    particle_id: int
    px: float
    py: float
    pz: float
    mass: float
    x: float
    y: float
    z: float
    E: float
    t_formed: float
    charge: Optional[int] = None
    baryon_density: Optional[float] = None

    @property
    def pt(self) -> float:
        # transverse momentum
        return math.hypot(self.px, self.py)

    @property
    def eta(self) -> Optional[float]:
        # pseudorapidity from momentum
        p2 = self.px * self.px + self.py * self.py + self.pz * self.pz
        if p2 == 0.0:
            return None
        p = math.sqrt(p2)
        # avoid division by zero when p == |pz|
        denom = p - abs(self.pz)
        if denom <= 1e-12:
            return None
        # standard eta definition: 0.5 * ln((p+pz)/(p-pz))
        try:
            return 0.5 * math.log((p + self.pz) / (p - self.pz))
        except (ZeroDivisionError, ValueError):
            return None
