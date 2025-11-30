from dataclasses import dataclass
from typing import Optional


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
