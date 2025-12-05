from enum import Enum


class FileFormat(Enum):
    """Supported file formats for collision system detection."""
    URQMD = "urqmd"          # Oscar1992A/1997A format
    PHQMD = "phqmd"          # PHQMD generator output
    HEPMC2 = "hepmc2"        # HepMC2 text format
    HEPMC3 = "hepmc3"        # HepMC3 text format
    UNKNOWN = "unknown"


FORMAT_SIGNATURES = {
    FileFormat.URQMD: [
        b'UrQMD',
        b'Oscar',
        b'final_id',
    ],
    FileFormat.PHQMD: [
        b'PHQMD',
        b'projectile',
        b'target',
    ],
    FileFormat.HEPMC2: [
        b'HepMC::Version',
        b'HepMC::IO',
        b'E ',  # Event line
    ],
    FileFormat.HEPMC3: [
        b'HepMC3',
        b'# PYTHIA EVENT FILE',
    ],
}