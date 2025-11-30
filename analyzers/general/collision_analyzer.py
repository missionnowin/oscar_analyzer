# collision_analyzer.py
# Extra analyzers on top of angle_analyzer: pT/y/eta, PID spectra, correlations, v2, cumulative candidates.

import sys
from utils.readers import Particle
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict, Optional, Tuple

try:
    from analyzers.general.angle_analyzer import AngleAnalyzer
except ImportError as e:
    print(f"Error importing modules: {e}")
    sys.exit(1)

class CollisionAnalyzer:
    """High-level analyzer for a single event (list[Particle])."""

    def __init__(self, particles: List[Particle], system_label: str = "Au+Au @ 10 GeV"):
        self.particles = particles
        self.system_label = system_label
        self.angle = AngleAnalyzer()

        # Precompute kinematics as flat arrays
        self._compute_kinematics()

    def _compute_kinematics(self) -> None:
        px = np.array([p.px for p in self.particles])
        py = np.array([p.py for p in self.particles])
        pz = np.array([p.pz for p in self.particles])
        E  = np.array([p.E  for p in self.particles])

        self.px = px
        self.py = py
        self.pz = pz
        self.E  = E

        self.pt = np.sqrt(px**2 + py**2)
        self.p  = np.sqrt(px**2 + py**2 + pz**2)

        # Angles
        self.phi = np.array([self.angle.azimuthal_angle(p) for p in self.particles])
        self.theta = np.array([self.angle.polar_angle(p) for p in self.particles])
        self.phi_deg = np.degrees(self.phi)
        self.theta_deg = np.degrees(self.theta)

        # Rapidity y (careful with domain)
        self.y = np.zeros_like(self.pz)
        mask = (self.E > np.abs(self.pz)) & (self.E - self.pz > 0)
        self.y[mask] = 0.5 * np.log((self.E[mask] + self.pz[mask]) / (self.E[mask] - self.pz[mask]))

        # Pseudorapidity η
        # η = - ln(tan(theta/2))
        # Avoid singularities at theta≈0,π
        small = 1e-8
        self.eta = -np.log(np.tan(np.clip(self.theta, small, np.pi - small) / 2.0))

        # PDG codes if available (for OSC1997A)
        try:
            self.pdg = np.array([p.particle_id for p in self.particles], dtype=int)
        except Exception:
            self.pdg = None

    # ---------- Basic 1D histograms ----------

    def plot_rapidity(self, out: Optional[str] = None, y_range: Tuple[float, float] = (-2.0, 2.0)) -> None:
        plt.figure(figsize=(7, 5))
        mask = (self.y > -100) & (self.y < 100)
        plt.hist(self.y[mask], bins=40, range=y_range, color="steelblue", edgecolor="black", alpha=0.75)
        plt.xlabel("Rapidity y")
        plt.ylabel("dN/dy (arb.)")
        plt.title(f"Rapidity distribution\n{self.system_label}")
        plt.grid(alpha=0.3, axis="y")
        plt.tight_layout()
        if out:
            Path(out).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(out, dpi=300, bbox_inches="tight")
        plt.close()

    def plot_pseudorapidity(self, out: Optional[str] = None, eta_range: Tuple[float, float] = (-3.0, 3.0)) -> None:
        plt.figure(figsize=(7, 5))
        mask = np.isfinite(self.eta)
        plt.hist(self.eta[mask], bins=40, range=eta_range, color="darkgreen", edgecolor="black", alpha=0.75)
        plt.xlabel("Pseudorapidity η")
        plt.ylabel("dN/dη (arb.)")
        plt.title(f"Pseudorapidity distribution\n{self.system_label}")
        plt.grid(alpha=0.3, axis="y")
        plt.tight_layout()
        if out:
            Path(out).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(out, dpi=300, bbox_inches="tight")
        plt.close()

    def plot_pt(self, out: Optional[str] = None, pt_max: float = 3.0) -> None:
        plt.figure(figsize=(7, 5))
        mask = (self.pt >= 0) & (self.pt < pt_max)
        plt.hist(self.pt[mask], bins=50, range=(0, pt_max),
                 color="teal", edgecolor="black", alpha=0.75)
        plt.xlabel(r"$p_T$ (GeV)")
        plt.ylabel("dN/dpT (arb.)")
        plt.title(f"Transverse momentum spectrum\n{self.system_label}")
        plt.yscale("log")
        plt.grid(alpha=0.3, axis="y")
        plt.tight_layout()
        if out:
            Path(out).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(out, dpi=300, bbox_inches="tight")
        plt.close()

    # ---------- Multiplicity / centrality proxy ----------

    def multiplicity_charged_midrap(self, eta_cut: float = 1.0) -> Dict[str, float]:
        if self.pdg is None:
            return {"Nch": float(len(self.pt))}
        # crude charged selection: exclude neutrons, lambdas etc.
        charged_mask = np.isin(np.abs(self.pdg), [211, 321, 2212])  # π, K, p
        eta_mask = np.abs(self.eta) < eta_cut
        mask = charged_mask & eta_mask
        Nch = int(mask.sum())
        return {"Nch": float(Nch)}

    # ---------- Identified-particle spectra ----------

    def _pid_mask(self, species: str) -> np.ndarray:
        if self.pdg is None:
            return np.ones_like(self.pt, dtype=bool)

        species = species.lower()
        if species == "pi+":
            return self.pdg == 211
        if species == "pi-":
            return self.pdg == -211
        if species == "p":
            return self.pdg == 2212
        if species == "pbar":
            return self.pdg == -2212
        if species == "k+":
            return self.pdg == 321
        if species == "k-":
            return self.pdg == -321
        # default: all
        return np.ones_like(self.pt, dtype=bool)

    def plot_pid_pt(self, species: str, out: Optional[str] = None,
                    pt_max: float = 3.0, y_window: Tuple[float, float] = (-0.5, 0.5)) -> None:
        mask = self._pid_mask(species)
        mask &= (self.y >= y_window[0]) & (self.y <= y_window[1])
        if not mask.any():
            return
        plt.figure(figsize=(7, 5))
        plt.hist(self.pt[mask], bins=40, range=(0, pt_max),
                 color="steelblue", edgecolor="black", alpha=0.8)
        plt.xlabel(r"$p_T$ (GeV)")
        plt.ylabel("dN/dpT (arb.)")
        plt.yscale("log")
        plt.title(f"{species} pT spectrum, {y_window[0]}<y<{y_window[1]}\n{self.system_label}")
        plt.grid(alpha=0.3, axis="y")
        plt.tight_layout()
        if out:
            Path(out).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(out, dpi=300, bbox_inches="tight")
        plt.close()

    def plot_pid_rapidity(self, species: str, out: Optional[str] = None,
                          y_range: Tuple[float, float] = (-2.0, 2.0)) -> None:
        mask = self._pid_mask(species)
        if not mask.any():
            return
        plt.figure(figsize=(7, 5))
        plt.hist(self.y[mask], bins=40, range=y_range,
                 color="darkorange", edgecolor="black", alpha=0.8)
        plt.xlabel("Rapidity y")
        plt.ylabel("dN/dy (arb.)")
        plt.title(f"{species} rapidity distribution\n{self.system_label}")
        plt.grid(alpha=0.3, axis="y")
        plt.tight_layout()
        if out:
            Path(out).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(out, dpi=300, bbox_inches="tight")
        plt.close()

    # ---------- 2D correlations ----------

    def plot_y_pt(self, out: Optional[str] = None,
                  y_range: Tuple[float, float] = (-2.0, 2.0),
                  pt_max: float = 3.0) -> None:
        plt.figure(figsize=(7, 5))
        y = np.clip(self.y, y_range[0], y_range[1])
        pt = np.clip(self.pt, 0, pt_max)
        h, xedges, yedges, im = plt.hist2d(y, pt, bins=[40, 40],
                                           range=[[y_range[0], y_range[1]], [0, pt_max]],
                                           cmap="viridis")
        plt.xlabel("y")
        plt.ylabel(r"$p_T$ (GeV)")
        plt.title(f"2D correlation: y vs pT\n{self.system_label}")
        cbar = plt.colorbar(im)
        cbar.set_label("Counts")
        plt.tight_layout()
        if out:
            Path(out).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(out, dpi=300, bbox_inches="tight")
        plt.close()

    def plot_eta_phi(self, out: Optional[str] = None,
                     eta_range: Tuple[float, float] = (-3.0, 3.0)) -> None:
        plt.figure(figsize=(7, 5))
        eta = np.clip(self.eta, eta_range[0], eta_range[1])
        phi = self.phi
        h, xedges, yedges, im = plt.hist2d(eta, phi, bins=[40, 40],
                                           range=[[eta_range[0], eta_range[1]], [0, 2*np.pi]],
                                           cmap="plasma")
        plt.xlabel("η")
        plt.ylabel("ϕ (rad)")
        plt.title(f"2D correlation: η vs ϕ\n{self.system_label}")
        cbar = plt.colorbar(im)
        cbar.set_label("Counts")
        plt.tight_layout()
        if out:
            Path(out).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(out, dpi=300, bbox_inches="tight")
        plt.close()

    # ---------- Very basic v2 estimator ----------

    def estimate_v2(self, pt_bins: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
        if pt_bins is None:
            pt_bins = np.linspace(0, 3.0, 7)  # 0–3 GeV, 6 bins

        Qx = np.sum(self.pt * np.cos(2 * self.phi))
        Qy = np.sum(self.pt * np.sin(2 * self.phi))
        psi2 = 0.5 * np.arctan2(Qy, Qx)

        v2_vals = []
        pt_centers = []
        for i in range(len(pt_bins) - 1):
            lo, hi = pt_bins[i], pt_bins[i+1]
            mask = (self.pt >= lo) & (self.pt < hi)
            if not mask.any():
                v2_vals.append(0.0)
                pt_centers.append(0.5 * (lo + hi))
                continue
            v2_bin = np.mean(np.cos(2 * (self.phi[mask] - psi2)))
            v2_vals.append(v2_bin)
            pt_centers.append(0.5 * (lo + hi))
        return {
            "pt_centers": np.array(pt_centers),
            "v2": np.array(v2_vals),
            "psi2": psi2,
        }

    def plot_v2(self, out: Optional[str] = None) -> None:
        res = self.estimate_v2()
        plt.figure(figsize=(7, 5))
        plt.plot(res["pt_centers"], res["v2"], "o-", color="purple")
        plt.axhline(0, color="black", linewidth=1)
        plt.xlabel(r"$p_T$ (GeV)")
        plt.ylabel(r"$v_2(p_T)$")
        plt.title(f"Basic elliptic flow estimate v2(pT)\n{self.system_label}")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        if out:
            Path(out).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(out, dpi=300, bbox_inches="tight")
        plt.close()

    # ---------- Cumulative candidate tagging ----------

    def cumulative_candidate_mask(self,
                                  pt_min: float = 1.0,
                                  y_window: Tuple[float, float] = (-0.5, 0.5)) -> np.ndarray:
        """Simple cumulative-like cut: high-pT particles in midrapidity."""
        mask = (self.pt >= pt_min) & (self.y >= y_window[0]) & (self.y <= y_window[1])
        return mask

    def plot_cumulative_candidates(self, out: Optional[str] = None,
                                   pt_min: float = 1.0,
                                   y_window: Tuple[float, float] = (-0.5, 0.5)) -> None:
        mask = self.cumulative_candidate_mask(pt_min, y_window)
        if not mask.any():
            return
        plt.figure(figsize=(7, 5))
        plt.scatter(self.y[mask], self.pt[mask], s=20, edgecolor="black",
                    facecolor="red", alpha=0.7)
        plt.xlabel("y")
        plt.ylabel(r"$p_T$ (GeV)")
        plt.title(f"Cumulative candidates (pT>{pt_min} GeV, {y_window[0]}<y<{y_window[1]})\n{self.system_label}")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        if out:
            Path(out).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(out, dpi=300, bbox_inches="tight")
        plt.close()