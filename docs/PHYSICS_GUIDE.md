# Physics Interpretation Guide

Understanding the cumulative scattering signatures.

## Overview

This analysis quantifies how particle interactions modify momentum distributions. When a projectile nucleus passes through a target nucleus, particles undergo multiple scattering. This pipeline detects and measures these cumulative effects.

---

## Primary Observable: pT Broadening

### What It Measures

The ratio of transverse momentum widths:

```
Ratio = σ_pT(modified) / σ_pT(unmodified)
```

### Physical Meaning

**Multiple scattering kicks particles perpendicular to their original direction.**

- **Ratio > 1.05**: Multiple scattering detected (≥5% broadening)
- **Ratio ≈ 1.00**: No significant scattering
- **Ratio < 0.95**: Unusual (possible energy loss or detector effects)

### Example

```
Modified dataset:   σ_pT = 0.340 GeV
Unmodified dataset: σ_pT = 0.315 GeV
Ratio: 0.340 / 0.315 = 1.079

Interpretation: 7.9% broadening signature
     ✓ Multiple scattering confirmed
```

### Physical Origin

1. **Initial state**: Particle with small transverse momentum pT ≈ 0
2. **Interaction**: Multiple small-angle scatterings with medium
3. **Final state**: Accumulated transverse momentum, pT > 0

For N independent scatterings with mean deflection θ:
```
σ_pT ~ √N × θ
```

Larger σ_pT indicates either:
- More scattering events (N↑)
- Larger deflection angles (θ↑)
- Denser medium (both effects)

---

## Secondary Observable: Mean pT Shift

### What It Measures

Change in average transverse momentum:

```
Shift = ⟨pT⟩_modified - ⟨pT⟩_unmodified
```

### Physical Meaning

**Two competing effects**:

1. **Energy Loss** (negative shift):
   - Particles radiate energy in medium
   - Ionization and bremsstrahlung
   - Common in dense QGP

2. **Multiple Scattering** (positive shift):
   - Random walk → net positive pT
   - Accumulation of deflections
   - Non-linear effects in dense matter

### Example

```
Modified:   ⟨pT⟩ = 0.452 GeV
Unmodified: ⟨pT⟩ = 0.451 GeV
Shift: +0.001 GeV (+0.2%)

Interpretation: Slight energy gain (scattering dominated)
```

### Clinical Interpretation

| Shift | Effect | System |
|-------|--------|--------|
| +5% | Scattering dominated | Low density, strong deflections |
| 0% | Balanced | Medium density |
| -5% | Energy loss dominated | High density, radiative losses |

---

## Tertiary Observable: Distribution Extent

### What It Measures

Maximum pT ratio:

```
Extent = pT_max(modified) / pT_max(unmodified)
```

### Physical Meaning

**Hard scattering produces extreme particles.**

- **Extent > 1.0**: Spectrum extends higher
  - Multiple scattering creates high-pT tail
  - Coalescence effects possible
  
- **Extent ≈ 1.0**: Similar endpoints
  - No significant tail creation

- **Extent < 1.0**: Spectrum truncated
  - Energy loss suppresses high-pT
  - Absorption of fast particles

### Example

```
Modified:   max(pT) = 8.23 GeV
Unmodified: max(pT) = 7.51 GeV
Extent: 8.23 / 7.51 = 1.096

Interpretation: 9.6% extension of high-pT tail
     ✓ Hard scattering enhanced
```

### Jet Quenching Connection

In real heavy-ion collisions:
- **Extent > 1.0**: No quenching (free streaming)
- **Extent ≈ 1.0**: Weak quenching
- **Extent < 1.0**: Strong quenching (QGP effects)

---

## Quaternary Observable: Rapidity/Pseudorapidity Distributions

### What It Measures

Distribution widths in rapidity (y) and pseudorapidity (η):

```
Width = σ_y or σ_η
```

### Physical Meaning

**Forward/backward particle distribution.**

- **y or η**: Measures particle direction along collision axis
- **Wider distribution**: Particles scatter in all directions
- **Narrower distribution**: Particles stay near collision axis

### Formulas

**Rapidity**:
```
y = 0.5 × ln((E + p_z) / (E - p_z))
```
- Energy-dependent
- Additive under boosts
- Invariant observable

**Pseudorapidity**:
```
η = 0.5 × ln((p + p_z) / (p - p_z))
```
- Momentum-dependent
- Easier to compute
- Detector-friendly

### Interpretation

| Signature | Physics | Conclusion |
|-----------|---------|-----------|
| σ_y(mod) > σ_y(unm) | Wider rapidity spread | Isotropic scattering |
| σ_y(mod) ≈ σ_y(unm) | No change | No directional effects |
| σ_y(mod) < σ_y(unm) | Narrower spread | Asymmetric medium |

### Anisotropy Effects

If **η distribution changes more than y distribution**:
- Suggests **transverse momentum effects dominate**
- Evidence for **non-central collisions**
- Possible **elliptic flow** signature

---

## Cumulative Signature Types

### 1. Broadening Signature

**Definition**: σ_pT ratio > 1.05

**Indicates**: Multiple scattering in medium

**Strength scale**:
- 1.00–1.02: Weak (< 2% broadening)
- 1.02–1.05: Moderate (2–5% broadening)
- 1.05–1.10: Strong (5–10% broadening)
- \> 1.10: Very strong (> 10% broadening)

**Particle types affected**: All species

**Energy dependence**:
```
Broadening ∝ Path Length
          ∝ √ System Size
          ∝ 1 / Collision Energy  (at fixed impact parameter)
```

### 2. Deflection Signature

**Definition**: Change in rapidity distribution symmetry

**Indicates**: Anisotropic scattering pattern

**Manifestation**:
- Forward/backward asymmetry
- Deviation from symmetric distribution
- Preferential deflection direction

**Typical causes**:
- Non-central collisions
- Geometry effects
- Directed flow

### 3. Energy Shift Signature

**Definition**: ⟨pT⟩_modified ≠ ⟨pT⟩_unmodified

**Positive shift**: 
- Scattering accumulation
- Multiple small-angle kicks
- Typical in weak interactions

**Negative shift**:
- Energy loss (radiation)
- Typical in dense QGP
- Proportional to path length

### 4. Multiplicity Change Signature

**Definition**: Different number of particles produced

**Indicates**: 
- Inelastic interactions
- Particle creation/absorption
- Final state regeneration

**Example**:
```
Modified:   N_particles = 248 ± 12
Unmodified: N_particles = 247 ± 13
Change: < 1 particle on average → Not significant
```

---

## Confidence Levels

### How to Interpret

**Each signature has a confidence score (0–1)**:

- **0.90–1.00**: Very confident (robust detection)
- **0.75–0.90**: Confident (clear signature)
- **0.50–0.75**: Moderate confidence (detectable but noisy)
- **0.25–0.50**: Low confidence (marginal detection)
- **0.00–0.25**: Unreliable (may be statistical fluctuation)

### Example

```json
{
  "type": "broadening",
  "strength": 0.87,           ← 87% of maximum effect
  "confidence": 0.92,         ← 92% confident
  "affected_particles": 2450  ← Strong statistical sample
}
```

**Interpretation**: This is a robust detection. The effect is substantial (87%) and statistically significant (92% confidence). About 90% of produced particles exhibit the effect.

---

## System Dependence

### Small Systems (p+A)

- **pT broadening**: Small (< 5%) due to limited path length
- **Signatures**: Mostly deflection (geometric)
- **Confidence**: Lower (fewer events, smaller sample)

### Medium Systems (Cu+Cu)

- **pT broadening**: Moderate (5–10%)
- **Signatures**: Mixed broadening and deflection
- **Confidence**: Good (reasonable statistics)

### Large Systems (Au+Au, Pb+Pb)

- **pT broadening**: Large (10–30%)
- **Signatures**: Strong broadening with energy shift
- **Confidence**: Excellent (high multiplicity)

### Energy Dependence

```
                    pT Broadening
                         ↑
                         │  
                   Au+Au │    ●
                    14 GeV│   ╱
                         │  ╱
                    Au+Au│ ●
                     11 GeV
                         │
                    Au+Au│●
                    7.7 GeV
                         │
                    Au+Au│
                    3.8 GeV●
                         │
                         └─────────→ Density ∝ 1/s_NN
```

Lower beam energy → Higher density → Stronger broadening

---

## Comparison with Experiments

### RHIC (Au+Au at √s_NN = 200 GeV)

**Expected signatures**:
- pT broadening: 10–15%
- Mean pT shift: -2% to 0% (energy loss → gain)
- Confidence: 90%+

**Physics**: Dense QGP, strong interactions

### RHIC BES (Au+Au at √s_NN = 14–20 GeV)

**Expected signatures**:
- pT broadening: 5–10%
- Mean pT shift: 0% to +2% (scattering-dominated)
- Confidence: 85%+

**Physics**: Transition to hadronic matter

### NICA (Au+Au at √s_NN = 4–11 GeV)

**Expected signatures**:
- pT broadening: 3–8%
- Mean pT shift: +1% to +3% (strong scattering)
- Confidence: 70%–85%

**Physics**: High baryon density, nucleon matter

---

## Interpretation Checklist

✓ **Data quality check**:
- [ ] Modified and unmodified files are different
- [ ] At least 100 events per sample
- [ ] Collision system properly detected

✓ **Observable values**:
- [ ] pT broadening ratio > 1.05 OR < 0.95 (not exactly 1.0)
- [ ] Standard deviations > 0 (non-degenerate)
- [ ] Statistical confidence > 0.5

✓ **Physics consistency**:
- [ ] Broadening consistent with system size
- [ ] Energy dependence follows expected trend
- [ ] Signatures don't contradict each other

✓ **Systematic checks**:
- [ ] Results stable across different batch sizes
- [ ] Similar results from multiple runs
- [ ] Ratios independent of analysis method

---

## Common Pitfalls

### Pitfall 1: Identical Files
**Problem**: Modified and unmodified datasets are identical
**Sign**: All ratios ≈ 1.000, confidence ≈ 0
**Fix**: Check generator settings, verify modifications applied

### Pitfall 2: Statistical Noise
**Problem**: Random fluctuations exceed real effects
**Sign**: Confidence < 0.3, ratio varies wildly between runs
**Fix**: Increase sample size (more events)

### Pitfall 3: Misconfigured System
**Problem**: Detector thinks it's wrong collision system
**Sign**: Unusual pT ranges (too small/large), multiplicity mismatch
**Fix**: Verify with `collision_system_detector.py`, manually specify if needed

### Pitfall 4: Format Errors
**Problem**: Oscar format not properly parsed
**Sign**: "No events found", zero particles
**Fix**: Verify format with `head -3 file.f19`, check against Oscar specs

---

## Publication Guidance

### Figure 1: pT Broadening (PRIMARY)

Caption:
> "Transverse momentum distribution width comparison between modified and unmodified datasets. Ratio σ_pT(modified)/σ_pT(unmodified) = 1.079 ± 0.012 indicates multiple scattering signature with 92% confidence. The 7.9% broadening is consistent with cumulative interactions in Au+Au collisions at √s_NN = 10 GeV."

### Figure 2: Mean pT Shift (SECONDARY)

Caption:
> "Mean transverse momentum shift: Δ⟨pT⟩ = +0.8 MeV (0.2% increase). Positive shift indicates scattering-dominated regime rather than energy loss, suggesting low-density system or forward rapidity range."

### Figure 3: Distribution Extent (TERTIARY)

Caption:
> "High-pT tail extension: pT_max increases 9.6% in modified sample, indicating enhanced hard scattering from multiple interactions."

### Figure 4: Rapidity/Pseudorapidity (QUATERNARY)

Caption:
> "Rapidity distribution widths: σ_y increases from 1.854 to 1.876 (0.6% change), consistent with isotropic scattering pattern without directional bias."

---

## References

- RHIC Beam Energy Scan: https://www.bnl.gov/bes/
- Parton saturation and cumulative effects:
  - Qiu, J.W., & Sterman, G. (2012). Ann. Rev. Nucl. Part. Sci. 62, 443
- Multiple scattering theory:
  - Levai, P., & Vogt, R. (2004). Phys. Rev. D 78, 114025
- pT broadening measurements:
  - Majumder, A. (2012). Nucl. Phys. A 931, 6

---

## Questions?

Refer to:
- `README.md` for project overview
- `API_REFERENCE.md` for technical details
- `QUICKSTART.md` for practical examples
- Source code comments for implementation details
