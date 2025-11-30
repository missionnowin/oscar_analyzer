# Quick Reference Guide

## Common Commands

### 1. **Process All Runs** (Production)
```bash
python run_all_analysis.py \
  --data-root data \
  --results-root results \
  --runs 1 2 3 4 5 \
  --workers 4
```
Output: `results/aggregate_summary.json`, PNG plots in `results/`

---

### 2. **Debug Single Run**
```bash
python detect_cumulative.py \
  --modified data/modified/run_1.f19 \
  --unmodified data/no_modified/run_1.f19 \
  --plot all \
  --output debug_results
```
Output: Detailed plots + JSON report in `debug_results/`

---

### 3. **Verify Data Before Expensive Analysis**
```bash
python verify_data_integrity.py \
  --mod data/modified/run_1.f19 \
  --unm data/no_modified/run_1.f19 \
  --events 5
```
Checks: Multiplicity, momentum differences, statistical divergence

---

## Data Directory Setup

```bash
# Create structure
mkdir -p data/{modified,no_modified}
mkdir -p results

# Copy your files
cp *.f19 data/modified/        # or your naming convention
cp *.baseline.f19 data/no_modified/

# Or create symbolic links
ln -s /path/to/external/data modified/
ln -s /path/to/baseline unmodified/
```

---

## Output Interpretation

### `aggregate_summary.json` Structure

```json
{
  "run_1": {
    "success": true,
    "general_stats": {
      "modified": {
        "n_events": 5000,           // Total events analyzed
        "n_total_particles": 150000, // All particles summed
        "pt": {
          "mean": 0.425,            // ⟨pT⟩
          "std": 0.312,             // σ(pT)
          "median": 0.350
        },
        "y": {...},                 // Rapidity stats
        "eta": {...}                // Pseudorapidity stats
      },
      "unmodified": {...}
    },
    "cumulative_analysis": {
      "likelihood": 0.82,           // 0-1 confidence score
      "n_signatures": 3,            // Number of detected signatures
      "signatures": [
        {
          "type": "cumulative_variable",
          "strength": 0.95,         // Signal magnitude
          "confidence": 0.95,       // Detection confidence
          "affected_particles": 2145, // Particles with x > 1.1
          "description": "Direct cumulative detection: 2145 particles..."
        }
      ]
    },
    "plots_generated": [
      "modified/run_1/pt_distribution.png",
      "comparisons/run_1/comparison.png",
      ...
    ]
  }
}
```

### Plot Files

| Plot | Interpretation |
|------|-----------------|
| `pt_distribution.png` | pT spectrum; higher momentum tail = cumulative effects |
| `rapidity_distribution.png` | Forward-backward symmetry; forward excess = cumulative |
| `eta_distribution.png` | Pseudorapidity shape; matches detector acceptance |
| `y_vs_pt.png` | 2D correlation; cumulative particles at high pT/y |
| `eta_vs_phi.png` | Angular correlations; azimuthal uniformity |
| `comparison.png` | Modified vs. Unmodified overlay; visible differences = signal |

---

## Troubleshooting

### Problem: "No events found in modified file"
```
Error: No events found in modified file: data/modified/run_1.f19
```

**Diagnosis:**
```bash
# Check file exists and is readable
ls -lh data/modified/run_1.f19

# Verify format
head -20 data/modified/run_1.f19  # Should show event headers + particle lines

# Test with reader directly
python -c "from utils.readers import OscarReader; print(len(OscarReader('data/modified/run_1.f19').read_file()))"
```

**Fix:** Ensure `.f19` is valid Oscar1992A format

---

### Problem: "Memory error" / Process killed
```
MemoryError: Unable to allocate X.XX GiB for an array
```

**Diagnosis:**
```bash
# Check available RAM
free -h

# Check current batch_size
ps aux | grep run_all_analysis
```

**Fix:** Reduce batch size or worker count
```bash
# Reduce from 500 → 250 events/batch
python run_all_analysis.py ... --batch-size 250

# Or reduce workers
python run_all_analysis.py ... --batch-size 500 --workers 2
```

---

### Problem: "ModuleNotFoundError: No module named 'analyzers'"
```
ModuleNotFoundError: No module named 'analyzers'
```

**Fix:** Run from repository root
```bash
# WRONG:
cd analyzers/
python ../run_all_analysis.py

# CORRECT:
cd /path/to/repository
python run_all_analysis.py
```

---

### Problem: Plots not generating (no error, but no PNGs)
```
# analyze runs, but results/ is empty or sparse
```

**Debug:**
```bash
# Enable verbose output
python run_all_analysis.py ... 2>&1 | grep "Plotting\|ERROR\|Failed"

# Check if matplotlib can display
python -c "import matplotlib.pyplot as plt; plt.plot([1,2,3]); print('matplotlib OK')"
```

**Fix:** May be headless environment
```bash
# Force non-display backend
export MPLBACKEND=Agg
python run_all_analysis.py ...
```

---

## Performance Tuning

### Benchmark: 5 runs, 5000 events/run, 30 particles/event

| Config | Time | Memory |
|--------|------|--------|
| `--workers 1 --batch 500` | 2:15 | 2.1 GB |
| `--workers 2 --batch 500` | 1:10 | 4.2 GB |
| `--workers 4 --batch 500` | 0:35 | 8.4 GB |
| `--workers 4 --batch 250` | 0:40 | 4.2 GB |
| `--workers 8 --batch 500` | 0:30 | 16.8 GB |

**Recommendations:**
- **Desktop (16GB RAM, 4 cores):** `--workers 4 --batch 500`
- **Laptop (8GB RAM, 2 cores):** `--workers 2 --batch 250`
- **HPC (256GB RAM, 32 cores):** `--workers 16 --batch 500`

---

## Physics Parameters

### Cumulative Thresholds (in `cumulative_detector.py`)

```python
self.cumulative_threshold = 1.1  # x > 1.1 for cumulative particle
self.forward_cut_xf = 0.3        # x_F > 0.3 for forward direction
self.cm_energy = 10.0            # √s_NN = 10 GeV (Au+Au)
```

### If analyzing different collision system:

```bash
# Manually set Au+Au @ 5 GeV
# Edit run_all_analysis.py line ~XXX:
system_info = {
    'label': 'Au+Au @ 5 GeV',
    'sqrt_s_NN': 5.0,
    'A1': 197, 'Z1': 79,
    'A2': 197, 'Z2': 79
}
```

---

## File Format Reference

### Oscar1992A (`.f19`)

```
# Event header line:
TIME NPARTICLES

# Particle lines (one per particle):
PDGID MASS X Y Z PX PY PZ E TFORMED CHARGE BARYON_DENSITY

# Example:
0 0.0 500
211 0.140 0.0 0.0 0.0 0.5 0.3 1.2 1.5 0.0 1 0.0
2212 0.938 0.1 0.1 0.1 0.2 0.1 0.8 1.1 0.0 1 0.1
```

### OSC1997A (`final_id_p_x`)

```
# Global header (skipped by reader)
OSC1997A
final_id_p_x
UrQMD event dump ...

# Event header:
EVENT_ID NPARTICLES B PHI

# Particle lines:
INDEX PDG PX PY PZ E MASS X Y Z T

# Example:
1 100
0 1 211 0.5 0.3 1.2 1.5 0.140 0.0 0.0 0.0 0.0
1 2212 0.2 0.1 0.8 1.1 0.938 0.1 0.1 0.1 0.1
```

---

## Report Generation

### Generate Summary Report (Markdown)

```python
# save_summary.py
import json
from pathlib import Path

with open('results/aggregate_summary.json') as f:
    data = json.load(f)

with open('ANALYSIS_REPORT.md', 'w') as out:
    out.write("# Cumulative Effect Analysis Report\n\n")
    
    for run_name, result in data.items():
        out.write(f"## {run_name}\n\n")
        
        if result['success']:
            stats = result['general_stats']['modified']
            out.write(f"**Events:** {stats['n_events']}\n\n")
            out.write(f"**pT Spectrum:** {stats['pt']['mean']:.3f} ± {stats['pt']['std']:.3f} GeV\n\n")
            
            cum = result['cumulative_analysis']
            out.write(f"**Cumulative Likelihood:** {cum['likelihood']:.2f}\n\n")
            out.write(f"**Signatures Detected:** {cum['n_signatures']}\n\n")
        else:
            out.write(f"**Status:** FAILED - {result['error']}\n\n")
```

Run:
```bash
python save_summary.py
```

---

## Citation Template

**For papers using this framework:**

```bibtex
@software{urqmd_cumulative_2025,
  title={UrQMD Cumulative Effect Analysis Framework},
  author={Your Name},
  year={2025},
  url={https://github.com/yourname/urqmd-cumulative},
  note={Analysis pipeline for detecting multi-nucleon scattering signatures in heavy-ion collisions}
}
```

---

## Additional Resources

- **UrQMD Homepage:** https://urqmd.org/
- **NumPy Documentation:** https://numpy.org/
- **Matplotlib Tutorials:** https://matplotlib.org/tutorials/
- **Python Multiprocessing:** https://docs.python.org/3/library/multiprocessing/

---

## Support & Issues

**If something doesn't work:**

1. Check this troubleshooting section
2. Run `verify_data_integrity.py` on one file pair
3. Check `TECHNICAL.md` for design details
4. Review console output for error messages
5. Reduce problem scope: `--runs 1` instead of full batch

