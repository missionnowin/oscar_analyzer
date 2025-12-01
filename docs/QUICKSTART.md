# Quick Start Guide

Get the analysis pipeline running in 5 minutes.

## Prerequisites

```bash
python3 -m pip install numpy matplotlib
```

Verify:
```bash
python3 -c "import numpy, matplotlib; print('✓ Dependencies OK')"
```

## File Setup

Organize your data as follows:

```
project_root/
├── data/
│   ├── modified/
│   │   ├── run_1.f19
│   │   ├── run_2.f19
│   │   └── run_3.f19
│   └── no_modified/
│       ├── run_1.f19
│       ├── run_2.f19
│       └── run_3.f19
├── results/                    # Will be created automatically
└── [code files from this repo]
```

## Step 1: Verify Data Quality

Before analyzing, confirm your files are different:

```bash
python3 verify_data_integrity.py \
  --mod data/modified/run_1.f19 \
  --unm data/no_modified/run_1.f19 \
  --events 5
```

Expected output:
```
Modified file: 1,000 events
Unmodified file: 1,000 events

Comparing first 5 events:
Event 0: ✓ Different (82/100 particles match)
Event 1: ✓ Different (79/100 particles match)
Event 2: ✗ IDENTICAL (100/100 particles match)  ← Some identical is OK
Event 3: ✓ Different (85/100 particles match)
Event 4: ✓ Different (81/100 particles match)

SUMMARY:
Different events: 4/5
✓ Files appear to be different (as expected)
```

**✓ Success**: Different events > 0
**✗ Problem**: All identical → Check your generator/data pipeline

## Step 2: Detect Collision System

Auto-identify your collision system:

```bash
python3 -m utils.collision_system_detector data/modified/run_1.f19
```

Output:
```
Detected Collision System:
 Label: Au+Au @ 10 GeV
 Projectile: Au (Z=79, A=197)
 Target: Au (Z=79, A=197)
 Energy: √s_NN = 10.0 GeV
 Confidence: 85%
 Events analyzed: 100
 Avg particles/event: 247
```

**Note**: Confidence < 70% means the detector is uncertain. You can manually specify the system in the analysis script.

## Step 3: Run Analysis

### Option A: Single Run (Test)

Quick test on one run:

```python
#!/usr/bin/env python3
from run_all_analysis import UnifiedAnalysisEngine
from pathlib import Path

engine = UnifiedAnalysisEngine(
    data_root=Path("data"),
    results_root=Path("results"),
    n_workers=4,           # Adjust for your system
    batch_size=500
)

engine.run_parallel(runs=[1])  # Just run 1
```

Run it:
```bash
python3 test_single_run.py
```

Expected output:
```
================================================================================
UNIFIED COLLISION ANALYSIS PIPELINE
================================================================================
Runs: [1] | Workers: 4 | Batch size: 500

[12:34:56] [run_1] Processing        | 500/1000 event pairs
[12:34:57] [run_1] Processing        | 1000/1000 event pairs
[12:34:58] [run_1] Plotting modified  | Generating...
[12:35:02] [run_1] Plotting unmodified | Generating...
[12:35:06] [run_1] Plotting cumulative | Generating...
[12:35:10] [run_1] Comparing          | Generating...
[12:35:14] [run_1] Saving             | Writing cumulative analysis...
[12:35:14] [run_1] COMPLETED          | Au+Au @ 10 GeV
```

Check results:
```bash
ls -la results/modified/run_1/
ls -la results/comparisons/run_1/
cat results/stats/run_1_stats.json | python3 -m json.tool
```

### Option B: Multiple Runs (Production)

Process multiple runs in parallel:

```python
#!/usr/bin/env python3
from run_all_analysis import UnifiedAnalysisEngine
from pathlib import Path

engine = UnifiedAnalysisEngine(
    data_root=Path("data"),
    results_root=Path("results"),
    n_workers=8,           # Use all cores
    batch_size=500
)

# Process runs 1–10
engine.run_parallel(runs=list(range(1, 11)))
```

Run it:
```bash
python3 run_analysis.py 2>&1 | tee analysis.log
```

Monitor progress:
```bash
# In another terminal:
watch -n 1 'ls results/comparisons/ | wc -l'  # Count finished runs
```

## Step 4: Examine Results

### Plots

Browse generated plots:

```bash
# Modified dataset
ls results/modified/run_1/

# Unmodified dataset
ls results/unmodified/run_1/

# Cumulative effects
ls results/cumulative/run_1/

# Comparisons (most important!)
ls results/comparisons/run_1/
```

Open comparison plots in your favorite viewer:

```bash
# macOS
open results/comparisons/run_1/01_pt_broadening_primary.png

# Linux
display results/comparisons/run_1/01_pt_broadening_primary.png

# or use image viewer in file manager
```

### Key Plot: pT Broadening (PRIMARY)

`01_pt_broadening_primary.png` shows:
- **Left bar**: Modified dataset pT width (σ)
- **Right bar**: Unmodified dataset pT width
- **Ratio annotation**: σ(mod) / σ(unm)
  - **Ratio > 1.05** ✓ Multiple scattering detected
  - **Ratio ≈ 1.00** ✗ No clear broadening

### Statistics JSON

View aggregate statistics:

```bash
python3 -c "
import json
with open('results/stats/run_1_stats.json') as f:
    data = json.load(f)
    
print('MODIFIED:')
print(f\"  pT mean: {data['modified']['pt']['mean']:.4f} GeV\")
print(f\"  pT std: {data['modified']['pt']['std']:.4f} GeV\")
print(f\"  Multiplicity: {data['modified']['multiplicity']['mean']:.0f} particles\")

print()
print('UNMODIFIED:')
print(f\"  pT mean: {data['unmodified']['pt']['mean']:.4f} GeV\")
print(f\"  pT std: {data['unmodified']['pt']['std']:.4f} GeV\")
print(f\"  Multiplicity: {data['unmodified']['multiplicity']['mean']:.0f} particles\")

pT_ratio = data['modified']['pt']['std'] / data['unmodified']['pt']['std']
print()
print(f'pT broadening ratio: {pT_ratio:.4f}')
if pT_ratio > 1.05:
    print('✓ SIGNIFICANT BROADENING DETECTED')
else:
    print('✗ No significant broadening')
"
```

### Cumulative Signatures

Check detected signatures:

```bash
python3 -c "
import json
with open('results/cumulative/run_1_cumulative.json') as f:
    data = json.load(f)
    
print(f\"Total events: {data['cumulative_analysis']['total_events']:,d}\")
print(f\"Signatures detected: {data['cumulative_analysis']['signatures_detected']}\")
print()
for sig in data['signatures']:
    print(f\"{sig['type']}:\")
    print(f\"  Strength: {sig['strength']:.2f}\")
    print(f\"  Confidence: {sig['confidence']:.2f}\")
    print(f\"  Affected: {sig['affected_particles']} particles\")
    print()
"
```

## Troubleshooting

### Q: Analysis is very slow
**A**: 
- Reduce batch size: `batch_size=100` (uses less memory)
- Or: Increase workers: `n_workers=16` (if you have cores)
- Or: Use fewer runs in test: `runs=[1]`

### Q: Out of memory error
**A**:
- Reduce batch size: `batch_size=100`
- Reduce workers: `n_workers=4`
- Or: Process fewer runs at once

### Q: "No events found" error
**A**:
- Check file format: `python3 verify_data_integrity.py --mod FILE`
- Verify first 3 lines are Oscar format:
  ```bash
  head -3 data/modified/run_1.f19
  ```
  Should show Oscar header, not binary garbage

### Q: Files are IDENTICAL (no differences)
**A**:
- Check generator settings (did you apply modifications?)
- Compare file sizes: `ls -la data/modified/ data/no_modified/`
- Sample manually: `head -100 data/modified/run_1.f19 > /tmp/m1.txt`
           `head -100 data/no_modified/run_1.f19 > /tmp/u1.txt`
           `diff /tmp/m1.txt /tmp/u1.txt`

### Q: Collision system not detected (confidence < 50%)
**A**:
- Use more events: `CollisionSystemDetector.detect_from_file(..., n_events_sample=1000)`
- Or manually specify in code:
  ```python
  system_info = {
      'label': 'Au+Au @ 10 GeV',
      'sqrt_s_NN': 10.0,
      'A1': 197, 'Z1': 79,
      'A2': 197, 'Z2': 79
  }
  ```

## Next Steps

1. **Understand the plots**:
   - 01_pt_broadening_primary.png: Main observable
   - 02_mean_pt_shift.png: Energy loss/gain
   - 03_pt_distribution_tails.png: Spectrum extent
   - 04_rapidity_eta_spectra.png: Angular distributions

2. **Batch processing**:
   - Process 100+ runs for statistical confidence
   - Compare signatures across runs
   - Build summary statistics

3. **Custom analysis**:
   - Modify analyzers for your physics questions
   - Add new observables (angular correlations, etc.)
   - Integrate with your detector simulation

4. **Publication**:
   - Use generated plots in papers/presentations
   - Include statistics JSON in supplementary material
   - Cite cumulative effect signatures

## Tips & Tricks

### Monitor job progress
```bash
# Terminal 1: Run analysis
python3 run_analysis.py

# Terminal 2: Watch output
watch -n 5 'ls results/comparisons | wc -l'
```

### Process specific runs only
```python
# Just runs 1, 5, 10
engine.run_parallel(runs=[1, 5, 10])
```

### Parallel processing on HPC
```python
# For 32 cores:
engine = UnifiedAnalysisEngine(
    data_root=Path("data"),
    results_root=Path("results"),
    n_workers=32,      # Use all available
    batch_size=100     # Smaller batches with many workers
)
```

### Save processing log
```bash
python3 run_analysis.py 2>&1 | tee analysis_$(date +%Y%m%d_%H%M%S).log
```

### Extract statistics quickly
```bash
python3 -c "
import json
from pathlib import Path

for json_file in Path('results/stats').glob('*.json'):
    with open(json_file) as f:
        data = json.load(f)
    ratio = data['modified']['pt']['std'] / data['unmodified']['pt']['std']
    print(f\"{json_file.stem}: ratio={ratio:.4f}\")
"
```

---

**Done!** You should now have:
- ✓ Verified data integrity
- ✓ Identified collision system
- ✓ Generated analysis plots
- ✓ Extracted physics observables
- ✓ Produced publication-ready comparisons

For detailed documentation, see `README.md` and `API_REFERENCE.md`.
