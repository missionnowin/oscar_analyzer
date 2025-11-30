# Technical Documentation: UrQMD Cumulative Effect Analysis Framework

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Module Reference](#module-reference)
3. [Data Flow](#data-flow)
4. [API Reference](#api-reference)
5. [Algorithm Details](#algorithm-details)
6. [Design Improvements & Recommendations](#design-improvements--recommendations)

---

## Architecture Overview

### Design Principles

The framework follows a **modular, pipeline-based architecture**:

```
Data Input (Oscar files)
    ↓
Format Detection & Parsing (readers.py)
    ↓
Per-Event Analysis (collision_analyzer.py)
    ↓
Cumulative Signature Detection (cumulative_detector.py)
    ↓
Cross-Event Aggregation (aggregate_analyzer.py)
    ↓
Visualization & Output (plot_* methods)
```

### Layer Separation

| Layer | Modules | Responsibility |
|-------|---------|-----------------|
| **I/O** | `readers.py`, `particle.py` | File parsing, data models |
| **Event Analysis** | `angle_analyzer.py`, `collision_analyzer.py` | Per-event kinematics |
| **Physics Detection** | `cumulative_detector.py` | Signature identification |
| **Aggregation** | `aggregate_analyzer.py` | Multi-event statistics |
| **Comparison** | `comparison_analyzer.py`, `collision_system_detector.py` | Cross-sample analysis |
| **Orchestration** | `run_all_analysis.py`, `detect_cumulative.py` | Workflow management |

---

## Module Reference

### 1. `readers.py` — File Format Handling

**Purpose:** Auto-detect and parse Oscar format files

**Classes:**
- `OscarFormat` (Enum) — Format types: OSCAR1992A, OSC1997A
- `OscarFormatDetector` — Format auto-detection
- `Oscar1992AReader` — Parser for `.f19` format
- `OSC1997AReader` — Parser for `final_id_p_x` format
- `OscarReader` (Universal) — Wrapper selecting appropriate reader

**Key Method:**
```python
def read_file(self) -> List[List[Particle]]:
    """Parse file → list of events, each event = list of Particle objects"""
```

**Example:**
```python
reader = OscarReader("run_1.f19")  # Auto-detects format
events = reader.read_file()        # Returns [[Particle, Particle, ...], ...]
```

---

### 2. `collision_analyzer.py` — Event-Level Kinematics

**Purpose:** Compute single-event kinematic quantities

**Main Class:** `CollisionAnalyzer`

**Computed Quantities:**
```python
# Precomputed as numpy arrays for entire event:
.px, .py, .pz              # Cartesian momentum
.pt                        # Transverse momentum
.phi, .theta               # Azimuthal & polar angles
.y                         # Rapidity
.eta                       # Pseudorapidity
.E                         # Energy
.pdg                       # PDG codes (if available)
```

**Methods:**
```python
# Plotting
plot_rapidity()            # dN/dy distribution
plot_pseudorapidity()      # dN/dη distribution
plot_pt()                  # dN/dpT spectrum
plot_pid_pt(species)       # Identified particle spectra (π, K, p)
plot_y_pt()                # 2D correlation
plot_eta_phi()             # 2D angular correlation
estimate_v2()              # Elliptic flow estimator
```

**Physics Variables:**
- **Rapidity:** $y = \frac{1}{2}\ln\left(\frac{E+p_z}{E-p_z}\right)$
- **Pseudorapidity:** $\eta = -\ln\left(\tan(\theta/2)\right)$
- **Polar angle:** $\theta = \arctan2(p_T, p_z)$

---

### 3. `cumulative_detector.py` — Cumulative Signature Detection

**Purpose:** Identify particles with $x > 1$ (multi-nucleon correlation signatures)

**Main Class:** `CumulativeEffectDetector`

**Detection Methods:**

#### Primary Detector
```python
def detect_cumulative_signal() -> CumulativeSignature:
    """Direct identification via cumulative variable x = (E - p_z) / m_N
    
    Returns signature if:
    - Single sample: x > 1.1 for any particle
    - Comparison: modified has higher x> 1.1 fraction than unmodified
    """
```

**Criteria for Cumulative Particles:**
1. $x = \frac{E - p_z}{m_N} > 1.1$ (kinematic limit exceeds single nucleon)
2. $x_F = \frac{p_z}{p_{z,max}} > 0.3$ (forward-going particles preferred)
3. Non-relativistic regime: $\beta < 0.95$ (optional additional cut)

#### Secondary Detectors
```python
def detect_pt_broadening()      # Transverse momentum dispersion ratio
def detect_forward_enhancement() # Enhanced forward particle production
```

**Likelihood Calculation:**
```python
def get_cumulative_likelihood() -> float:
    """
    Weighted score (0-1):
    - Primary (cumulative_variable): 70% weight
    - Secondary signals: 30% weight
    """
```

**Output:** `CumulativeSignature` with:
- `signature_type` — Detection method
- `strength` — 0-1 signal magnitude
- `confidence` — 0-1 confidence in detection
- `affected_particles` — Count of particles matching criterion
- `description` — Physics interpretation

---

### 4. `aggregate_analyzer.py` — Multi-Event Accumulation

**Purpose:** Accumulate statistics across all events in a file

**Main Class:** `AggregateAnalyzer`

**Key Method:**
```python
def accumulate_event(particles: List[Particle]) -> None:
    """Add one event's particles to running statistics"""
    
def get_statistics() -> Dict:
    """Return comprehensive statistics dictionary:
    {
        'n_events': int,
        'n_total_particles': int,
        'pt': {'mean': float, 'std': float, 'median': float, ...},
        'y': {...},
        'eta': {...},
        ...
    }
    """
```

**Plotting:**
```python
def plot_distributions(output_dir) -> List[str]:
    """Generate PNG plots:
    - pT distribution (density, log scale)
    - Rapidity distribution
    - Pseudorapidity distribution
    - 2D y vs pT correlation
    - 2D η vs φ correlation
    
    Returns list of generated filenames
    """
```

---

### 5. `comparison_analyzer.py` — Modified vs. Unmodified Comparison

**Purpose:** Statistical comparison between sample pair

**Main Class:** `RunComparisonAnalyzer`

**Comparison Metrics:**
```python
def generate_all_comparisons():
    """
    Generates comparative plots:
    - pT mean/width ratio (broadening indicator)
    - Rapidity shape difference
    - Particle yield ratios
    - Event-by-event fluctuation analysis
    """
```

---

### 6. `collision_system_detector.py` — Collision System Auto-Detection

**Purpose:** Automatically infer collision system from first few events

**Main Class:** `CollisionSystemDetector`

**Detection Logic:**
```python
def detect_from_file(filepath, n_events_sample=10) -> Dict:
    """
    Sample first N events to estimate:
    - Collision energy (√s_NN) via multiplicity, pT spectra
    - Projectile/target (A₁, Z₁, A₂, Z₂) via mass considerations
    
    Returns:
    {
        'label': 'Au+Au @ 10 GeV',
        'sqrt_s_NN': 10.0,
        'A1': 197, 'Z1': 79,
        'A2': 197, 'Z2': 79
    }
    """
```

---

## Data Flow

### Single Run Analysis Flow

```
Modified .f19 file        Unmodified .f19 file
         |                         |
         v                         v
    OscarReader               OscarReader
         |                         |
    List[Event]              List[Event]
         |                         |
         +----------+----------+
                    |
                    v
        CollisionSystemDetector
         (auto-detect Au+Au 10 GeV)
                    |
        +---+-------+-------+---+
        |   |               |   |
        v   v               v   v
       Event 1       ...   Event N
        
   CollisionAnalyzer    AggregateAnalyzer
   (per-event)          (accumulate)
        |                    |
   Kinematics           Statistics Dict
        |                    |
   CumulativeDetector        v
   (detect x > 1.1)     plot_distributions()
        |                    |
   Signature List            v
        |              PNG Plots
        +------+-------+
               |
          compare_events()
               |
          RunComparisonAnalyzer
               |
          Comparison Plots
               |
          JSON Summary
```

### Full Pipeline Flow

```
run_all_analysis.py
    |
    +-- For each run_N in runs:
    |       |
    |       +-- Detect collision system
    |       |
    |       +-- Process modified/run_N.f19
    |       |       +-- Read events (batches)
    |       |       +-- Accumulate statistics
    |       |       +-- Generate plots
    |       |
    |       +-- Process unmodified/run_N.f19
    |       |       +-- (same as above)
    |       |
    |       +-- Compare samples
    |       |
    |       +-- Detect cumulative effects
    |       |
    |       +-- Save run_N JSON
    |
    +-- Aggregate all runs
    |       +-- Combine statistics
    |       +-- Generate cross-run plot
    |
    +-- Save aggregate_summary.json
```

---

## API Reference

### `CumulativeEffectDetector` Complete API

```python
class CumulativeEffectDetector:
    def __init__(self, particles_modified, particles_unmodified=None):
        """Initialize with particle list(s)"""
    
    @staticmethod
    def calculate_cumulative_variable(px, py, pz, mass=0.938) -> float:
        """x = (E - p_L) / m_N"""
    
    @staticmethod
    def calculate_feynman_xf(pz, p_L_max=None) -> float:
        """Feynman scaling variable"""
    
    def identify_cumulative_particles(particles) -> Dict:
        """
        Returns: {
            'cumulative_mask': np.ndarray bool,
            'x_values': np.ndarray,
            'xf_values': np.ndarray,
            'n_cumulative': int,
            'fraction': float,
            'x_mean': float,
            'x_max': float
        }
        """
    
    def detect_cumulative_signal() -> CumulativeSignature | None
    def detect_pt_broadening() -> CumulativeSignature | None
    def detect_forward_enhancement() -> CumulativeSignature | None
    
    def detect_all_signatures() -> List[CumulativeSignature]
    
    def get_cumulative_likelihood() -> float:
        """0-1 overall likelihood score"""
    
    def print_report() -> None
```

### `AggregateAnalyzer` Complete API

```python
class AggregateAnalyzer:
    def __init__(self, system_label=None, sqrt_s_NN=None,
                 A1=None, Z1=None, A2=None, Z2=None):
        """Initialize with collision system parameters"""
    
    def accumulate_event(particles: List[Particle]) -> None
    
    def get_statistics() -> Dict:
        """Return comprehensive statistics"""
    
    def plot_distributions(output_dir: Path) -> List[str]:
        """Generate all distribution plots"""
```

### `CollisionAnalyzer` Complete API

```python
class CollisionAnalyzer:
    def __init__(self, particles: List[Particle], system_label: str):
        """Initialize with event particles"""
        # Auto-computes:
        # .px, .py, .pz, .E, .pt, .p
        # .phi, .theta, .phi_deg, .theta_deg
        # .y, .eta, .pdg (if available)
    
    def plot_rapidity(out=None, y_range=(-2.0, 2.0)) -> None
    def plot_pseudorapidity(out=None, eta_range=(-3.0, 3.0)) -> None
    def plot_pt(out=None, pt_max=3.0) -> None
    def plot_pid_pt(species, out=None, pt_max=3.0) -> None
    def plot_y_pt(out=None, y_range=(-2.0, 2.0), pt_max=3.0) -> None
    def plot_eta_phi(out=None, eta_range=(-3.0, 3.0)) -> None
    def estimate_v2(pt_bins=None) -> Dict
    def plot_v2(out=None) -> None
```

---

## Algorithm Details

### Cumulative Variable Calculation

**Formula:**
$$x = \frac{E - p_z}{m_N}$$

Where:
- $E = \sqrt{p_x^2 + p_y^2 + p_z^2 + m^2}$ (total energy)
- $p_z$ = longitudinal momentum (beam direction)
- $m_N$ = 0.938 GeV (nucleon mass)

**Physical Interpretation:**
- $x = 1$: Single nucleon at rest carrying this momentum
- $x > 1$: Requires multiple nucleons to produce this momentum
- $x = 1.5$: Equivalent to 1.5 nucleons' momentum

### Multi-Event Aggregation Strategy

**Problem:** Combining statistics from 5000+ events with millions of particles

**Solution:** Running accumulation with numpy arrays

```python
# Initialize empty lists
self.pt_all = []
self.y_all = []
self.eta_all = []

# Per event:
def accumulate_event(particles):
    coll = CollisionAnalyzer(particles)
    self.pt_all.extend(coll.pt.tolist())  # Extend with all pT values
    self.y_all.extend(coll.y.tolist())
    self.eta_all.extend(coll.eta[np.isfinite(coll.eta)].tolist())
    
# Statistics:
def get_statistics():
    return {
        'pt': {
            'mean': np.mean(self.pt_all),
            'std': np.std(self.pt_all),
            'median': np.median(self.pt_all)
        },
        ...
    }
```

**Memory Usage:**
- ~100 bytes per particle
- 5000 events × 30 particles/event = 15M particles
- 15M × 100 bytes ≈ 1.5 GB total

**Optimization:** Batch processing keeps only 500×30 particles in memory at once.

### Parallel Processing Strategy

```python
with ProcessPoolExecutor(max_workers=4) as executor:
    # Each process analyzes one run independently
    futures = {
        executor.submit(analyze_run, run_num): run_num
        for run_num in [1, 2, 3, 4, 5]
    }
    
    # Results processed as they complete (not in order)
    for future in as_completed(futures):
        result = future.result()
        # Save results, print progress
```

**Why ProcessPoolExecutor?**
- Avoids Python's GIL (threads would serialize)
- Each run completely independent
- Automatic process pooling

---

## Design Improvements & Recommendations

### Current Strengths ✅

1. **Modular Architecture** — Clear separation of concerns
2. **Format Agnostic** — Handles Oscar1992A and OSC1997A
3. **Memory Efficient** — Batch processing prevents RAM overflow
4. **Parallel-Ready** — Process-based parallelism for true multi-core
5. **Auto-Detection** — Collision system inferred from data

### Recommended Improvements 🚀

#### 1. **Use `CollisionAnalyzer` Instead of `AggregateAnalyzer` for Initial Data Retrieval** ⭐ PRIMARY RECOMMENDATION

**Current Problem:**
- `AggregateAnalyzer` directly uses `CollisionAnalyzer` internally via `coll = CollisionAnalyzer(particles)`
- This is inefficient: `AggregateAnalyzer` should be **thin wrapper** accumulating pre-computed data from `CollisionAnalyzer`

**Recommended Refactor:**

```python
# CURRENT (run_all_analysis.py):
agg_mod = AggregateAnalyzer(system_label=...)
for event in reader.read_file():
    agg_mod.accumulate_event(event)  # Creates CollisionAnalyzer inside

# BETTER DESIGN:
collisions = []
for event in reader.read_file():
    coll = CollisionAnalyzer(event)
    collisions.append(coll)

# Then aggregate separately:
agg_mod = AggregateAnalyzer(system_label=...)
for coll in collisions:
    agg_mod.accumulate_precomputed(coll)  # Pass pre-computed kinematics
```

**Why?**
- Decouples **data analysis** from **aggregation**
- Enables reusing `CollisionAnalyzer` results for other analyses
- Easier testing and debugging

**Implementation:**
```python
class AggregateAnalyzer:
    def accumulate_precomputed(self, collision: CollisionAnalyzer) -> None:
        """Add pre-computed CollisionAnalyzer results"""
        self.pt_all.extend(collision.pt.tolist())
        self.y_all.extend(collision.y.tolist())
        # ... etc
```

---

#### 2. **Create High-Level `EventAnalysisPipeline` Class** 🔧

**Problem:** `run_all_analysis.py` is 500+ lines of orchestration logic

**Solution:** Abstract orchestration into reusable pipeline class

```python
class EventAnalysisPipeline:
    """High-level workflow for analyzing file pair"""
    
    def __init__(self, system_detector=None):
        self.system_detector = system_detector
        self.results = {}
    
    def analyze_pair(self, mod_file, unm_file=None):
        """
        Returns: {
            'collisions_mod': List[CollisionAnalyzer],
            'collisions_unm': List[CollisionAnalyzer],
            'aggregate_mod': AggregateAnalyzer,
            'aggregate_unm': AggregateAnalyzer,
            'comparison': RunComparisonAnalyzer,
            'cumulative': CumulativeEffectDetector,
            'plots': List[str]
        }
        """
        # 1. Detect system
        # 2. Read files
        # 3. Analyze events → CollisionAnalyzer list
        # 4. Accumulate → AggregateAnalyzer
        # 5. Compare → RunComparisonAnalyzer
        # 6. Detect cumulative → CumulativeEffectDetector
        # 7. Plot & save
        # 8. Return results dict
```

**Benefits:**
- Reusable for different workflows
- Testable in isolation
- Cleaner main script

---

#### 3. **Add Caching Layer for Expensive Computations** 📦

**Problem:** If re-running analysis on same data, recalculates everything

**Solution:** Optional pickle-based caching

```python
import pickle
from pathlib import Path

class CachedAnalyzer:
    def __init__(self, cache_dir='.cache'):
        self.cache_dir = Path(cache_dir)
    
    def analyze_file(self, filepath, use_cache=True):
        cache_file = self.cache_dir / f"{Path(filepath).stem}.pkl"
        
        if use_cache and cache_file.exists():
            with open(cache_file, 'rb') as f:
                return pickle.load(f)  # Load cached CollisionAnalyzer list
        
        # Otherwise compute and cache
        collisions = [self._analyze_event(e) for e in events]
        cache_file.parent.mkdir(exist_ok=True)
        with open(cache_file, 'wb') as f:
            pickle.dump(collisions, f)
        
        return collisions
```

---

#### 4. **Config File Support (YAML/TOML)** ⚙️

**Current:** Command-line arguments only

**Better:** Configuration file for reproducibility

```yaml
# config.yaml
data:
  root: /data
  modified_dir: modified
  unmodified_dir: no_modified

runs:
  ids: [1, 2, 3, 4, 5]

processing:
  workers: 4
  batch_size: 500

physics:
  cumulative_threshold: 1.1
  forward_cut: 0.3
  pt_broadening_threshold: 0.20

output:
  root: results
  formats:
    - json
    - png
```

**Usage:**
```bash
python run_all_analysis.py --config config.yaml
```

---

#### 5. **Structured Logging** 📋

**Current:** Print statements scattered throughout

**Better:** Python logging module with levels

```python
import logging

logger = logging.getLogger(__name__)

logger.info(f"Processing run_{run_num}")
logger.debug(f"Detected system: {system_info}")
logger.warning(f"No cumulative effects in run_{run_num}")
logger.error(f"File not found: {filepath}")
```

**Config:**
```python
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('analysis.log'),
        logging.StreamHandler()
    ]
)
```

---

#### 6. **Type Hints & Dataclasses** 🔍

**Current:** Some type hints, inconsistent

**Better:** Full typing + dataclasses for clarity

```python
from dataclasses import dataclass
from typing import Optional, List

@dataclass
class CollisionSystemInfo:
    label: str
    sqrt_s_NN: float
    A1: int
    Z1: int
    A2: int
    Z2: int

@dataclass
class AnalysisResult:
    run_name: str
    success: bool
    error: Optional[str]
    general_stats: Dict
    cumulative_analysis: Dict
    plots: List[str]

# Now analyzers can use clear type signatures:
def analyze_run(run_num: int) -> AnalysisResult:
    ...
```

---

#### 7. **Alternative Data Retriever: Direct SQL/HDF5** 💾

**For very large datasets (>1000 runs):**

Instead of reading ASCII `.f19` files repeatedly:

```python
# Convert once to HDF5:
import h5py

with h5py.File('collisions.h5', 'w') as f:
    for run_num, filepath in enumerate(run_files):
        events = OscarReader(filepath).read_file()
        group = f.create_group(f'run_{run_num}')
        
        for evt_idx, particles in enumerate(events):
            dataset = group.create_dataset(f'event_{evt_idx}', 
                data=np.array([
                    [p.px, p.py, p.pz, p.E, p.particle_id] 
                    for p in particles
                ]))

# Then retrieve instantly:
with h5py.File('collisions.h5', 'r') as f:
    particles = f['run_1/event_0'][:]  # 1000× faster than parsing ASCII
```

**Benefits:**
- ~100× faster I/O
- Structured storage
- Query-able with SQL backends
- Shareable compressed archives

---

#### 8. **Add Unit Tests** ✓

```python
# tests/test_cumulative_detector.py
import unittest

class TestCumulativeDetector(unittest.TestCase):
    def setUp(self):
        self.particles = [...]  # Mock particles
        self.detector = CumulativeEffectDetector(self.particles)
    
    def test_cumulative_variable_calculation(self):
        x = self.detector.calculate_cumulative_variable(
            px=1.0, py=0.5, pz=2.0
        )
        self.assertGreater(x, 0)
    
    def test_cumulative_threshold_detection(self):
        sigs = self.detector.detect_all_signatures()
        self.assertGreater(len(sigs), 0)
```

---

#### 9. **Visualization Dashboard (Optional)** 📊

**Instead of just PNG output:**

```python
# Generate interactive HTML dashboard with Plotly
import plotly.graph_objects as go
import plotly.subplots as sp

def generate_dashboard(run_results):
    fig = sp.make_subplots(rows=2, cols=2)
    
    # Add traces for each run
    for run, stats in run_results.items():
        fig.add_trace(
            go.Histogram(x=stats['pt_all'], name=run),
            row=1, col=1
        )
    
    fig.write_html('dashboard.html')
```

**Then open `dashboard.html` in browser for interactive exploration.**

---

### Summary: Priority-Ranked Improvements

| Priority | Improvement | Effort | Impact |
|----------|-------------|--------|--------|
| 🔴 High | Use `CollisionAnalyzer` in aggregation | 2h | **Decouples analysis from stats** |
| 🔴 High | Create `EventAnalysisPipeline` class | 3h | **Cleaner orchestration** |
| 🟡 Medium | Config file support | 1h | **Reproducibility** |
| 🟡 Medium | Structured logging | 1h | **Debugging** |
| 🟢 Low | Type hints + dataclasses | 2h | **Code clarity** |
| 🟢 Low | Unit tests | 2h | **Reliability** |
| 🔵 Optional | HDF5 caching | 4h | **For large datasets** |
| 🔵 Optional | Interactive dashboard | 3h | **Better UX** |

---

## Best Practices

### For Contributors

1. **Always use `CollisionAnalyzer`** for per-event analysis
2. **Pass pre-computed results** to `AggregateAnalyzer` instead of raw particles
3. **Document physics** in docstrings (equations, thresholds, references)
4. **Use numpy** for array operations (avoid Python loops)
5. **Add type hints** to new functions

### For Users

1. **Always run `verify_data_integrity.py` first**
2. **Start with small run counts** (e.g., `--runs 1 2 3`) before full batch
3. **Monitor memory** with large batch sizes: `--batch-size 250` if uncertain
4. **Save configurations** in YAML for reproducible analyses
5. **Archive JSON summaries** for long-term record-keeping

---

## References

- UrQMD: https://urqmd.org/
- Cumulative variables: XXXX (cite paper)
- NumPy documentation: https://numpy.org/doc/
- Python multiprocessing: https://docs.python.org/3/library/multiprocessing.html

