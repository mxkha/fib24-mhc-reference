# Fib24+mHC Reference Implementation

A mathematically rigorous implementation of **Manifold-Constrained Hyper-Connections (mHC)** with **Fib24 temporal scheduling**, combining two research papers into a unified PyTorch module.

**Status:** Complete | **Format:** Jupyter Notebook | **License:** No License

---

## Overview

This repository contains a comprehensive Jupyter notebook that implements the complete **Fib24+mHC** system:

1. **mHC (Manifold-Constrained Hyper-Connections)** from arXiv:2512.24880 [Xie et al., 2024]
2. **Fib24 Mandelbrot Set** (collapsed Fibonacci modulo 9 dynamics) [Mahi, 2026]

The implementation strictly separates **PAPER FACTS** (with exact citations), **DERIVED** constructs, and **ENGINEERING GUESS** defaults, ensuring mathematical rigor and transparency.

### Key Features

- ✓ **Self-contained**: All paper facts embedded—no external PDFs required
- ✓ **Fully tested**: 7 comprehensive verification tests covering all mandatory requirements
- ✓ **Production-ready**: Complete mHC layer with Sinkhorn-Knopp projection
- ✓ **Fib24 scheduling**: 24-cycle door scheduling with attractor classification
- ✓ **Protocol ready**: 4-stream state system for turn-based interaction
- ✓ **Interactive**: Run, modify, and experiment directly in Jupyter

---

## Installation

### Prerequisites

- **Python 3.8+**
- **PyTorch 1.13+** (CPU or GPU)
- **NumPy**
- **Jupyter Notebook** or **Jupyter Lab** (or Google Colab)

### Step 1: Clone the Repository

```bash
git clone https://github.com/mxkha/fib24-mhc-reference.git
cd fib24-mhc-reference
```

### Step 2: Install Dependencies

**Option A: Using pip**

```bash
pip install torch numpy jupyter
```

**Option B: Using conda**

```bash
conda create -n fib24-mhc python=3.11
conda activate fib24-mhc
conda install pytorch numpy jupyter -c pytorch
```

**Option C: Google Colab (No Installation Needed)**

Simply upload the notebook to Colab and run—PyTorch and Jupyter are pre-installed.

### Step 3: Launch the Notebook

**Local Jupyter:**

```bash
jupyter notebook Fib24_mHC_Reference.ipynb
```

**Jupyter Lab:**

```bash
jupyter lab Fib24_mHC_Reference.ipynb
```

**Google Colab:**

1. Go to [Google Colab](https://colab.research.google.com)
2. Click "File" → "Upload notebook"
3. Select `Fib24_mHC_Reference.ipynb`
4. Run all cells

---

## Quick Start

### 1. Run All Cells

Once the notebook is open, press **Ctrl+A** (or **Cmd+A** on Mac) to select all cells, then **Shift+Enter** to run them.

The notebook will:
- Display all mathematical specifications
- Implement the mHC layer and Fib24 scheduler
- Run 7 comprehensive verification tests
- Initialize the protocol state

### 2. Expected Output

You should see:
```
============================================================================
Test 1: MHC Layer Without Fib24 Scheduling
============================================================================

Input shape: torch.Size([2, 8, 4, 64])
Output shape: torch.Size([2, 8, 4, 64])
Forward gain: 1.0234
Backward gain: 1.0234

Doubly-stochastic check:
  Row sums (should be ~1): min=0.999999, max=1.000001
  Col sums (should be ~1): min=0.999999, max=1.000001

[All 7 tests pass]
```

### 3. Modify and Experiment

Edit any cell to:
- Change layer dimensions
- Experiment with different input shapes
- Add custom analysis
- Visualize the Fib24 cycle
- Implement your own residual functions

---

## Notebook Structure

### Part 1: Mathematical Specification
- Complete mHC equations (Eq. 1, 3, 6, 7, 8, 9) with citations
- Complete Fib24 equations (Eq. 8, 12) with citations
- 24-cycle sequence and door attractor classification

### Part 2: Full Implementation
- `DoorMatrixMapper`: Maps doors to doubly-stochastic matrices
- `SinkhornKnopp`: Sinkhorn-Knopp projection (Eq. 9, t_max=20)
- `RMSNorm`: Root Mean Square normalization (Eq. 7)
- `MHCLayer`: Complete mHC architecture (Eq. 3, 7, 8)
- `Fib24Scheduler`: Protocol state management

### Part 3: Verification Suite
- 7 comprehensive tests covering all mandatory requirements
- Shape preservation, doubly-stochastic property, door bookkeeping
- Numerical stability, backpropagation
- All tests pass ✓

### Part 4: Phase 3 Protocol
- 4-stream state system (S0-S3) initialized
- Fib24 scheduler ready for turn-based interaction
- 24-turn cycle visualization

---

## Usage Examples

### Example 1: Basic MHC Layer

```python
# Initialize layer
layer = MHCLayer(C=64, n=4, use_fib24_scheduling=False)

# Create input: (batch_size, seq_len, n_streams, feature_dim)
x = torch.randn(2, 8, 4, 64)

# Forward pass
x_out, diagnostics = layer(x, return_diagnostics=True)

print(f"Input shape: {x.shape}")
print(f"Output shape: {x_out.shape}")
print(f"Forward gain: {diagnostics['forward_gain']:.4f}")
```

### Example 2: MHC with Fib24 Scheduling

```python
# Initialize with Fib24 scheduling
layer = MHCLayer(C=64, n=4, use_fib24_scheduling=True)

# Forward passes use different doors from Fib24 cycle
for turn in range(24):
    x = torch.randn(2, 8, 4, 64)
    x_out, diag = layer(x, return_diagnostics=True)
    
    print(f"Turn {turn}: door={diag['used_door']}, next_door={diag['next_door']}")
```

### Example 3: Door→Matrix Mapping

```python
# Create mapper
mapper = DoorMatrixMapper(n=4)
mapper.to('cpu')

# Get doubly-stochastic matrix for each door
for door in range(1, 10):
    M = mapper.get_matrix(door)
    print(f"Door {door}:\n{M}")
```

---

## Paper References

### Paper 1: mHC: Manifold-Constrained Hyper-Connections

**Citation:** Xie, Z., Wei, Y., Cao, H., et al. (2024). *mHC: Manifold-Constrained Hyper-Connections*. arXiv:2512.24880.

**Key Contributions:**
- Extends Hyper-Connections by projecting residual matrices onto the Birkhoff polytope
- Restores identity mapping property for stable large-scale training
- Uses Sinkhorn-Knopp algorithm for doubly-stochastic projection
- Achieves 6.7% overhead with expansion rate n=4

### Paper 2: Fib24 Mandelbrot Set

**Citation:** Mahi, K. (2026). *Fib24 Mandelbrot Set*. Unpublished manuscript.

**Key Contributions:**
- Defines digital root collapse of Fibonacci sequence modulo 9
- Produces a 24-cycle door schedule
- Maps each door to an attractor (fixed point, 2-cycle, or 3-cycle)
- Provides categorical memory for finite-state dynamics

---

## Verification Checklist

All mandatory requirements are verified in the notebook:

- [x] **Shape Preservation**: Input and output have same shape
- [x] **Doubly-Stochastic**: $\mathbf{H}_{res}$ has row sums = 1, column sums = 1
- [x] **Door Bookkeeping**: used_door matches applied matrix
- [x] **Fib24 Cycle**: Correct 24-cycle sequence
- [x] **Attractor Classification**: Correct cycle/fixed point types
- [x] **Numerical Stability**: No NaNs or Infs with extreme inputs
- [x] **Backpropagation**: Gradients flow correctly

---

## Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'torch'"

**Solution:** Install PyTorch
```bash
pip install torch
```

### Issue: "Notebook won't open"

**Solution:** Ensure Jupyter is installed
```bash
pip install jupyter
```

### Issue: "CUDA out of memory"

**Solution:** Use CPU instead
```python
device = 'cpu'  # Add this to cells that create tensors
```

### Issue: "Tests fail with NaN values"

**Solution:** Ensure you're using PyTorch 1.13+ and NumPy is up to date
```bash
pip install --upgrade torch numpy
```

---

## Contributing

This is a reference implementation. If you find issues or have suggestions, feel free to:

1. Open an issue on GitHub
2. Submit a pull request with improvements
3. Fork and extend for your own research

---

## Citation

If you use this implementation in your research, please cite both papers:

```bibtex
@article{xie2024mhc,
  title={mHC: Manifold-Constrained Hyper-Connections},
  author={Xie, Z. and Wei, Y. and Cao, H. and others},
  journal={arXiv preprint arXiv:2512.24880},
  year={2024}
}

@unpublished{mahi2026fib24,
  title={Fib24 Mandelbrot Set},
  author={Mahi, K.},
  year={2026}
}
```

---

## Repository

**GitHub:** https://github.com/mxkha/fib24-mhc-reference

---

## Questions?

For issues, questions, or suggestions:
- **Open an issue** on the [GitHub repository](https://github.com/mxkha/fib24-mhc-reference/issues)
- **Email:** khadar.mahi@hotmail.com

Feel free to reach out with any questions, feedback, or collaboration inquiries.

---

**Last Updated:** January 3, 2026  
**Status:** Complete and tested ✓
