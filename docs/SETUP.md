# Setup

## Prerequisites

- Python 3.11
- [uv](https://docs.astral.sh/uv/getting-started/installation/)
- A local copy of this repository

Run all commands from the repository root.

## Install dependencies

Choose one installation for your environment. Both commands create `.venv`
and install the locked dependencies, development tools, and notebook packages.

### CPU

```bash
uv sync --locked --extra cpu --extra models --group notebooks
```

### NVIDIA GPU (Linux)

The GPU installation uses the PyTorch CUDA 12.6 build and requires a compatible
NVIDIA driver.

```bash
bash scripts/setup_pace.sh cuda
```

## Verify the installation

Run the tests:

```bash
.venv/bin/python -m pytest
```

Run the CPU smoke test:

```bash
.venv/bin/python -m dora_spectral.smoke --device cpu --output runs/local-smoke/smoke.json
```

For a GPU installation, run the GPU smoke test on a GPU node, inside a compute
allocation when using PACE ICE:

```bash
.venv/bin/python -m dora_spectral.smoke --device cuda --output runs/gpu-smoke/smoke.json
```

## Open the notebook

```bash
.venv/bin/jupyter lab notebooks/01_initialization_checks.ipynb
```

To register the environment as a kernel for an existing Jupyter installation:

```bash
.venv/bin/python -m ipykernel install --user \
  --name dora-spectral --display-name "DoRA spectral"
```
