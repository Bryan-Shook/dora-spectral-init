# Setup

## PACE (VS-Code Version)

Navigate to [PACE-ICE](https://ondemand-ice.pace.gatech.edu/pun/sys/dashboard). **You must be on GT VPN in order to connect to PACE**. After navigating to the dashboard, click on "Microsoft VS Code" to launch an interactive VS Code session. 

Settings: (as of 10/12/2026)
- **VS Code Version**: 1.99.2
- **Environment Setup**: Default Modules
- **Quality of Service**: Default or COC-ICE
- **Node Type**: 
  - First Available with CUDA 12 or 13 (Random Testing)
  - H100 (Experiments)
- **Number of Cores (CPUs)**: 8
- **Number of GPUs**: 1
- **Memory (GB)**: 128
- **Number of Hours**: Use-case

Press **Launch** and wait for your turn in queue. After the session says "Running", click on the "Connect" button. 

You will start in /home/hice1/GTID. We will use the scratch directory which has extra memory.
```bash
cd scratch/
git clone https://github.com/Bryan-Shook/dora-spectral-init.git
```

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
