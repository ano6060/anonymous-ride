# Installation

RIDE requires Python 3.11.

All commands in the tutorials assume they are run from the repository root.

## Create an Environment

```bash
conda create -n ride python=3.11
conda activate ride
```

## Install Dependencies

From the repository root:

```bash
pip install -r requirements.txt
```

PyTorch installation can depend on the local CUDA setup. If the default
installation does not match your hardware, follow the official PyTorch
installation selector and then rerun the command above for the remaining
dependencies.

## Check the Installation

```bash
python -c "import pandas, pyarrow, torch, torch_geometric; print('RIDE environment ready')"
```

[Back to README](../README.md#what-do-you-want-to-do)
