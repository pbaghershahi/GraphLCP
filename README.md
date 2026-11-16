# GraphLCP

Graph-aware conformal prediction. Supported methods:

| CLI method | Python class |
| --- | --- |
| `split_cp` | `SplitCP` |
| `rand_lcp` | `RandomlyLocalizedCP` |
| `graph_rlcp` | `GraphRLCP` |
| `cal_lcp` | `CalibratedLocalizedCP` |
| `test_cp` | `TestCP` |
| `cfgnn` | `CFGNN` |
| `aps` | `APS` |
| `daps` | `DAPS` |
| `snaps` | `SNAPS` |
| `naps` | `NAPS` |
| `graph_uniform_lcp` | `GraphUniformLCP` |
| `matched_rr_mondrian` | `MatchedRRMondrianComparison` |
| `scp_alpha_sweep` | `SCPAlphaSweep` |


## Setup

Use Python 3.12 and an isolated environment. The default installation uses CPU
PyTorch 2.5.0 and matching prebuilt PyG extensions:

```bash
python -m venv .venv
source .venv/bin/activate
bash installation.sh
python -c 'import main; from methods import *'
python main.py --help
```

To install CUDA 12.4 wheels instead, run the installer with matching indexes:

```bash
TORCH_WHEEL_INDEX_URL=https://download.pytorch.org/whl/cu124 \
PYG_WHEEL_URL=https://data.pyg.org/whl/torch-2.5.0+cu124.html \
bash installation.sh
```

## Run

Run commands from the repository root. Configurations start with an empty
`pretrained_path`, so they train a model instead of referring to existing checkpoints. CLI options override YAML settings.

```bash
python main.py -cff config/cora_ml.yaml -m graph_rlcp -cem wsc marginal -eid example
```

Datasets supported by PyG and the included GOOD adapters are downloaded on demand. Downloading datasets may require internet
access. Checkpoints, logs, and results are generated in `pretrained/`, `log/`, and
`outputs/`.

## Runtime layout

- `main.py`: training, conformal method selection, evaluation, and result output.
- `data_utils.py`, `good_datasets/`: dataset loading and splits.
- `model.py`, `good_models.py`, `train_utils.py`: GNN models and training.
- `methods/`: thirteen conformal methods and their shared runtime helpers.
- `eval/`: predictive evaluation and the bundled `covmetrics` library, including
  its upstream license and package metadata.
- `utils.py`, `metrics.py`: shared utilities and prediction-weight diagnostics.
- `config/`: dataset configurations.
