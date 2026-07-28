# Repository Guidelines

## Project overview

H-Net is a Python byte-level language-model training project. It is primarily used
for Japanese-data experiments in CUDA/Colab environments. Core library code lives
in `hnet/`; the repository-root Python files are command-line entry points for
training, validation, generation, dataset preparation, inspection, and plotting.

## Layout

- `hnet/models/`: H-Net model and configuration loading.
- `hnet/modules/`: neural-network building blocks.
- `hnet/training/`: datasets, configuration, training, and validation logic.
- `hnet/sft/`: supervised fine-tuning support.
- `configs/`: model JSON configurations.
- `tests/unit/`: lightweight unit tests.
- `scripts/`: evaluation and data-preparation utilities.
- `results/`, `artifacts/`, and checkpoint/data files: experiment outputs; treat
  these as generated or user-owned unless a task explicitly targets them.
- `llm-jp-eval/`: an ignored external/nested project; do not modify it as part of
  ordinary H-Net changes.

## Environment and commands

Use Python 3.9 or newer. The full install includes GPU-specific packages such as
PyTorch, Mamba SSM, FlashAttention, Triton, and causal-conv1d; consult `README.md`
for the supported CUDA/Colab setup. Do not reinstall or upgrade these dependencies
unless the task requires it.

Common checks and workflows:

```sh
python -m pytest tests/unit
python train.py --help
python validate.py --help
python generate.py --help
python prepare_packed_dataset.py --help
```

Training, validation against remote datasets, checkpoint generation, and packed
dataset preparation may require a GPU, network access, substantial disk space, or
hours of runtime. Do not use them as routine verification. Prefer focused unit
tests and small synthetic inputs.

## Code conventions

- Follow the existing PEP 8-style formatting and four-space indentation.
- Add type annotations to new public functions and non-obvious data structures.
- Use `pathlib.Path` for filesystem paths and explicit UTF-8 encoding for text.
- Keep CLI argument parsing in the entry-point script and reusable behavior in
  the `hnet` package.
- Use the frozen dataclasses in `hnet/training/config.py` for training and dataset
  settings. Preserve backward compatibility when loading saved JSON configs;
  historical artifacts can contain unknown fields.
- Avoid broad refactors in model, chunking, or training code when making a focused
  fix. Numerical behavior, tensor shapes, dtypes, and device placement are part of
  the interface.

## Tests

Add focused tests under `tests/unit/` and mirror the package area being changed.
Tests should be deterministic, CPU-friendly, and independent of Hugging Face
downloads, CUDA availability, and existing experiment artifacts. Use pytest's
`tmp_path` for filesystem behavior.

Run the narrowest relevant test first, then `python -m pytest tests/unit` when the
local environment has the project's compiled dependencies. If imports fail because
CUDA extensions are unavailable, report that limitation rather than changing
production imports solely to accommodate the local machine.

## Data and repository hygiene

- Never commit checkpoints, packed binary shards, downloaded datasets, secrets,
  caches, or virtual environments.
- Preserve unrelated working-tree changes and untracked research notes/artifacts.
- Do not overwrite experiment CSV/JSON output merely to validate a code change.
- Update `README.md` when changing a user-facing CLI, output format, or documented
  training workflow.
