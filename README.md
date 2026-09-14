# VFM

Course project exploring vision foundation models (VFMs) for computational pathology.
Runs pathology image patches through pretrained encoders --
[UNI2-h](https://huggingface.co/MahmoodLab/UNI2-h) and
[Virchow2](https://huggingface.co/paige-ai/Virchow2) -- and studies the resulting
embeddings across three datasets:

- **PathMNIST** -- fast iteration / on-ramp (~200 MB, auto-download)
- **CAMELYON17-WILDS** -- domain shift / stain-normalization (~11 GB, auto-download)
- **PANDA** -- ISUP grading, multi-institution generalization (~350+ GB raw WSIs,
  requires a Kaggle account and competition rules acceptance)

## Setup

```bash
pip install -r requirements.txt
pip install -e .
pre-commit install
```

The repo root is marked by `.project-root`, used by `pyrootutils` in the notebooks to
resolve `src/` regardless of where the Jupyter kernel's working directory ends up.

## Structure

```
notebooks/
  00_data_download/   download and unpack each dataset
  01_exploration/      inspect datasets and metadata before encoding
  02_encodings/         run patches through UNI2-h / Virchow2 and save embeddings
src/
  vfm_encoders.py      shared encoder loading + embedding helpers
  dataloader/           dataset-specific loading utilities
```

Notebooks are meant to be run in order (`00` -> `01` -> `02`) per dataset. Each
`02_encodings` notebook saves per-model `.npy` embeddings plus a metadata CSV
alongside the source dataset.

## Notes

- UNI2-h and Virchow2 are gated on Hugging Face -- request access on each model page,
  then set `HF_TOKEN` (or log in interactively when prompted).
- Run the download and encoding notebooks on a server with adequate disk (PANDA) and
  a GPU (encoding); CPU works for small smoke tests only.

## License

GPL-3.0 -- see [LICENSE](LICENSE).
