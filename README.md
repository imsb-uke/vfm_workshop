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
  00_data_download/       download and unpack each dataset
  01_exploration/         inspect datasets and metadata before encoding
  02_encodings/           run patches through UNI2-h / Virchow2 and save embeddings
  03_hands_on_encoding/   encoder internals (tokens, pooling, model options) and a
                          from-scratch mini encoding run, before the full pipeline
  04_encoding_analysis/   probe the saved embeddings: linear/kNN classification,
                          domain-origin leakage, patch-level similarity
src/
  vfm_encoders.py      shared encoder loading + embedding helpers
  probing.py            shared linear/kNN probing helpers for 04_encoding_analysis
  patch_similarity.py   patch-token cosine-similarity helpers for 04_encoding_analysis
  dataloader/           dataset-specific loading utilities
```

Notebooks are meant to be run in order (`00` -> `01` -> `02` -> `03` -> `04`) per
dataset. `03_hands_on_encoding` is a first look at the encoders themselves -- run
once, using PathMNIST, before `02_encodings` automates the same steps at full-dataset
scale for all three datasets. Each `02_encodings` notebook saves per-model `.npy`
embeddings plus a metadata CSV under `~/data/<dataset>/encodings` (your own home
directory, not the shared dataset volume -- source datasets are read from
`/home/shared/data/...`, but encodings are per-user output); `04_encoding_analysis`
notebooks load those files back in from the same location (most `02_encodings` runs
are parameterized -- subset size, sample size -- so the `04` notebooks glob for the
most recently produced file rather than a fixed name). Two of the six
`04_encoding_analysis` notebooks (patch similarity / cross-image retrieval) instead
load the encoders directly, since they need the raw spatial patch tokens rather than
the pooled vectors `02_encodings` saves.

## Notes

- UNI2-h and Virchow2 are gated on Hugging Face -- request access on each model page,
  then set `HF_TOKEN` (or log in interactively when prompted).
- Datasets live on the shared volume (`/home/shared/data/...`); encodings you produce
  are saved under your own home directory (`~/data/<dataset>/encodings`) so they don't
  collide with other users' runs or consume shared disk.
- Run the download and encoding notebooks on a server with adequate disk (PANDA) and
  a GPU (encoding); CPU works for small smoke tests only.

## License

GPL-3.0 -- see [LICENSE](LICENSE).
