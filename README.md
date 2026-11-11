# PRISM - Provenance-Routed Inference via Specialist Models for Fake News Detection

## Reproducing LIFE: LLM-Generated Fake News Detection

**Python · PyTorch · HuggingFace Transformers · NLTK · Pandas · Scikit-learn**

An independent, end-to-end reproduction of the paper *"Prompt-Induced Linguistic Fingerprints
for LLM-Generated Fake News Detection"* (LIFE), built from the authors' released research code.

The released code did not run as published. I fixed it, made it runnable on a single
free-tier GPU, reimplemented the classifier, and matched the paper's reported results.

> Reproduction by **Hsien Zo Cheong** (University of Pittsburgh, B.S. Computer Science).
> The LIFE method and the original code belong to the paper's authors; this repo is a
> reproduction for study and verification.

---

## Results

| | Accuracy | F1 |
|---|---|---|
| LIFE paper (reported) | 93.7% | 92.4% |
| **This reproduction** | **93.4%** | **93.2%** |

Trained and evaluated on an 8,000-row LLM-generated fake-news dataset. Accuracy lands
within 0.3 points of the paper, and F1 comes in 0.8 points higher.

---

## What I did

- **Got the released code running.** It failed at import time and at run time. I fixed
  missing files, undefined variables, hardcoded author-machine paths, and deprecated
  library imports.
- **Removed the multi-GPU server dependency.** The original feature step needed a
  [mosec](https://github.com/mosecorg/mosec) inference server running several 2.7B–7B
  LLMs. I wrote `dataset/3_gen_features_local.py`, which computes the same per-token
  log-likelihood fingerprints in-process, so the whole pipeline fits on one T4.
- **Converted the raw data.** The public datasets ship as JSON dicts (`HF/HR/MF/MR.json`),
  not the JSONL format the pipeline expects. `dataset/0_convert.py` handles the conversion
  and the 4-class labeling.
- **Reimplemented and trained the classifier.** A Transformer + CRF sequence classifier over
  the perplexity fingerprints (`LIFE_train/`).
- **Packaged it as a Colab notebook.** Ran experiments on Google Colab.

