# PRISM - Provenance-Routed Inference via Specialist Models for Fake News Detection

## Reproducing LIFE: LLM-Generated Fake News Detection

**Python · PyTorch · HuggingFace Transformers · NLTK · Pandas · Scikit-learn**

An independent, end-to-end reproduction of the paper *"Prompt-Induced Linguistic Fingerprints
for LLM-Generated Fake News Detection"* (LIFE), built from the authors' released research code.

The released code did not run as published. I fixed it, made it runnable on a single
free-tier GPU, reimplemented the classifier, and matched the paper's reported results.

> Reproduction by **Hsien Zo Cheong** (University of Pittsburgh, B.S. Computer Science).
> The LIFE method and the original code belong to the paper's authors <https://arxiv.org/abs/2508.12632>. This repo is a
> reproduction for study, verification and experiments.

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
  LLMs. I wrote `3_gen_features_local.py`, which computes the same per-token
  log-likelihood fingerprints in-process.
- **Converted the raw data.** The public datasets ship as JSON dicts (`HF/HR/MF/MR.json`),
  not the JSONL format the pipeline expects. 
- **Reimplemented and trained the classifier.** A Transformer + CRF sequence classifier over
  the perplexity fingerprints.
- **Packaged it as a Colab notebook.** Ran experiments on Google Colab.

## AI vs human text classification

LIFE works well on LLM-generated fake news, but it breaks down once human-written
articles are mixed in. When I pooled human and machine articles in PolitiFact++,
accuracy dropped from 86.8% to 71.9%. A four-class breakdown showed why: the
detector caught **0% of human-written fake news**. It had learned whether text was
machine-generated, not whether it was true.

PRISM separates those two questions. Its first stage is a router, a fine-tuned
[RoBERTa / BERT] classifier that predicts whether an article was written by a human
or an LLM.

| Router | Accuracy | F1 |
|---|---|---|
| Human vs LLM | [x%] | [x%] |

## Human written fake news classification

Articles the router labels as human-written go to a separate classifier trained
only on human-written real and fake news. Articles labeled as LLM-written go to the
LIFE-based fingerprint classifier.

| Specialist | Model | Accuracy | F1 |
|---|---|---|---|
| Human-written | [RoBERTa / BERT] | [x%] | [x%] |
| LLM-written | LIFE (Transformer + CRF) | [x%] | [x%] |

## PRISM results

End to end, router errors carry through to the specialists, so this is the number
that matters. On 20,000+ articles across human and LLM sources:

| System | Accuracy | F1 |
|---|---|---|
| LIFE alone (pooled human + LLM) | [x%] | [x%] |
| **PRISM** | **84.4%** | **83.6%** |

## Dataset
<https://github.com/mbzuai-nlp/Fakenews-dataset>
