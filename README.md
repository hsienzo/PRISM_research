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

LIFE separates LLM-written fake news from LLM-written real news well, but it fails
once human-written articles are mixed in. On the full GossipCop++ benchmark
(human-fake, human-real, machine-fake, machine-real), LIFE caught **0% of
human-written fake news** (0% precision, 0% recall). It learns whether text was
machine-generated, not whether it is true.

PRISM's first stage is a **RoBERTa provenance router** that labels each article
as human-written or LLM-generated.

| Router (GossipCop++) | Accuracy |
|---|---|
| Human vs LLM | **95.01%** |

5.8% of human-written articles (711 of 12,252) are misrouted to LIFE.

## Human written fake news classification

Articles the router labels as human-written go to a **fine-tuned BERT** classifier
trained on veracity within human-written news. Articles labeled LLM-generated go
to LIFE. Each branch is scored on the articles it actually receives from the router:

| Branch | GossipCop++ (Acc / F1) | PolitiFact++ (Acc) |
|---|---|---|
| Human-written → BERT | 80.78 / 77.92 | 80.56 |
| LLM-generated → LIFE | 89.44 / 89.42 | 77.71 |

On GossipCop++, the human branch is the weaker half. The PolitiFact++ reversal
isn't reliable: with only 520 articles, LIFE's branch varies by ±17 points
across seeds.

## PRISM results

Cross-validated, so every one of GossipCop++'s 20,505 articles is scored once by
a model that never trained on it. Fake = human-fake + machine-fake; real =
human-real + machine-real.

| System | GossipCop++ (Acc / F1) | PolitiFact++ (Acc / F1) |
|---|---|---|
| LIFE alone | 76.3 / 72.0 | 71.9 / 65.2 |
| **PRISM** | **84.4 / 83.6** | **80.0 / 77.8** |
| BERT alone (no routing) | 85.2 / 84.7 | 85.0 / 83.6 |

PRISM beats LIFE alone by 8–9 accuracy points, but it doesn't beat a single BERT
classifier trained on everything. The gap comes from the weaker human branch and
from the router's misroutes, which send human-written fakes to LIFE, which cannot
detect them.

## Dataset
<https://github.com/mbzuai-nlp/Fakenews-dataset>
