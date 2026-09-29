# A Confounder-Controlled Self-Consistency Diagnostic for GNN Explanations

Open code artifact for the paper *"Statistically Consistent, Logically Contradictory? A
Confounder-Controlled Self-Consistency Diagnostic for GNN Explanations"* (under review).

The tool audits a post-hoc GNN-classification explainer on **two axes at once**: a **statistical**
axis (chance-corrected top-*k* mask overlap and Kendall's τ across seeds) and a **logical** axis
(non-contradiction of the per-seed *class rules* the explanations induce over a fixed motif
vocabulary, decided exactly by enumerating the `2^{|V|}` concept assignments). It controls two
confounders that otherwise manufacture apparent dissociations — the capacity of the rule aggregator
and the trained-model seed — and reports results under **three aggregators** (decision tree, OneR,
greedy conjunction), replicated across model seeds with seed-clustered bootstrap intervals.

## Install

```bash
python3.10 -m venv .venv && source .venv/bin/activate
pip install -e ".[explainers]"     # core + torch / torch-geometric / captum for the real explainers
# optional: pip install -e ".[dev]" for the test suite
```
CPU is sufficient: the models are small GINs on small graphs. No GPU is required.
The `corels` extra is **optional and off by default** — the canonical aggregator is the pure-Python
decision tree, so no native build is needed to reproduce any reported number.

## Data (not bundled — links only)

- **BA-2Motifs**, **BAMultiShapes** — synthetic, generated on the fly by `gnnxc/data.py`
  (planted-motif ground truth included). Nothing to download.
- **MUTAG** — from the TUDataset collection, fetched by `torch-geometric`
  (<https://chrsmrrs.github.io/datasets/>).

## Reproduce the paper results

All scripts are CPU-only, deterministic given the fixed seeds, and write committed logs under
`outputs/`. Run from the repo root with the package on the path:

```bash
PYTHONPATH=. python experiments/<script>.py
```

| Script | Paper result |
|---|---|
| `experiments/rigor_grid_v2.py` | Real-data grid: the confounder-controlled **null** (cross-seed logical consistency under 3 aggregators × up to 10 model seeds), the IG-ceiling **calibration** (specificity), and the threshold sweep. |
| `experiments/synthetic_positive_control.py` | **Sensitivity** positive control: a planted dissociation is recovered (statistical 0.898 vs logical 0.483, robust to the aggregator). |
| `experiments/rigor_cross.py` | **Sensitivity** on real data: audited explainers vs the random floor (cross-explainer). |
| `experiments/rigor_manifold.py` | **Manifold-restricted** logical consistency λ^man vs the full-cube reading. |
| `experiments/ees_vocab.py` | **Vocabulary-robustness** of the null (three motif vocabularies). |
| `experiments/rigor_eer.py` | **Evidential reliability**: consensus-agreement vs faithfulness, the fidelity comparison, and the certified-core precision/recall. |
| `experiments/ees_loeo.py` | **Crowd-robustness** of consensus-agreement (leave-one-explainer-out). |
| `experiments/illustrative_e2e.py` | The end-to-end illustrative walkthrough on MUTAG / PGExplainer. |

## Package layout

```
gnnxc/
  data.py            synthetic generators + MUTAG loader (with motif ground truth)
  model.py           GIN classifier + trainer
  explainers/        GNNExplainer / PGExplainer / IG adapters (skip-if-import) + a mock explainer
  pipeline.py        the shared per-explainer, process-isolated audit pipeline
  project/           top-k thresholding, subgraph-monomorphism concept projection, fixed vocabulary
  aggregate/rule.py  the three deterministic rule aggregators (tree / OneR / conjunction)
  layers/            statistical and logical consistency (incl. manifold-restricted non-contradiction)
  report/            determinism/isolation certificate, confounder verdict, dissociation plane
```

## Tests

```bash
PYTHONPATH=. pytest -q          # MockExplainer-first; validates the pipeline with no real compute
```

## License

Released for review under the MIT License (see `LICENSE`).
