# Reference outputs

`rigor_grid_v2.jsonl` is the authoritative grid behind the paper's headline results
(the confounder-controlled **null**, the IG-ceiling **calibration**, the synthetic
**positive control**, and the threshold sweep): 4 explainers x 3 datasets x up to 10
trained-model seeds x swept k, faithful PGExplainer (100 epochs). Per-seed fields:
`stat`, `logic_tree`, `logic_onerule`, `logic_conj`, with seed-clustered CIs.

`regrid_eer.log` / `regrid_cross.log` hold the reliability results (cross-explainer
consensus-agreement vs faithfulness; the fidelity comparison; the certified core).

Regenerate everything with the scripts in `../experiments/` (see the README's
"Reproduce the paper results" table). All runs are CPU-only and deterministic given
the fixed seeds.
