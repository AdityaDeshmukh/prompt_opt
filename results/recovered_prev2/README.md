# Recovered pre-v2 per-lambda evals

These five JSONs are the **only per-lambda evaluation files that still exist
anywhere**. Every `outputs.step.*.json` and `test/output.json` under
`/scratch/ad11/prompt_opt/outputs` was destroyed by the scratch purge around
2026-09-03..05 (see `analysis/recover_from_wandb.py`), and wandb never held the
per-lambda arrays because `ScoreTrainer.evaluate()` averages over the lambda
grid before logging.

**Where they came from**: VSCode's local file history
(`~/.vscode-server/data/User/History/`), which snapshots files opened in the
editor. Recovered 2026-09-08. Original paths, from each snapshot's
`entries.json`:

| file here | original path | note |
|---|---|---|
| `new_l1_scaled_test_output.json` | `/u/ad11/prompt_opt/new_l1_scaled/test/output.json` | **full test-split eval**, converged; labelled "R-REBEL ($\ell_1$)" by `plot2.py` |
| `grpo_eval_step540.json` | `/u/ad11/scratch/prompt_opt/grpo/eval/outputs.step.540.json` | train-time eval, step 540 only |
| `drgo_explore_eval_step865.json` | `/u/ad11/scratch/prompt_opt/drgo_explore/eval/outputs.step.865.json` | train-time eval |
| `drgo_explore_eval_step545.json` | `/u/ad11/scratch/prompt_opt/drgo_explore/eval/outputs.step.545.json` | train-time eval |
| `drgo_mse_eval_step1700.json` | `/u/ad11/scratch/prompt_opt/drgo_mse/eval/outputs.step.1700.json` | train-time eval |

## Do not use these as results

They are **2025, pre-v2 runs**: before the fairness fixes (retry-loop gradient
accumulation, inconsistent reward scaling, NaN-unguarded std), at wildly
unmatched training steps, and from a code path that has since been refactored
twice. They are not comparable to each other and they are not the v3 campaign.

What they *are* good for, and all `analysis/paper_results.py` uses them for
(`fig_recovered_frontier`), is establishing the **shape** of the target figure:
expected sentiment against expected content, one point per lambda
(tau = 100*lambda), for a single conditioned policy. The
`new_l1_scaled` file traces a clean monotone frontier from (25, 88) at tau=0 to
(72, 16) at tau=90. The GRPO file is step 540 — far too early to trace anything
— which is precisely why it cannot substitute for the real comparison.

The v3-comparable frontier comes from `slurm/train_v4.slurm` ->
`slurm/eval_v4.slurm`; `paper_results.py` switches the paper's figure over
automatically once those land.
