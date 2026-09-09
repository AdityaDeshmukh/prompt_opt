# Paper draft — R-REBEL vs GRPO

Draft started 2026-09-08. **Scope of this draft: the algorithms and the results
only**, as requested. Intro is thin on purpose, related work is a stub, and
there is no discussion/conclusion yet.

## Build

```bash
cd paper && make          # pdflatex x2 -> main.pdf (14 pages)
make results              # regenerate tables+figures from data, then build
make clean
```

Toolchain on this cluster: `pdflatex` and `bibtex` are present; `latexmk`,
`algorithm2e`, `algpseudocode` and `pgfplots` are **not**. Hence
`algorithm`+`algorithmic` for pseudocode and pre-rendered matplotlib PDFs for
every figure. Don't add a package without checking `kpsewhich <pkg>.sty` first.

## Where the numbers come from

Nothing here is typed by hand. Everything flows from one committed data file:

```
results/v3_wandb_export.json        <- analysis/recover_from_wandb.py  (needs network)
  -> paper/tables/*.tex             <- analysis/paper_results.py --tex
  -> paper/figures/*.pdf            <- analysis/paper_results.py
```

`analysis/paper_results.py` also prints the tables to stdout with the pairwise
gaps and a noise-floor verdict on each, which is the quickest way to sanity-check
a claim in the text.

## Read this before trusting a number

The v3 campaign completed, but `/scratch` was purged around 2026-09-03..05 and
**every checkpoint and every per-lambda eval JSON is gone** (0 `*.pth`,
0 `outputs.step.*.json`). Consequences, all disclosed in
`sections/results.tex` §"Data provenance":

1. **Scores are solid.** Recovered from wandb and validated three ways per run
   (true-step anchors, distinct-step count, perfectly regular step grid). All 15
   runs pass; the script refuses to emit data if a check fails.
2. **The lambda-frontier does not exist yet.** `evaluate()` averages over the
   lambda grid *before* logging (`trainers/score_trainer.py:282-284`), so wandb
   never had per-lambda content/style. Figure `tradeoff.pdf` currently shows
   lambda-**averaged operating points** and says so on its face. It is not a
   frontier — do not describe it as one.
3. **Step 11500, not 12000, is the headline.** Three runs (`*_seed1` of
   rrebel_huber_std, rrebel_l1_ent, grpo_ent) lack a step-12000 eval: their last
   cycle ran under the trainer's wandb-init fallback, so those evals went to
   disk only and the disk copies were purged. 11500 is the largest step where
   all 15 runs are present.
4. **Distinct-prompt collapse counts are not quoted as measured.** They came
   from the purged eval JSONs. The surviving `num_tokens_explored` metric cannot
   substitute — it is a per-process cumulative set that restarts on every 4-hour
   resubmit, so its peak reflects early exploration, not steady-state collapse.
5. **No pre-v2 data goes in the paper.** Five per-lambda eval JSONs recovered
   from editor history live in `results/recovered_prev2/`, but they are 2025 runs
   from before the fairness fixes. `analysis/paper_results.py` will only plot
   them under an explicit `--recovered` flag, into a filename the paper does not
   include. Do not re-add them.

## Getting the frontier figure

`slurm/train_v4.slurm` retrains one seed per arm (3 runs x 12000 steps:
rrebel_l1_std, grpo_ent, grpo_baseref) purely to regenerate the per-lambda
evals; its EXIT trap auto-submits `slurm/eval_v4.slurm` per arm on completion,
which writes `results/v4/<run>/test/output*.json` -- the **500-sentence test
split**, fp32 scorers, fixed vLLM seed. `paper_results.py` detects those files
and switches `tradeoff.pdf` from operating points to the true frontier
automatically, printing the step it drew. No text change is needed beyond
removing the TODO in the tradeoff subsection and the caveat in the limitations.

`eval_v4.slurm` evaluates the **newest available checkpoint** when step 12000 is
absent, so a real test-split frontier can be produced mid-training -- submit it
by hand once checkpoints exist. It never reads the train-time
`eval/outputs.step.*.json` files, and neither does the figure code: those are
the 10-sentence DEV evals, not test.

Both v4 scripts archive to `results/v4/` in **home** after every cycle — that is
the fix for the root cause of the data loss above. Nothing a paper depends on
should ever live only on `/scratch`.

## Open TODOs in the draft

`grep -rn 'TODO' sections/ main.tex` — currently:

- populate `refs.bib` and convert by-name mentions to `\citep`
- related-work section
- replace the operating-point figure with the true frontier (needs v4)
- restore the distinct-prompt collapse numbers (needs v4)
- confirm whether evaluating with the *train*-split style classifier
  (`run_eval.py` calls `get_style_classifier('train', ...)`) is intentional
