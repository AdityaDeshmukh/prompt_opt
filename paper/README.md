# Paper draft — R-REBEL vs GRPO, with theory

Scope: algorithms, theory, and results. Related work is still a stub (the
only open `\todo`), and there is no conclusion section yet.

## Build

```bash
cd paper && make          # pdflatex, bibtex, pdflatex x2 -> main.pdf
make results              # regenerate every table/figure/number from data, then build
make clean
```

Toolchain on this cluster: `pdflatex` and `bibtex` are present; `latexmk`,
`algorithm2e`, `algpseudocode`, `pgfplots`, `thmtools` and `cleveref` are
**not**. Hence `algorithm`+`algorithmic`, plain `amsthm`, and pre-rendered
matplotlib PDFs for every figure. Check `kpsewhich <pkg>.sty` before adding a
package. There is no `pdftoppm`; preview pages with
`gs -sDEVICE=png16m -r80 -dFirstPage=N -dLastPage=N -o p.png main.pdf`.

## Structure

| Section | File | Evidence |
|---|---|---|
| 2 Setup, 3 Algorithms | `sections/setup.tex`, `algorithms.tex` | code |
| 4 Theory of R-REBEL-std | `sections/theory.tex` (+ proofs in `appendix.tex`) | `analysis/theory_checks.py` |
| 5 Experimental setup | `sections/experiments.tex` | |
| 6.1–6.4 Main results (v4, test split) | `sections/results.tex` | `analysis/full_eval.py` |
| 6.5 Reward-scale measurement | `sections/results.tex` | `analysis/reward_scale_analysis.py` |
| 6.6–6.7 Seed replication (v3, dev set), provenance | `sections/results_v3.tex` | `analysis/paper_results.py` |
| 7 Limitations | `sections/limitations.tex` | |

## Where the numbers come from — nothing is typed by hand

Every number in the prose is a LaTeX macro written by a script, loaded in the
preamble of `main.tex`:

```
results/v4/v4_<arm>_seed0/test_full/output.step.<N>.seed<S>.json   <- slurm/eval_full.slurm
  -> analysis/full_eval.py  -> paper/tables/{full_results,pairwise,prompts,v4_numbers}.tex
                               paper/figures/{frontier,control,test_learning}.pdf
                               results/v4/full_eval_summary.json
results/v4/reward_scale_probe/*.json                               <- slurm/reward_probe.slurm
  -> analysis/reward_scale_analysis.py -> paper/tables/scale_numbers.tex, figures/reward_scale.pdf
analysis/theory_checks.py --fig ...    -> paper/figures/theory_toy.pdf  (31 checks, exits 1 on failure)
results/v3_wandb_export.json          -> analysis/paper_results.py --tex -> tables/main_results.tex,
                                                                           figures/learning_curves.pdf
```

`full_eval.py` refuses to write outputs unless every arm has all 5 task-LM
seeds at step 12000 (`--allow-partial` exists for drafting only). If you change
a sentence that quotes a number, use the macro; if the macro does not exist,
add it to `macros()` in the script rather than typing the value.

## Evaluation protocol (v4)

- **500-sentence Yelp test split**, never the 10-sentence dev set, for every
  v4 number. The dev set is used for exactly one thing: selecting a checkpoint
  in the early-stopping comparison.
- 20 floors, `lambda in {0, 0.05, ..., 0.95}` (training draws `lambda ~ U[0,1)`).
- Greedy prompt per (sentence, lambda); N=50 task-LM samples per prompt;
  fp32 scorers; `vllm_seed` pinned per task-LM seed.
- 5 task-LM seeds at step 12000 (prompts are bit-identical across seeds on the same GPU model, and flip for 0-29% of rows across GPU models;
  the reward's seed-to-seed SD is ~0.04), 1 seed at the learning-curve steps
  (1500, 3000, ..., 10500).
- 95% percentile bootstrap over sentences, 2000 resamples, **paired** across
  arms.

## Read this before trusting a number

1. **The v3 campaign's checkpoints and per-lambda evals were destroyed** by a
   scratch purge. Its scores survive only in wandb and are 10-sentence dev
   numbers on a 10-point grid; they appear only in the seed-replication
   subsection, labelled as such.
2. **No pre-v2 data goes in the paper.** Recovered 2025 curves in
   `results/recovered_prev2/` are plotted by `paper_results.py` only under an
   explicit `--recovered` flag, into a filename the paper does not include.
   Do not re-add them.
3. **v4 has one training seed per arm.** The test intervals do not include
   training-seed variation; the limitations section says so.
4. `figures/{components,tradeoff,baseline_debug}.pdf` and
   `tables/{results_12000,v4_test_results}.tex` are still produced by
   `paper_results.py` but are no longer included; the v4 figures supersede them.
