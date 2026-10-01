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
| 2 Reward I vs Reward II (per-sample vs expectation floor) | `sections/setup.tex` | `tst_score.py` (`reward_constraint`) |
| 4.5 Which constraint? (Lemma cluster, Props eps/penalty) | `sections/theory.tex` (+ proofs, App. B sim) | `analysis/theory_checks.py`, `analysis/constraint_sim.py` |
| 6.1–6.4 Main results (v5 = Reward II, test split) | `sections/results.tex` | `analysis/full_eval.py --campaign v5` |
| 6.5 Why the per-sample floor clusters (v4 = Reward I) | `sections/results.tex` | `analysis/frontier_probe.py` → `frontier_diagnosis.py`, `reward_design.py`, `diagnosis_macros.py` |
| 6.x Worked example + per-sample outcome audit | `sections/results.tex` | `analysis/example_outputs.py` → `example_table.py` |
| 6.6 Reward-scale measurement (Reward I) | `sections/results.tex` | `analysis/reward_scale_analysis.py` |
| 6.6–6.7 Seed replication (v3, dev set), provenance | `sections/results_v3.tex` | `analysis/paper_results.py` |
| 7 Limitations | `sections/limitations.tex` | |

## Where the numbers come from — nothing is typed by hand

Every number in the prose is a LaTeX macro written by a script, loaded in the
preamble of `main.tex`:

```
results/<c>/<c>_<arm>_seed0/test_full/output.step.<N>.seed<S>.json  <- slurm/eval_full.slurm (EVAL_CAMPAIGN=c)
  -> analysis/full_eval.py --campaign v4 -> tables/{full_results,pairwise,prompts,v4_numbers}.tex (\v...)
  -> analysis/full_eval.py --campaign v5 -> tables/{*_v5,v5_numbers}.tex (\w...), figures/*_v5.pdf,
                                            figures/compare_v4_v5.pdf
results/v4/frontier_probe/*.npz        <- slurm/frontier_probe.slurm (48-prompt per-sample re-scoring)
  -> analysis/frontier_diagnosis.py, reward_design.py, diagnosis_macros.py -> tables/diag_numbers.tex (\dg...),
                                            figures/menu_mechanism.pdf
  -> analysis/constraint_sim.py          -> tables/{constraint_sim,sim_numbers}.tex (\sim...)
results/examples/*.json                <- slurm/example_outputs.slurm, slurm/sample_outcomes.slurm
  -> analysis/example_table.py           -> tables/{example,outcomes,example_numbers}.tex (\ex...)
results/v4/reward_scale_probe/*.json                               <- slurm/reward_probe.slurm
  -> analysis/reward_scale_analysis.py -> paper/tables/scale_numbers.tex, figures/reward_scale.pdf
analysis/theory_checks.py --fig ...    -> paper/figures/theory_toy.pdf  (40 checks, exits 1 on failure)
results/v3_wandb_export.json          -> analysis/paper_results.py --tex -> tables/main_results.tex,
                                                                           figures/learning_curves.pdf
```

`full_eval.py` refuses to write outputs unless every arm has all 5 task-LM
seeds at step 12000 (`--allow-partial` exists for drafting only). If you change
a sentence that quotes a number, use the macro; if the macro does not exist,
add it to `macros()` in the script rather than typing the value.

## Evaluation protocol (v4 and v5)

- **500-sentence Yelp test split**, never the 10-sentence dev set, for every
  v4 number. The dev set is used for exactly one thing: selecting a checkpoint
  in the early-stopping comparison.
- 20 floors, `lambda in {0, 0.05, ..., 0.95}` (training draws `lambda ~ U[0,1)`).
- One prompt per (sentence, lambda); N=50 task-LM samples per prompt; fp32
  scorers; `vllm_seed` pinned per seed. **v5 samples the prompt** from the
  training sampler (`eval_decode=sample`, torch seed = eval seed). v4 used the
  legacy decoder, which is NOT greedy: it samples from the top-3 tokens
  (`eval_decode=top3`, near-greedy for those policies).
- 5 seeds at step 12000, 1 seed at the learning-curve steps (1500, ..., 10500).
  In v4 the top-3 sampler had a fixed seed, so prompts matched across seeds on
  one GPU model and differed across models only through the random stream.
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
3. **v4 and v5 have one training seed per arm.** The test intervals do not include
   training-seed variation; the limitations section says so.
4. `figures/{components,tradeoff,baseline_debug}.pdf` and
   `tables/{results_12000,v4_test_results}.tex` are still produced by
   `paper_results.py` but are no longer included; the v4 figures supersede them.
