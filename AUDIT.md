# Implementation audit — 2026-09-08

Scope: the whole `prompt_opt` tree, with emphasis on whether **GRPO** is
implemented properly and whether the R-REBEL-beats-GRPO comparison is sound.

Method: 10 independent auditors (one per dimension: GRPO loss core, GRPO
reference anchor, R-REBEL loss, arm fairness, reward pipeline, log-prob
alignment, training loop, eval validity, policy model, algo dispatch), 109 raw
findings, deduped to 26, each attacked by a 3-lens adversarial panel
(code-reality / impact / reference-correctness); 10 survived. **Every finding
below was then re-verified by hand** — the numbers and file:line references here
are ones I reproduced directly, not agent claims.

## Verdict

**GRPO's loss function is implemented correctly. Its *configuration* in this
campaign is not, and the plain-GRPO arm is a dead run.**

The surrogate, the group-relative advantage, the $k_3$ KL estimator, the
pairwise R-REBEL regression and the log-prob/token alignment all match their
references. Three configuration-and-protocol defects, however, bound what the
campaign can claim. The direction of the headline result survives all of them
(R-REBEL ~39.3 vs the best GRPO arm ~33.4, every R-REBEL seed above every GRPO
seed), but the framing "we debugged and tuned the baseline, and it still loses"
needed material correction, which has been applied to the draft.

## Confirmed — these affect the headline

### 1. All reported numbers are a 10-sentence **dev** score, not the test split
`run_tst_multi_obj.py:32` passes `val_dataset` as `ScoreTrainer`'s
`eval_dataset`, and `tst_helpers.py:12-14` caps the Yelp dev split to
**`max_size = 10`** (the comment says 16, the code says 10). So every
`outputs.step.*.json`, every wandb `eval/scores/*`, and therefore every number
in the paper, is a mean over **10 dev sentences** — while
`data/yelp/clean/sentiment.test.0.clean` holds **500** test sentences that no v3
run was ever evaluated on. Only `run_eval.py:20` uses `test_dataset`, and it was
never run for v3.

*Effect*: the 6-point family gap is far outside the 1-point noise floor and will
almost certainly survive, but the confidence interval is much wider than "3
seeds" implies and the sub-2-point variant gaps are not resolvable.
*Fix*: `slurm/eval_v4.slurm` runs the real 500-sentence test eval for the v4
arms. v3 cannot be upgraded — its checkpoints are gone.

### 2. GRPO's pinned KL reference is numerically **uniform**, so β·KL is an entropy bonus, not a trust region
Two independent facts combine:
- `logit_bias` is a **dead config knob**. Its only occurrence in the entire tree
  is the assignment at `models/lm_adaptor_model.py:73`; `_adapted_logits`
  (`:96-108`) never applies it. The paper's claimed "−10 logit bias" never
  happened.
- The adaptor is initialized with `xavier_uniform_(gain=1e-4)` and bias `-1e-4`
  (`models/lm_adaptor_model.py:42-45`), driving its output logits to ~0, so the
  initial policy is **exactly uniform** over the 50,257-token vocabulary.
  Verified against the logs: `loss/entropy` at step 0 is **10.824904** versus
  `ln(50257) = 10.824905`.

With `grpo_ref_sync_steps=0` the anchor is that uniform policy for the whole
run, so `β·KL(π‖Unif) = β(log V − H(π))` — up to a constant, `−β·H(π)`.

*Effect*: the arm labelled "GRPO" is REINFORCE + group-normalized advantage +
an entropy coefficient of 0.04, with **no trust region at all**; and the
"GRPO + entropy" arm only raises the total to ≈0.05. The two GRPO arms are one
configuration at two nearby settings of the *same* lever — **not** two
independent tunings, so their agreeing on a ~31–33 ceiling is not the
corroboration it looks like. This claim has been retracted in the draft.
*Fix (missing experiment)*: a GRPO baseline anchored on a trained/SFT reference.

### 3. Plain GRPO never learns — it is frozen from step 500
Arm mean **30.38 at step 500 → 30.19 at step 11,500: Δ = −0.18** over 11,000
steps, inside the noise floor and slightly negative (per-seed +0.11, +1.09,
−1.75). Over the same interval R-REBEL ℓ₁-std gains +6.74/+6.23/+7.77 and
grpo_ent gains +17.74/+3.23/+16.77. Corroborating diagnostics: logged
`loss/kl` averages **9.8237** against the $k_3$ estimator's analytic ceiling
**9.824925** for V=50257 — attained only when the sampled token has probability
≈1 — and distinct prompt tokens explored per cycle has median 5, i.e. one
frozen 5-token, λ-independent prompt.

*Effect*: "GRPO plateaus at ~31" describes a collapsed policy, not a converged
baseline. The defensible GRPO baseline is the **grpo_ent** arm (33.36), which
does learn. The draft now says so and treats plain GRPO as a collapse
diagnostic.

### 4. Adam state is never checkpointed
`ScoreTrainer._save_checkpoint` (`trainers/score_trainer.py:75-81`) stores only
`model_state_dict`, and `get_default_train_op` (`trainers/trainer_utils.py:9-11`)
constructs a fresh `optim.Adam` on every process start. Each run is therefore
~20 concatenated Adam warm restarts with zeroed moments, not one trajectory.
Cycle counts are close across arms (63–65), so it is not a systematic handicap
to either family, but it is a real methodological defect.
*Fix*: save/restore `optimizer.state_dict()`.

### 5. The authors' own health gate classifies both GRPO arms as pathological
`analysis/gate_grpo.py` exists to decide whether GRPO is healthy enough to
publish, and its escalation ladder's first rung (`grpo_beta` 0.04 → 0.1) was
never run. GRPO's one live hyperparameter was never searched.

## Verified correct

- **GRPO surrogate and advantage** — `(r − mean)/(std + 1e-8)` is computed
  within the group of G=16 sharing one source, on the correct axis
  (`loss_functions.py:146-151`).
- **The clip really is inert**, and for the right reason: `ratio = exp(lp −
  lp.detach())` is 1 in value, and since 1 is strictly interior to
  `[0.8, 1.2]` the clamp passes gradient unchanged, so both branches of the
  `min` agree in value *and* gradient. Note `∇ρ = ∇log π ≠ 0` — the ratio is
  what carries the gradient. (The draft originally got this backwards and has
  been corrected.)
- **KL estimator** is the standard non-negative $k_3$ form with the correct sign
  convention (`loss_functions.py:160-168`).
- **No length bias**: prompts are fixed at `prompt_length=5` with
  `eos_token_id=null`, so the token mean is uniform across samples — the
  Dr. GRPO length-normalization critique does not bite here.
- **Reference anchor survives resume**: `_ref_model` is a registered submodule
  and round-trips through `state_dict`, so the pinned anchor is not silently
  re-anchored to the current policy by the 4-hour resubmit chain
  (`score_loss_module.py:113-122`).
- **R-REBEL pairing**: `_pairwise_diff_triu` enumerates unordered within-group
  pairs, and the `view(num_src, -1)` reshape is safe because scores are laid out
  source-major (`tst_score.py:70-72` uses `repeat_interleave` / `_repeat_texts`),
  so pairs never cross sources.
- **Scale fairness**: for all three reported R-REBEL arms `reward_std_scale=True`
  (`score_loss_module.py:42-47`), so the group-std division cancels `score_scale`
  and `g(λ)` exactly; GRPO's advantage normalization cancels scale the same way.
  Neither family benefits from a tuned reward scale.
- **Entropy bonus is genuinely active for ℓ₁-ent** and off for ℓ₁-std
  (`score_loss_module.py:164-169` passes `ent_coef=0.01`; `make_rrebel_loss`
  gates it on `use_entropy`). An audit finding claiming these two arms are
  duplicates was **rejected** — the term is live and the observed 1.85-point gap
  exceeds the noise floor.
- **Reward** matches the constrained form the draft now states:
  `score = style if content ≥ 100λ else 0.01·(content − 100λ)`
  (`tst_score.py:103-106`), averaged over N=50 task-LM samples.

## Coverage gaps — what this audit did *not* settle

- **83 of 109 raw findings were never verified.** The workflow capped
  adversarial verification at the top 26 by severity; the rest were dropped
  unverified and are neither confirmed nor refuted.
- **10 verification agents and both synthesis agents failed** on an account
  spend limit, so 4 of the 26 findings (at `final_table.py:23`,
  `score_trainer.py:60`, `run_eval.py:24`, `loss_functions.py:210`) carry fewer
  than 3 votes, and this report was written by hand rather than synthesized.
- **Nothing could be re-measured empirically**: the v3 checkpoints and per-λ
  evals were purged from `/scratch`, so every dynamic claim here rests on the
  surviving wandb scalar history plus static reading of the code.
- The **train-split style classifier** used by `run_eval.py:24`
  (`get_style_classifier('train', ...)`) is flagged but unresolved — it scores
  every arm identically so it cannot explain the gap, but it makes absolute
  sentiment numbers non-comparable to test-split literature.
