"""Standalone per-phase timing harness for one training step.

Imports the live modules (no edits to the repo) and reproduces exactly what
ScoreLossModule._forward does for an R-REBEL step, timing each phase with
cuda.synchronize. Also (a) sweeps max_gen_batch_size for the task-LM phase and
(b) prototypes prefill-shared task-LM generation to measure the ceiling of that
optimization. Run via slurm/bench_phases.slurm on a secondary GPU.
"""
import os, sys, time, math, copy, statistics
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
import torch
from omegaconf import OmegaConf

sys.path.insert(0, '/u/ad11/prompt_opt')
os.chdir('/u/ad11/prompt_opt')

from tst_helpers import make_text_style_transfer_datasets, get_style_classifier
from models import build_policy_model, SinglePromptModel
from tst_score import PromptedTextStyleTransferScore
from tst_modules.objectives import ListDataset

torch.manual_seed(0)
DEV = torch.device('cuda:0')


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def med(xs):
    xs = list(xs)
    return statistics.median(xs) if xs else float('nan')


def build():
    cfg = OmegaConf.load('tst_config.yaml')
    cfg.algo = 'rrebel_l1_std'
    cfg.task_lm_backend = 'hf'  # this whole script diagnoses the HF path
    cfg.dataset = 'yelp'
    cfg.dataset_seed = None
    cfg.style_classifier = get_style_classifier('train', cfg)
    cfg.device = 'cuda:0'
    cfg.device_scorer = 'cuda:0'
    cfg.report_to_wandb = False
    cfg.random_seed = 0
    return cfg


def main():
    print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU", flush=True)
    cfg = build()
    NB = cfg.train_batch_size          # 15 sources
    NR = cfg.num_repeats               # 16 -> 240 rows
    N = cfg.num_samples * cfg.num_bootstraps  # 50 task-LM samples / prompt
    print(f"config: sources={NB} num_repeats={NR} -> rows={NB*NR}; "
          f"samples/prompt={N} -> outputs/step={NB*NR*N}", flush=True)

    train, dev, test = make_text_style_transfer_datasets(cfg)
    srcs = list(train.source_texts[:NB])
    labels = list(train.target_labels[:NB])

    print("loading policy + ref + scorers ...", flush=True)
    policy = SinglePromptModel(build_policy_model(cfg), cfg)
    ref_policy = copy.deepcopy(policy)
    for p in ref_policy.parameters():
        p.requires_grad = False
    score = PromptedTextStyleTransferScore(cfg)
    gen = score.generator
    print("loaded.\n", flush=True)

    # ---- full phase split at the campaign's max_gen_batch_size (128) -------
    def one_step(mgbs, reps=3, do_backward=True):
        gen.task_lm.max_gen_batch_size = mgbs
        T = {k: [] for k in ('rollout', 'ref_tf', 'task_gen',
                             'content', 'style', 'backward')}
        for r in range(reps):
            lmbda = torch.rand(NB).to(DEV)
            sync(); t = time.time()
            outputs = policy.generate(source_texts=srcs, lmbda=lmbda,
                                      do_sample=True, top_k=cfg.top_k,
                                      top_p=cfg.top_p, num_beams=1, num_repeats=NR)
            sync(); T['rollout'].append(time.time() - t)

            sync(); t = time.time()
            with torch.no_grad():
                ref_policy.teacher_forcing(lmbda=lmbda, source_texts=srcs,
                                           sample_ids=outputs['sample_ids'],
                                           num_repeats=NR)
            sync(); T['ref_tf'].append(time.time() - t)

            prompt_strs = score._convert_tokens_to_string(outputs['sample_tokens'])
            source_strs = score._repeat_texts(srcs)          # 240

            sync(); t = time.time()
            hypos = gen.sample_generate_grouped(prompt_strs, source_strs, N,
                                                score.top_k, score.top_p)
            sync(); T['task_gen'].append(time.time() - t)

            flat_srcs = [s for s in source_strs for _ in range(N)]
            flat_hypos = [h for row in hypos for h in row]

            sync(); t = time.time()
            score.objectives.bert_scorer.score(
                flat_hypos, flat_srcs,
                batch_size=max(cfg.style_batch_size, 64))
            sync(); T['content'].append(time.time() - t)

            sync(); t = time.time()
            for _ in score.objectives.style_classifier(
                    ListDataset(flat_hypos),
                    batch_size=cfg.style_batch_size, truncation=True):
                pass
            sync(); T['style'].append(time.time() - t)

            if do_backward:
                sync(); t = time.time()
                loss = outputs['sample_logits'].float().pow(2).mean()
                loss.backward()
                policy.zero_grad(set_to_none=True)
                sync(); T['backward'].append(time.time() - t)
            del outputs
            torch.cuda.empty_cache()
        return {k: med(v[1:] or v) for k, v in T.items()}  # drop warmup rep

    print("=== PHASE SPLIT @ max_gen_batch_size=128 (campaign setting) ===", flush=True)
    split = one_step(128, reps=3)
    total = sum(split.values())
    order = ['rollout', 'ref_tf', 'task_gen', 'content', 'style', 'backward']
    for k in order:
        print(f"  {k:10s} {split[k]:7.2f}s  {100*split[k]/total:5.1f}%", flush=True)
    print(f"  {'TOTAL':10s} {total:7.2f}s  100.0%", flush=True)
    print(f"  (task_gen + content + style = "
          f"{100*(split['task_gen']+split['content']+split['style'])/total:.1f}% "
          f"of the step)\n", flush=True)

    # ---- task-LM batch-size sensitivity -----------------------------------
    print("=== TASK-LM GEN vs max_gen_batch_size ===", flush=True)
    base = None
    for mgbs in (128, 256, 400, 512):
        try:
            gen.task_lm.max_gen_batch_size = mgbs
            lmbda = torch.rand(NB).to(DEV)
            with torch.no_grad():
                outputs = policy.generate(source_texts=srcs, lmbda=lmbda,
                                          do_sample=True, top_k=cfg.top_k,
                                          top_p=cfg.top_p, num_beams=1,
                                          num_repeats=NR)
            prompt_strs = score._convert_tokens_to_string(outputs['sample_tokens'])
            source_strs = score._repeat_texts(srcs)
            rpc = max(1, mgbs // N)
            n_calls = math.ceil((NB * NR) / rpc)
            # warm + timed
            gen.sample_generate_grouped(prompt_strs, source_strs, N, score.top_k, score.top_p)
            sync(); t = time.time()
            gen.sample_generate_grouped(prompt_strs, source_strs, N, score.top_k, score.top_p)
            sync(); dt = time.time() - t
            base = base or dt
            print(f"  mgbs={mgbs:4d}  rows/chunk={rpc:2d}  calls/step={n_calls:3d}  "
                  f"gen={dt:6.2f}s  ({base/dt:.2f}x vs mgbs=128)", flush=True)
            del outputs; torch.cuda.empty_cache()
        except RuntimeError as e:
            print(f"  mgbs={mgbs:4d}  OOM/err: {str(e)[:60]}", flush=True)
            torch.cuda.empty_cache()

    # ---- prefill-shared generation prototype ------------------------------
    print("\n=== PREFILL-SHARED TASK-LM GEN (prototype) ===", flush=True)
    prefill_shared_bench(gen, score, policy, cfg, srcs, NB, NR, N)


@torch.no_grad()
def prefill_shared_bench(gen, score, policy, cfg, srcs, NB, NR, N):
    """Prefill the prompt once per prompt, then repeat_interleave the KV cache
    N times and decode. Compares wall time against the HF num_return_sequences
    path and prints a sample to confirm it produces sane text."""
    tok = gen.task_lm.tokenizer
    model = gen.task_lm.model
    mgbs = 400
    lmbda = torch.rand(NB).to(DEV)
    outputs = policy.generate(source_texts=srcs, lmbda=lmbda, do_sample=True,
                              top_k=cfg.top_k, top_p=cfg.top_p, num_beams=1,
                              num_repeats=NR)
    prompts = score._convert_tokens_to_string(outputs['sample_tokens'])
    source_texts = score._repeat_texts(srcs)
    n_rows = len(prompts)
    templates = [gen.template.format(prompt=p, sentence_1=s)
                 for p, s in zip(prompts, source_texts)]
    src_lens = [len(tok(s)['input_ids']) for s in source_texts]
    max_new = [int(math.ceil(gen._get_max_new_tokens(l))) for l in src_lens]
    order = sorted(range(n_rows), key=lambda i: max_new[i])
    rows_per_chunk = max(1, mgbs // N)
    top_k = score.top_k

    def run_shared():
        results = [None] * n_rows
        for start in range(0, n_rows, rows_per_chunk):
            idxs = order[start:start + rows_per_chunk]
            chunk_templates = [templates[i] for i in idxs]
            chunk_max_new = max(max_new[i] for i in idxs)
            enc = tok(chunk_templates, return_tensors='pt', padding=True).to(gen.device)
            input_ids, attn = enc['input_ids'], enc['attention_mask']
            out = model(input_ids=input_ids, attention_mask=attn, use_cache=True)
            past = out.past_key_values
            # expand KV + logits + mask by N (legacy tuple cache)
            past = tuple(tuple(t.repeat_interleave(N, dim=0) for t in layer)
                         for layer in past)
            attn = attn.repeat_interleave(N, dim=0)
            logits = out.logits[:, -1, :].repeat_interleave(N, dim=0)
            B = attn.size(0)
            gen_ids = torch.empty(B, chunk_max_new, dtype=torch.long, device=gen.device)
            for step in range(chunk_max_new):
                # top-k sample
                v, ix = torch.topk(logits, top_k, dim=-1)
                probs = torch.softmax(v, dim=-1)
                nxt = ix.gather(-1, torch.multinomial(probs, 1))  # [B,1]
                gen_ids[:, step] = nxt.squeeze(-1)
                attn = torch.cat([attn, torch.ones_like(attn[:, :1])], dim=-1)
                out = model(input_ids=nxt, attention_mask=attn,
                            past_key_values=past, use_cache=True)
                past = out.past_key_values
                logits = out.logits[:, -1, :]
            for row_pos, i in enumerate(idxs):
                rows = gen_ids[row_pos*N:(row_pos+1)*N, :max_new[i]]
                results[i] = tok.batch_decode(rows, skip_special_tokens=True)
        return results

    gen.task_lm.max_gen_batch_size = mgbs
    # baseline (HF path)
    gen.sample_generate_grouped(prompts, source_texts, N, top_k, score.top_p)
    sync(); t = time.time()
    gen.sample_generate_grouped(prompts, source_texts, N, top_k, score.top_p)
    sync(); base = time.time() - t

    # shared
    run_shared()
    sync(); t = time.time()
    shared = run_shared()
    sync(); sh = time.time() - t

    print(f"  HF num_return_sequences path : {base:6.2f}s  (mgbs={mgbs})", flush=True)
    print(f"  prefill-shared path          : {sh:6.2f}s  ({base/sh:.2f}x)", flush=True)
    ex = next((r for r in shared if r), [""])[0]
    print(f"  sample shared output: {ex[:90]!r}", flush=True)


if __name__ == '__main__':
    main()
