"""Trimmed generation-only benchmark (task_gen is ~98% of a step).

Measures, on ONE fast card: (a) task-LM gen time vs max_gen_batch_size,
(b) bf16 vs fp16 at a fixed batch size, (c) prefill-shared decode prototype.
Builds only the policy + task-LM generator (skips the scorers — irrelevant here).
"""
import os, sys, time, math
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
import torch
from omegaconf import OmegaConf

sys.path.insert(0, '/u/ad11/prompt_opt')
os.chdir('/u/ad11/prompt_opt')
from tst_helpers import make_text_style_transfer_datasets, get_style_classifier
from models import build_policy_model, SinglePromptModel
from tst_modules import PromptedGenerator

torch.manual_seed(0)
DEV = torch.device('cuda:0')


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def main():
    print("GPU:", torch.cuda.get_device_name(0), flush=True)
    print("native bf16 supported:", torch.cuda.is_bf16_supported(), flush=True)

    cfg = OmegaConf.load('tst_config.yaml')
    cfg.algo = 'rrebel_l1_std'; cfg.dataset = 'yelp'; cfg.dataset_seed = None
    cfg.style_classifier = get_style_classifier('train', cfg)
    cfg.device = 'cuda:0'; cfg.device_scorer = 'cuda:0'
    cfg.report_to_wandb = False; cfg.random_seed = 0

    NB = cfg.train_batch_size; NR = cfg.num_repeats
    N = cfg.num_samples * cfg.num_bootstraps
    train, dev, test = make_text_style_transfer_datasets(cfg)
    srcs = list(train.source_texts[:NB])
    print(f"rows={NB*NR}  samples/prompt={N}  outputs/step={NB*NR*N}", flush=True)

    policy = SinglePromptModel(build_policy_model(cfg), cfg)
    ptok = policy._model.tokenizer

    def make_gen(dtype):
        return PromptedGenerator(cfg.task_lm, cfg.template, cfg.end_punct,
                                 cfg.pad_token, DEV, cfg.lower_outputs,
                                 cfg.control_output_length, dtype=dtype,
                                 max_gen_batch_size=128, backend='hf')

    # fixed set of 240 prompts for every measurement
    with torch.no_grad():
        lmbda = torch.rand(NB).to(DEV)
        outputs = policy.generate(source_texts=srcs, lmbda=lmbda, do_sample=True,
                                  top_k=cfg.top_k, top_p=cfg.top_p, num_beams=1,
                                  num_repeats=NR)
    prompts = [ptok.convert_tokens_to_string(s) for s in outputs['sample_tokens']]
    source_texts = [s for s in srcs for _ in range(NR)]

    def gen_time(gen, mgbs):
        gen.task_lm.max_gen_batch_size = mgbs
        gen.sample_generate_grouped(prompts, source_texts, N, cfg.task_top_k, 1.0)  # warm
        sync(); t = time.time()
        gen.sample_generate_grouped(prompts, source_texts, N, cfg.task_top_k, 1.0)
        sync(); return time.time() - t

    print("\n=== task-LM gen vs max_gen_batch_size (bf16) ===", flush=True)
    gen = make_gen(cfg.get('task_lm_dtype', 'bfloat16'))
    base = None
    for mgbs in (128, 256, 512, 768, 1024):
        try:
            dt = gen_time(gen, mgbs)
            base = base or dt
            rpc = max(1, mgbs // N); ncalls = math.ceil((NB * NR) / rpc)
            print(f"  mgbs={mgbs:4d}  rows/chunk={rpc:2d}  calls={ncalls:3d}  "
                  f"gen={dt:6.2f}s  {base/dt:.2f}x", flush=True)
        except RuntimeError as e:
            print(f"  mgbs={mgbs:4d}  OOM/err: {str(e)[:50]}", flush=True)
            torch.cuda.empty_cache()

    print("\n=== bf16 vs fp16 @ mgbs=512 ===", flush=True)
    try:
        bf = gen_time(gen, 512)
        del gen; torch.cuda.empty_cache()
        genf = make_gen('float16')
        fp = gen_time(genf, 512)
        print(f"  bf16={bf:.2f}s  fp16={fp:.2f}s  ({bf/fp:.2f}x)", flush=True)
        del genf; torch.cuda.empty_cache()
    except RuntimeError as e:
        print("  err:", str(e)[:60], flush=True)
        torch.cuda.empty_cache()

    print("\n=== prefill-shared decode prototype @ mgbs=512 (bf16) ===", flush=True)
    prefill_shared(cfg, policy, srcs, NB, NR, N, prompts, source_texts)


@torch.no_grad()
def prefill_shared(cfg, policy, srcs, NB, NR, N, prompts, source_texts):
    gen = PromptedGenerator(cfg.task_lm, cfg.template, cfg.end_punct, cfg.pad_token,
                            DEV, cfg.lower_outputs, cfg.control_output_length,
                            dtype=cfg.get('task_lm_dtype', 'bfloat16'),
                            max_gen_batch_size=512, backend='hf')
    tok = gen.task_lm.tokenizer; model = gen.task_lm.model; mgbs = 512; top_k = cfg.task_top_k
    n_rows = len(prompts)
    templates = [gen.template.format(prompt=p, sentence_1=s)
                 for p, s in zip(prompts, source_texts)]
    src_lens = [len(tok(s)['input_ids']) for s in source_texts]
    max_new = [int(math.ceil(gen._get_max_new_tokens(l))) for l in src_lens]
    order = sorted(range(n_rows), key=lambda i: max_new[i])
    rows_per_chunk = max(1, mgbs // N)

    def run_shared():
        results = [None] * n_rows
        for start in range(0, n_rows, rows_per_chunk):
            idxs = order[start:start + rows_per_chunk]
            ct = [templates[i] for i in idxs]
            cmax = max(max_new[i] for i in idxs)
            enc = tok(ct, return_tensors='pt', padding=True).to(gen.device)
            attn = enc['attention_mask']
            out = model(input_ids=enc['input_ids'], attention_mask=attn, use_cache=True)
            past = tuple(tuple(t.repeat_interleave(N, dim=0) for t in layer)
                         for layer in out.past_key_values)
            attn = attn.repeat_interleave(N, dim=0)
            logits = out.logits[:, -1, :].repeat_interleave(N, dim=0)
            B = attn.size(0)
            gen_ids = torch.empty(B, cmax, dtype=torch.long, device=gen.device)
            for step in range(cmax):
                v, ix = torch.topk(logits, top_k, dim=-1)
                nxt = ix.gather(-1, torch.multinomial(torch.softmax(v, -1), 1))
                gen_ids[:, step] = nxt.squeeze(-1)
                attn = torch.cat([attn, torch.ones_like(attn[:, :1])], dim=-1)
                out = model(input_ids=nxt, attention_mask=attn,
                            past_key_values=past, use_cache=True)
                past = out.past_key_values
                logits = out.logits[:, -1, :]
            for rp, i in enumerate(idxs):
                results[i] = tok.batch_decode(gen_ids[rp*N:(rp+1)*N, :max_new[i]],
                                              skip_special_tokens=True)
        return results

    gen.sample_generate_grouped(prompts, source_texts, N, top_k, 1.0)  # warm HF
    sync(); t = time.time()
    gen.sample_generate_grouped(prompts, source_texts, N, top_k, 1.0)
    sync(); base = time.time() - t
    run_shared()  # warm
    sync(); t = time.time()
    sh = run_shared()
    sync(); sht = time.time() - t
    print(f"  HF num_return_sequences : {base:6.2f}s", flush=True)
    print(f"  prefill-shared          : {sht:6.2f}s  ({base/sht:.2f}x)", flush=True)
    ex = next((r for r in sh if r), [""])[0]
    print(f"  sample output: {ex[:80]!r}", flush=True)


if __name__ == '__main__':
    main()
