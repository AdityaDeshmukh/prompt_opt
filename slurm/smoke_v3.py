"""GPU smoke test for the prompt_opt_v3 env (vllm + torch 2.11 + transformers 5.x).

Validates, in dependency order:
  1. policy rollout (distilgpt2 + MLP adaptor) under the new transformers
  2. v2 checkpoint loads under the new torch (weights_only default)
  3. scorers (BERTScore roberta-large + sentiment pipeline)
  4. vLLM engine on gpt2-xl + grouped generation through PromptedGenerator
  5. six end-to-end training steps (rrebel_l1_ent) with step timing
Exits nonzero on the first failure.
"""
import os, sys, time, glob
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
import torch
from omegaconf import OmegaConf

sys.path.insert(0, '/u/ad11/prompt_opt')
os.chdir('/u/ad11/prompt_opt')

DEV = torch.device('cuda:0')
print("GPU:", torch.cuda.get_device_name(0), flush=True)


def step(name):
    print(f"\n===== {name} =====", flush=True)


def main():
    from tst_helpers import make_text_style_transfer_datasets, get_style_classifier
    cfg = OmegaConf.load('tst_config.yaml')
    cfg.algo = 'rrebel_l1_ent'
    cfg.dataset = 'yelp'; cfg.dataset_seed = None
    cfg.style_classifier = get_style_classifier('train', cfg)
    cfg.device = 'cuda:0'; cfg.device_scorer = 'cuda:0'
    cfg.report_to_wandb = False; cfg.random_seed = 0
    cfg.task_lm_backend = 'vllm'

    step("1. policy rollout (transformers 5.x KV-cache path)")
    from models import build_policy_model, SinglePromptModel
    policy = SinglePromptModel(build_policy_model(cfg), cfg)
    train, dev, test = make_text_style_transfer_datasets(cfg)
    srcs = list(train.source_texts[:3])
    lmbda = torch.rand(3).to(DEV)
    out = policy.generate(source_texts=srcs, lmbda=lmbda, do_sample=True,
                          top_k=cfg.top_k, top_p=cfg.top_p, num_beams=1,
                          num_repeats=4)
    assert out['sample_ids'].shape == (12, cfg.prompt_length), out['sample_ids'].shape
    assert out['sample_logits'].requires_grad
    # teacher forcing too (ref-model path)
    tf = policy.teacher_forcing(lmbda=lmbda, source_texts=srcs,
                                sample_ids=out['sample_ids'], num_repeats=4)
    assert tf['sample_logits'].shape == out['sample_logits'].shape
    print("rollout OK:", out['sample_ids'].shape, "logits grad:",
          out['sample_logits'].requires_grad, flush=True)

    step("2. v2 checkpoint load (torch 2.11 weights_only default)")
    from modules import ScoreLossModule
    ckpts = sorted(glob.glob(
        'outputs/v2/v2_rrebel_l1_ent_seed0/ckpt/ckpt.step.1200.pth'))
    assert ckpts, "no v2 checkpoint found for load test"
    ck = torch.load(ckpts[0], map_location='cpu')
    print("ckpt keys:", list(ck.keys()), "steps:", ck['steps'], flush=True)
    # a throwaway module to check strict state-dict compatibility
    probe = ScoreLossModule(policy, None, cfg)
    probe.load_state_dict(ck['model_state_dict'])
    print("state_dict strict-load OK", flush=True)
    del probe, ck

    step("3. scorers under transformers 5.x")
    from tst_modules.objectives import TextStyleTransferObjectives
    obj = TextStyleTransferObjectives(cfg.style_classifier,
                                      cfg.style_tokenizer, 64, DEV,
                                      scorer_dtype='float16')
    c, s = obj.compute_scores_flat(
        ["the food was terrible."] * 3,
        ["the food was wonderful.", "the food was awful.", "great place!"],
        ["LABEL_1"] * 3)
    print("content:", [round(x, 1) for x in c.tolist()],
          "style:", [round(x, 1) for x in s.tolist()], flush=True)
    assert c.shape == (3,) and s.shape == (3,)
    del obj
    torch.cuda.empty_cache()

    step("4. vLLM engine + grouped generation")
    from tst_score import PromptedTextStyleTransferScore
    score = PromptedTextStyleTransferScore(cfg)
    hy = score.generator.sample_generate_grouped(
        ["Rewrite positively:"] * 2,
        ["the service was slow and rude.", "i hated the food."],
        5, cfg.task_top_k, 1.0)
    assert len(hy) == 2 and len(hy[0]) == 5
    print("vllm sample:", repr(hy[0][0])[:100], flush=True)

    step("5. six end-to-end training steps (timed)")
    from trainers import ScoreTrainer
    algo_module = ScoreLossModule(policy, score, cfg)
    cfg.save_dir = '/scratch/ad11/prompt_opt/outputs/smoke_v3'
    cfg.max_train_steps = 6
    cfg.do_eval = False; cfg.do_save = False
    trainer = ScoreTrainer(algo_module, train, dev, cfg)
    t0 = time.time()
    trainer.train(report_to_wandb=False)
    dt = time.time() - t0
    print(f"\n6 steps in {dt:.1f}s -> {dt/6:.1f}s/step "
          f"(first step includes warmup)", flush=True)
    print("\nALL SMOKE TESTS PASSED", flush=True)


if __name__ == '__main__':
    main()
