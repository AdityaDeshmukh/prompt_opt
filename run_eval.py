import os
import re
import hydra
from omegaconf import DictConfig, OmegaConf

from trainers import ScoreTrainer
from modules import ScoreLossModule
from models import build_policy_model, SinglePromptModel
from utils.utils import (colorful_print, get_hydra_output_dir)
from tst_helpers import (make_text_style_transfer_datasets, get_style_classifier)

from tst_score import PromptedTextStyleTransferScore

@hydra.main(version_base=None, config_path="./", config_name="tst_config")
def main(config: "DictConfig"):
    colorful_print(OmegaConf.to_yaml(config), fg='red')
    output_dir = get_hydra_output_dir()

    train_dataset, val_dataset, test_dataset = \
        make_text_style_transfer_datasets(config)

    policy_model = build_policy_model(config)
    prompt_model = SinglePromptModel(policy_model, config)
    config.style_classifier = get_style_classifier('train', config)
    score_module = PromptedTextStyleTransferScore(config)
    algo_module = ScoreLossModule(prompt_model, score_module, config)

    config.save_dir = os.path.join(output_dir, config.save_dir)
    trainer = ScoreTrainer(algo_module, train_dataset, test_dataset, config)
    eval_save_dir = os.path.join(config.save_dir, "test")
    if not os.path.exists(eval_save_dir):
        os.makedirs(eval_save_dir)
    # name the result by checkpoint step so evals at different steps coexist
    out_name = "output.json"
    if config.checkpoint_path:
        m = re.search(r"step\.(\d+)\.pth$", str(config.checkpoint_path))
        if m:
            out_name = f"output.step.{m.group(1)}.json"
    trainer.evaluate(output_save_path=os.path.join(eval_save_dir, out_name))


if __name__ == "__main__":
    main()