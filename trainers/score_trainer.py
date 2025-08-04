import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from typing import Optional, Dict, Any, Union, List
import os
import wandb
import json
import click

from modules import BaseScoreModule
from utils import utils
from .trainer_utils import get_default_train_op, set_random_seed
from omegaconf import DictConfig

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
class ScoreTrainer:
    """Trainer that runs for a specified number of epochs. 

    Each epoch can run for a specified number of batches.
    Evaluation is done at the end of each epoch """

    def __init__(
        self,
        module: BaseScoreModule,
        train_dataset: Optional[Dataset],
        eval_dataset: Optional[Dataset],
        config: "DictConfig"
    ):
        assert config.do_eval == False or eval_dataset is not None, \
            "Need to have eval_dataset if do_eval is True"
        self.module = module

        self.train_dataset = train_dataset
        self.train_batch_size = config.train_batch_size
        self.train_shuffle = config.train_shuffle
        self.train_drop_last = config.train_drop_last
        self.num_train_epochs = config.num_train_epochs
        self.max_train_steps = config.max_train_steps

        self.eval_dataset = eval_dataset
        self.do_eval = config.do_eval
        self.eval_batch_size = config.eval_batch_size
        self.eval_steps = config.eval_steps

        self.do_save = config.do_save
        self.save_dir = config.save_dir
        self.save_steps = config.save_steps
        self.saved_steps: int = -1

        self.train_op = get_default_train_op(self.module._model,
                                             config.learning_rate,
                                             config.gradient_clip,
                                             config.gradient_clip_norm)

        if config.checkpoint_path is not None:
            self._load_checkpoint(config.checkpoint_path)

        if config.random_seed is not None:
            set_random_seed(config.random_seed)

        self.report_to_wandb = config.report_to_wandb
        self.project_name = config.project_name
        self.run_name = config.run_name

        self.random_lmbda = config.random_lmbda

    def _load_checkpoint(self, checkpoint_path: str) -> None:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        self.module.load_state_dict(checkpoint["model_state_dict"])
        self.saved_steps = checkpoint["steps"]
        print(click.style(f"Loaded module from {checkpoint_path}", fg="green"))

    def _get_train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_dataset,
                          shuffle=self.train_shuffle,
                          batch_size=self.train_batch_size,
                          drop_last=self.train_drop_last)

    # @torch.no_grad
    def _train_step(
        self,
        step: int,
        batch: Dict[str, Any]
    ) -> Dict[str, Any]:
        model = self.module.train()
        
        print('----------------------------------------------')
        print("Step:", step)

        n = len(batch['source_texts'])
        model._pre_steps(step)

        if self.random_lmbda:
            lmbda = torch.rand(n).to(device)
        else:
            lmbda = torch.tensor([0.5]).repeat(n).to(device)

        k = 0
        max_loss = 1000
        loss = torch.tensor([max_loss+1]).to(device)
        while loss.item() > 1000 and k < 10:
            loss, batch_log = model(lmbda, batch)
            loss.backward()
            k += 1

        self.train_op()

        return batch_log

    def train(self,
              report_to_wandb: Optional[bool] = None,
              project_name: Optional[str] = None,
              run_name: Optional[str] = None,
              config: Optional["DictConfig"] = None) -> None:
        # Configure Wandb reporting
        if report_to_wandb is None:
            report_to_wandb = self.report_to_wandb
        if project_name is None:
            project_name = self.project_name
        if run_name is None: 
            run_name = self.run_name
        if config is not None: 
            config = eval(str(config))
        if report_to_wandb:
            wandb.init(project=project_name, name=run_name, config=config)
            wandb.watch(self.module, log=None)

        # Create saving path
        eval_save_dir = os.path.join(self.save_dir, "eval")
        ckpt_save_dir = os.path.join(self.save_dir, "ckpt")
        if not os.path.exists(eval_save_dir):
            os.makedirs(eval_save_dir)
        if not os.path.exists(ckpt_save_dir):
            os.makedirs(ckpt_save_dir)

        train_dataloader = self._get_train_dataloader()
        # Determine whether to train by epoch or steps
        if self.max_train_steps < 0:
            total_train_epochs = self.num_train_epochs
        else:
            num_batches_per_epoch = len(train_dataloader)
            total_train_epochs = \
                (self.max_train_steps // num_batches_per_epoch
                 + int(self.max_train_steps % num_batches_per_epoch > 0))

        # Determine whether to evaluate by epoch or steps
        eval_by_steps = self.eval_steps > 0
        # Determine whether to save by epoch or steps
        save_by_steps = self.save_steps > 0

        print(f"Length of train dataloader: {len(train_dataloader)}")
        # print(eval_by_steps, save_by_steps)
        total_steps = max(0, self.saved_steps)
        for epoch in range(total_train_epochs):
            for step, batch in enumerate(train_dataloader, 1):
                if step > self.saved_steps:
                    batch_log = self._train_step(step, batch)
                    if report_to_wandb:
                        wandb.log(batch_log)
                    total_steps += 1

                    if self.do_eval and eval_by_steps \
                            and total_steps % self.eval_steps == 0:
                        output_save_path = \
                            os.path.join(eval_save_dir,
                                        f'outputs.step.{total_steps}.json')
                        with torch.no_grad():
                            eval_log = self.evaluate(output_save_path=output_save_path)
                        if report_to_wandb:
                            wandb.log(eval_log)

                    if self.do_save and save_by_steps \
                            and total_steps % self.save_steps == 0:
                        torch.save({"steps": total_steps,
                                    "model_state_dict": self.module.state_dict()},
                                os.path.join(ckpt_save_dir,
                                                f"ckpt.step.{total_steps}.pth"))

                    if total_steps == self.max_train_steps:
                        break

            if self.do_eval and not eval_by_steps:
                print('Saving...........')
                output_save_path = os.path.join(eval_save_dir,
                                                f'outputs.epoch.{epoch+1}.json')
                eval_log = self.evaluate(output_save_path=output_save_path)
                wandb.log(eval_log)

            if self.do_save and not save_by_steps:
                torch.save({"steps": total_steps,
                            "model_state_dict": self.module.state_dict()},
                           os.path.join(ckpt_save_dir,
                                        f"ckpt.epoch.{epoch+1}.pth"))

    def _get_eval_dataloader(self, eval_dataset: Dataset) -> DataLoader:
        return DataLoader(eval_dataset,
                          batch_size=self.eval_batch_size)

    def evaluate(
        self,
        lmbda: torch.Tensor = -1,
        eval_dataset: Optional[Dataset] = None,
        output_save_path: Optional[str] = None,
        compute_scores: bool = True
    ) -> Dict[str, np.number]:
        if eval_dataset is None:
            eval_dataset = self.eval_dataset
        eval_dataloader = self._get_eval_dataloader(eval_dataset)
        if lmbda == -1:
            lmbdas = torch.arange(0, 1, 0.1)
        else:
            lmbdas = [lmbda]
        model = self.module.eval()
        hypos = []
        mean_scores = []
        mean_contents= []
        mean_styles = []
        print('------------------------------Start Eval--------------------------------')
        for batch in eval_dataloader:
            for lmbda in lmbdas:
                infer_outputs: Dict[str, Union[torch.Tensor, List[List[str]]]]
                lmbda = lmbda.repeat_interleave(len(batch['source_texts'])).to(device)
                infer_outputs = model.infer(lmbda,batch)
                hypos += infer_outputs['sample_tokens']

                scores_tensor, content_tensor, style_tensor, score_log = model.compute_scores(
                    lmbda=lmbda,
                    batch=batch,
                    output_tokens=infer_outputs['sample_tokens'])
                mean_scores.append(score_log['mean_score'].tolist())
                mean_contents.append(score_log['mean_content'].tolist())
                mean_styles.append(score_log['mean_style'].tolist())
            break
        if output_save_path is not None:
            json.dump({'output_tokens': hypos,
                       'lmbdas': lmbdas.tolist(),
                       'mean_scores': mean_scores,
                       'mean_contents': mean_contents,
                       'mean_styles': mean_styles},
                      open(output_save_path, 'w'))
        
        mean_score = torch.Tensor(mean_scores).mean(dim=-1).mean().item()
        mean_content = torch.Tensor(mean_contents).mean(dim=-1).mean().item()
        mean_style = torch.Tensor(mean_styles).mean(dim=-1).mean().item()

        score_log = {
            "mean_score": mean_score,
            "mean_content": mean_content,
            "mean_style": mean_style
        }

        utils.add_prefix_to_dict_keys_inplace(
            score_log,
            prefix=f"eval/scores/")

        print('Finish Eval')
        return utils.unionize_dicts([
            score_log,
            # gem_scores_dict,
            {
                f"eval/output_length": np.mean([len(tokens) \
                                                for tokens in hypos])
            }
        ])