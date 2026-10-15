if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

import os
import json
import hydra
import torch
from omegaconf import OmegaConf
import pathlib
from torch.utils.data import DataLoader, WeightedRandomSampler
import copy
import numpy as np
import random
import wandb
import tqdm
import shutil

from diffusion_policy.common.pytorch_util import dict_apply, optimizer_to
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.policy.diffusion_unet_lowdim_policy import DiffusionUnetLowdimPolicy
from diffusion_policy.dataset.base_dataset import BaseLowdimDataset
from diffusion_policy.env_runner.base_lowdim_runner import BaseLowdimRunner
from diffusion_policy.common.checkpoint_util import TopKCheckpointManager
from diffusion_policy.common.json_logger import JsonLogger
from diffusion_policy.model.common.lr_scheduler import get_scheduler
from diffusers.training_utils import EMAModel

OmegaConf.register_new_resolver("eval", eval, replace=True)


def _link_or_copy(source: pathlib.Path, destination: pathlib.Path) -> None:
    """Create a cheap same-filesystem snapshot, with a copy fallback."""
    try:
        destination.unlink()
    except FileNotFoundError:
        pass
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def _atomic_workspace_checkpoint(workspace, destination: pathlib.Path) -> None:
    """Synchronously persist a named checkpoint without exposing a partial file."""
    tmp_path = destination.with_name(destination.name + f'.{os.getpid()}.tmp')
    workspace.save_checkpoint(path=tmp_path, use_thread=False)
    os.replace(tmp_path, destination)


# %%
class TrainDiffusionUnetLowdimWorkspace(BaseWorkspace):
    include_keys = ['global_step', 'epoch', 'optimizer_steps']

    def __init__(self, cfg: OmegaConf, output_dir=None):
        super().__init__(cfg, output_dir=output_dir)

        # set seed
        seed = cfg.training.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # configure model
        self.model: DiffusionUnetLowdimPolicy
        self.model = hydra.utils.instantiate(cfg.policy)

        self.ema_model: DiffusionUnetLowdimPolicy = None
        if cfg.training.use_ema:
            self.ema_model = copy.deepcopy(self.model)

        # configure training state
        self.optimizer = hydra.utils.instantiate(
            cfg.optimizer, params=self.model.parameters())

        self.global_step = 0
        self.epoch = 0
        self.optimizer_steps = 0

    def run(self):
        cfg = copy.deepcopy(self.cfg)

        # resume training
        if cfg.training.resume:
            lastest_ckpt_path = self.get_checkpoint_path()
            if lastest_ckpt_path.is_file():
                print(f"Resuming from checkpoint {lastest_ckpt_path}")
                self.load_checkpoint(path=lastest_ckpt_path)
                # Checkpoints created before D25 stored global_step but not the
                # explicit optimizer-step counter.  In this workspace they are
                # identical for gradient_accumulate_every=1, so recover the
                # counter rather than replaying a second full T budget.
                # Older checkpoints predate the explicit counter.  New D25
                # archive checkpoints carry it and it is authoritative: an
                # archive written immediately after an optimizer update can
                # have a one-step-behind logging ``global_step``.
                if self.optimizer_steps <= 0 and self.global_step > 0:
                    self.optimizer_steps = int(self.global_step)
                self.global_step = int(self.optimizer_steps)

        # configure dataset
        dataset: BaseLowdimDataset
        dataset = hydra.utils.instantiate(cfg.task.dataset)
        assert isinstance(dataset, BaseLowdimDataset)
        loader_cfg = OmegaConf.to_container(cfg.dataloader, resolve=True)
        if getattr(dataset, 'sample_weights', None) is not None:
            loader_cfg['shuffle'] = False
            loader_cfg['sampler'] = WeightedRandomSampler(
                torch.as_tensor(dataset.sample_weights, dtype=torch.double),
                num_samples=len(dataset), replacement=True)
        train_dataloader = DataLoader(dataset, **loader_cfg)
        normalizer = dataset.get_normalizer()

        # configure validation dataset
        val_dataset = dataset.get_validation_dataset()
        val_dataloader = DataLoader(val_dataset, **cfg.val_dataloader)

        self.model.set_normalizer(normalizer)
        if cfg.training.use_ema:
            self.ema_model.set_normalizer(normalizer)

        # configure lr scheduler
        lr_scheduler = get_scheduler(
            cfg.training.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=cfg.training.lr_warmup_steps,
            num_training_steps=int(cfg.training.get(
                'total_optimizer_steps',
                (len(train_dataloader) * cfg.training.num_epochs) \
                    // cfg.training.gradient_accumulate_every)),
            # pytorch assumes stepping LRScheduler every epoch
            # however huggingface diffusers steps it every batch
            last_epoch=self.global_step-1
        )

        # configure ema
        ema: EMAModel = None
        if cfg.training.use_ema:
            ema = hydra.utils.instantiate(
                cfg.ema,
                model=self.ema_model)

        # configure env runner
        env_runner: BaseLowdimRunner
        env_runner = hydra.utils.instantiate(
            cfg.task.env_runner,
            output_dir=self.output_dir)
        assert isinstance(env_runner, BaseLowdimRunner)

        # configure logging
        wandb_run = wandb.init(
            dir=str(self.output_dir),
            config=OmegaConf.to_container(cfg, resolve=True),
            **cfg.logging
        )
        wandb.config.update(
            {
                "output_dir": self.output_dir,
            }
        )

        # configure checkpoint
        topk_manager = TopKCheckpointManager(
            save_dir=os.path.join(self.output_dir, 'checkpoints'),
            **cfg.checkpoint.topk
        )

        # device transfer
        device = torch.device(cfg.training.device)
        self.model.to(device)
        if self.ema_model is not None:
            self.ema_model.to(device)
        optimizer_to(self.optimizer, device)

        # save batch for sampling
        train_sampling_batch = None

        if cfg.training.debug:
            cfg.training.num_epochs = 2
            cfg.training.max_train_steps = 3
            cfg.training.max_val_steps = 3
            cfg.training.rollout_every = 1
            cfg.training.checkpoint_every = 1
            cfg.training.val_every = 1
            cfg.training.sample_every = 1

        # training loop
        log_path = os.path.join(self.output_dir, 'logs.json.txt')
        batch_log_every = max(1, int(cfg.training.get("batch_log_every", 1)))
        with JsonLogger(log_path) as json_logger:
            stop_after_epoch = cfg.training.get("stop_after_epoch", None)
            run_end_epoch = (
                cfg.training.num_epochs
                if stop_after_epoch is None
                else min(cfg.training.num_epochs, int(stop_after_epoch))
            )
            target_steps = cfg.training.get('total_optimizer_steps', None)
            # D25 evaluates a fixed learning curve.  These named checkpoints
            # must be written at exact optimizer steps, never inferred later
            # from an epoch boundary or reconstructed from ``latest``.
            archive_steps = {
                int(x) for x in cfg.training.get('archive_checkpoint_steps', [])
            }
            archive_dir = pathlib.Path(self.output_dir).joinpath('checkpoints')
            written_archive_steps = {
                step for step in archive_steps
                if archive_dir.joinpath(f'step={step:06d}.ckpt').is_file()
            }
            while self.epoch < run_end_epoch and (target_steps is None or self.optimizer_steps < int(target_steps)):
                step_log = dict()
                # ========= train for this epoch ==========
                train_losses = list()
                with tqdm.tqdm(train_dataloader, desc=f"Training epoch {self.epoch}", 
                        leave=False, mininterval=cfg.training.tqdm_interval_sec) as tepoch:
                    for batch_idx, batch in enumerate(tepoch):
                        # device transfer
                        batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                        if train_sampling_batch is None:
                            train_sampling_batch = batch

                        # compute loss
                        raw_loss = self.model.compute_loss(batch)
                        loss = raw_loss / cfg.training.gradient_accumulate_every
                        loss.backward()

                        # step optimizer
                        if self.global_step % cfg.training.gradient_accumulate_every == 0:
                            self.optimizer.step()
                            self.optimizer.zero_grad()
                            lr_scheduler.step()
                            self.optimizer_steps += 1
                        
                        # update ema
                        if cfg.training.use_ema:
                            ema.step(self.model)

                        if self.optimizer_steps in archive_steps:
                            archive_path = pathlib.Path(self.output_dir).joinpath(
                                'checkpoints',
                                f'step={self.optimizer_steps:06d}.ckpt')
                            if not archive_path.exists():
                                _atomic_workspace_checkpoint(self, archive_path)
                            written_archive_steps.add(self.optimizer_steps)

                        # logging
                        raw_loss_cpu = raw_loss.item()
                        tepoch.set_postfix(loss=raw_loss_cpu, refresh=False)
                        train_losses.append(raw_loss_cpu)
                        step_log = {
                            'train_loss': raw_loss_cpu,
                            'global_step': self.global_step,
                            'epoch': self.epoch,
                            'lr': lr_scheduler.get_last_lr()[0]
                        }

                        is_last_batch = (batch_idx == (len(train_dataloader)-1))
                        if not is_last_batch:
                            # log of last step is combined with validation and rollout
                            if self.global_step % batch_log_every == 0:
                                wandb_run.log(step_log, step=self.global_step)
                                json_logger.log(step_log)
                            self.global_step += 1

                        if target_steps is not None and self.optimizer_steps >= int(target_steps):
                            break

                        if (cfg.training.max_train_steps is not None) \
                            and batch_idx >= (cfg.training.max_train_steps-1):
                            break
                
                # at the end of each epoch
                # replace train_loss with epoch average
                train_loss = np.mean(train_losses)
                step_log['train_loss'] = train_loss

                # ========= eval for this epoch ==========
                policy = self.model
                if cfg.training.use_ema:
                    policy = self.ema_model
                policy.eval()

                # Rollout uses multiple subprocesses and is the least reliable part of
                # an epoch. Persist training state synchronously before starting it so a
                # supervisor restart loses at most the current epoch. An atomic rename
                # keeps the previous checkpoint loadable if this process is interrupted
                # while torch is serializing the new one.
                is_checkpoint_epoch = (
                    self.epoch % cfg.training.checkpoint_every) == 0
                if is_checkpoint_epoch and cfg.checkpoint.save_last_ckpt:
                    ckpt_path = self.get_checkpoint_path()
                    # A recovery supervisor can overlap briefly with a stale
                    # process.  Never share a fixed temporary filename between
                    # writers: a distinct temp file makes the final replace
                    # atomic without allowing one writer to delete another
                    # writer's payload.
                    tmp_ckpt_path = ckpt_path.with_name(
                        ckpt_path.name + f'.{os.getpid()}.tmp')
                    previous_ckpt_path = ckpt_path.with_name(
                        ckpt_path.name + '.previous')
                    self.save_checkpoint(path=tmp_ckpt_path, use_thread=False)
                    if ckpt_path.is_file():
                        tmp_previous_path = previous_ckpt_path.with_name(
                            previous_ckpt_path.name + f'.{os.getpid()}.tmp')
                        _link_or_copy(ckpt_path, tmp_previous_path)
                        os.replace(tmp_previous_path, previous_ckpt_path)
                    os.replace(tmp_ckpt_path, ckpt_path)

                # run rollout
                skip_initial_rollout = (
                    self.epoch == 0
                    and cfg.training.get("skip_initial_rollout", False)
                )
                if (
                    (self.epoch % cfg.training.rollout_every) == 0
                    and not skip_initial_rollout
                ):
                    runner_log = env_runner.run(policy)
                    # log all
                    step_log.update(runner_log)

                # run validation
                if (self.epoch % cfg.training.val_every) == 0:
                    with torch.no_grad():
                        val_losses = list()
                        with tqdm.tqdm(val_dataloader, desc=f"Validation epoch {self.epoch}", 
                                leave=False, mininterval=cfg.training.tqdm_interval_sec) as tepoch:
                            for batch_idx, batch in enumerate(tepoch):
                                batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                                loss = self.model.compute_loss(batch)
                                val_losses.append(loss)
                                if (cfg.training.max_val_steps is not None) \
                                    and batch_idx >= (cfg.training.max_val_steps-1):
                                    break
                        if len(val_losses) > 0:
                            val_loss = torch.mean(torch.tensor(val_losses)).item()
                            # log epoch average validation loss
                            step_log['val_loss'] = val_loss

                # run diffusion sampling on a training batch
                if (self.epoch % cfg.training.sample_every) == 0:
                    with torch.no_grad():
                        # sample trajectory from training set, and evaluate difference
                        batch = train_sampling_batch
                        obs_dict = {'obs': batch['obs']}
                        gt_action = batch['action']
                        
                        result = policy.predict_action(obs_dict)
                        if cfg.pred_action_steps_only:
                            pred_action = result['action']
                            start = cfg.n_obs_steps - 1
                            end = start + cfg.n_action_steps
                            gt_action = gt_action[:,start:end]
                        else:
                            pred_action = result['action_pred']
                        mse = torch.nn.functional.mse_loss(pred_action, gt_action)
                        # log
                        step_log['train_action_mse_error'] = mse.item()
                        # release RAM
                        del batch
                        del obs_dict
                        del gt_action
                        del result
                        del pred_action
                        del mse
                
                # checkpoint
                if is_checkpoint_epoch:
                    # checkpointing
                    if cfg.checkpoint.save_last_snapshot:
                        self.save_snapshot()

                    # sanitize metric names
                    metric_dict = dict()
                    for key, value in step_log.items():
                        new_key = key.replace('/', '_')
                        metric_dict[new_key] = value
                    
                    # We can't copy the last checkpoint here
                    # since save_checkpoint uses threads.
                    # therefore at this point the file might have been empty!
                    topk_ckpt_path = topk_manager.get_ckpt_path(metric_dict)

                    if topk_ckpt_path is not None:
                        self.save_checkpoint(path=topk_ckpt_path)
                # ========= eval end for this epoch ==========
                policy.train()

                # end of epoch
                # log of last step is combined with validation and rollout
                wandb_run.log(step_log, step=self.global_step)
                json_logger.log(step_log)
                if target_steps is None:
                    self.global_step += 1
                else:
                    self.global_step = self.optimizer_steps
                self.epoch += 1

            if archive_steps and written_archive_steps != archive_steps:
                missing = sorted(archive_steps - written_archive_steps)
                raise RuntimeError(
                    f'Missing exact archive checkpoints for optimizer steps: {missing}')

            # A durable, run-internal completion witness.  The outer launcher
            # may be reclaimed after Python exits; this marker is written only
            # after the exact-step archive invariant above has passed.
            if target_steps is not None and self.optimizer_steps == int(target_steps):
                completion_path = pathlib.Path(self.output_dir).joinpath(
                    'training_complete.json')
                completion_tmp = completion_path.with_name(
                    completion_path.name + f'.{os.getpid()}.tmp')
                completion_tmp.write_text(json.dumps({
                    'optimizer_steps': int(self.optimizer_steps),
                    'target_optimizer_steps': int(target_steps),
                    'archive_checkpoint_steps': sorted(archive_steps),
                    'training_seed': int(cfg.training.seed),
                }, sort_keys=True) + '\n')
                os.replace(completion_tmp, completion_path)

@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.parent.joinpath("config")), 
    config_name=pathlib.Path(__file__).stem)
def main(cfg):
    workspace = TrainDiffusionUnetLowdimWorkspace(cfg)
    workspace.run()

if __name__ == "__main__":
    main()
