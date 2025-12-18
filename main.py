import os
import random
from datetime import datetime
import torch.cuda

import torch.optim

import wandb

from DataDownloader import DataDownloader

from StyleOracleTrainer import StyleOracleTrainer

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# I was asked to do this
random.seed(42)
torch.manual_seed(42)

wandb_api_key = os.environ.get("WANDB_API_KEY", None)
device = torch.device(os.environ.get("DEVICE", "cuda"))
print(f"{torch.cuda.is_available() = }")
print(f"{wandb_api_key=}")
print(f"{torch.__version__=}")
model_size = os.environ.get("MODEL_SIZE", "large")

run_name = f"{datetime.now().replace(microsecond=0)}"
if wandb_api_key is not None:
    wandb.login(key=wandb_api_key)
    wandb.init(
        resume="never",

        # job_type=None,
        dir=None,
        config={},  # TODO: maybe todo
        project="ExtremelyCoolLampProject",
        entity="dawidfranzoso",
        reinit=True,
        # tags=[f"hp_opt{hp_optimizer_run_id}"] + [[], ["hp_opt_baseline"]][hp_constant_run_id == 1],
        # group=f"hp_opt{hp_optimizer_run_id}",
        name=run_name,
        # notes=None,
        # magic=None,
        # config_exclude_keys=None,
        # config_include_keys=None,
        anonymous="never",
        mode="online",
        # to override mode, enter it as argument :) ex."disabled"
        allow_val_change=False,
        # One day I will add learning rate instead of default learning rate to configs :)
        # force=None,
        # tensorboard=None,
        # sync_tensorboard=None,
        # monitor_gym=None,
        save_code=False,
        # id=run_id,
        # settings=None
    )


epochs = 20  # from paper
batch_size = 64
warmup_steps_triplet = round(0.05 * (10437 / batch_size) * epochs)  # from paper (lamp-7 training set size = 10437)
warmup_steps_ce = round(0.05 * 10437 * epochs)  # from paper (lamp-7 training set size = 10437) (no batch size yet)

DataDownloader.maybe_download_all(tasks=[7])

run_id = f"{datetime.now().replace(microsecond=0)}"

baseline_trainer = StyleOracleTrainer.from_pretrained_flan_t5(device=device, size=model_size)
baseline_trainer.run(
    run_name=f"baseline_{run_id}",
    triplet_mode=False,
    epochs=epochs,
    batch_size=1,
    optimizer_factory=lambda params: torch.optim.AdamW(  # from paper
        params=params,
        lr=5e-5,  # from paper
        weight_decay=1e-4,  # from paper
    ),
    lr_scheduler_factory=lambda optimizer: torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: min(warmup_steps_ce, step) / warmup_steps_ce
    ),
)
del baseline_trainer

style_oracle_trainer = StyleOracleTrainer.from_pretrained_flan_t5(device=device, size=model_size)
style_oracle_trainer.run(
    run_name=f"oracle_1triplet_{run_id}",
    triplet_mode=True,
    epochs=epochs,
    batch_size=batch_size,
    optimizer_factory=lambda params: torch.optim.AdamW(  # from paper
        params=params,
        lr=5e-5,  # from paper
        weight_decay=1e-4,  # from paper
    ),
    lr_scheduler_factory=lambda optimizer: torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: min(warmup_steps_triplet, step) / warmup_steps_triplet
    ),
    loss_fn=StyleOracleTrainer.build_triplet_loss(positive_weight=4)
)
style_oracle_trainer.run(
    run_name=f"oracle_2classifier_{run_id}",
    triplet_mode=False,
    epochs=epochs,
    batch_size=1,
    optimizer_factory=lambda params: torch.optim.AdamW(  # from paper
        params=params,
        lr=5e-5,  # from paper
        weight_decay=1e-4,  # from paper
    ),
    lr_scheduler_factory=lambda optimizer: torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: min(warmup_steps_ce, step) / warmup_steps_ce
    ),
)

# TODO: speedrun
#  rouge from evaluate
#  powerpoint B)
#  present this shat
#  gf christmas gift
#  Marko's birthday
