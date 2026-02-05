import json
import os
import random
from datetime import datetime
import torch.cuda

import torch.optim

from DataDownloader import DataDownloader

from StyleOracleTrainer import StyleOracleTrainer
import argparse

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# I was asked to do this
random.seed(42)
torch.manual_seed(42)

parser = argparse.ArgumentParser("python main.py")
parser.add_argument("--max_recency", help="Max amount of recent profiles (tweets) used.", type=int, default=4, required=False)
parser.add_argument("--baseline", action="store_true", required=False)
parser.add_argument("--triplet_mode_weight_regularization", type=float, default=0, required=False)
args = parser.parse_args()

args.baseline = True  # TODO: remove (debug)

wandb_api_key = os.environ.get("WANDB_API_KEY", None)
device = torch.device(os.environ.get("DEVICE", "cuda"))
print(f"{torch.cuda.is_available() = }")
print(f"{wandb_api_key=}")
print(f"{torch.__version__=}")
model_size = os.environ.get("MODEL_SIZE", "large")
override_batch_size = os.environ.get("OVERRIDE_BATCH_SIZE", None)

run_name = f"{datetime.now().replace(microsecond=0)}"
if wandb_api_key is not None:
    import wandb

    wandb.login(key=wandb_api_key)
    wandb.init(
        resume="never",

        # job_type=None,
        dir=None,
        config=vars(args),
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

if override_batch_size is not None:
    batch_size = int(override_batch_size)
    assert batch_size > 0

DataDownloader.maybe_download_all(tasks=[7])

run_id = f"{datetime.now().replace(microsecond=0)}"

if args.baseline:
    print("Baseline training")
    baseline_trainer = StyleOracleTrainer.from_pretrained_flan_t5(device=device, size=model_size)
    baseline_trainer.run(
        run_name=f"baseline_{run_id}_{vars(args)}",
        triplet_mode=False,
        epochs=epochs,
        batch_size=batch_size,
        optimizer_factory=lambda params: torch.optim.AdamW(  # from paper
            params=params,
            lr=5e-5,  # from paper
            weight_decay=1e-4,  # from paper
        ),
        lr_scheduler_factory=lambda optimizer: torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lambda step: min(warmup_steps_ce, step) / warmup_steps_ce
        ),
        max_recency=args.max_recency,
    )
else:
    print("Weaved training")
    style_oracle_trainer = StyleOracleTrainer.from_pretrained_flan_t5(device=device, size=model_size)

    weave_history = []

    for epoch in range(1, 1 + epochs):
        weave_metrics = {}

        weave_metrics["triplet"] = style_oracle_trainer.run(
            run_name=f"oracle_1triplet_{run_id}_{vars(args)}_weave{epoch}",
            triplet_mode=True,
            epochs=1,
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
            loss_fn=StyleOracleTrainer.build_triplet_loss(positive_weight=4),
            max_recency=args.max_recency,
            triplet_mode_weight_regularization=args.triplet_mode_weight_regularization,
        )

        weave_metrics["classifier"] = style_oracle_trainer.run(
            run_name=f"oracle_2classifier_{run_id}_{vars(args)}_weave{epoch}",
            triplet_mode=False,
            epochs=1,
            batch_size=batch_size,
            optimizer_factory=lambda params: torch.optim.AdamW(  # from paper
                params=params,
                lr=5e-5,  # from paper
                weight_decay=1e-4,  # from paper
            ),
            lr_scheduler_factory=lambda optimizer: torch.optim.lr_scheduler.LambdaLR(
                optimizer,
                lambda step: min(warmup_steps_ce, step) / warmup_steps_ce
            ),
            max_recency=args.max_recency,
        )

        weave_history.append(weave_metrics)

        with open(f"outputs/weaved/{run_id}_{vars(args)}_weave{epoch}.history", "w") as f:
            json.dump(weave_history, f)


# TODO: speedrun
#  rouge from evaluate
#  present this shat
