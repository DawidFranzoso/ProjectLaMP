import contextlib
import copy
from datetime import datetime
import json
import os
import random
from typing import Literal, Iterable, Iterator, Optional, Callable

import torch
from torch import nn
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from DataDownloader import DataDownloader


class StyleOracleTrainer:
    @staticmethod
    def from_pretrained_flan_t5(device: torch.device, size: str = "small", mean_pooling: bool = True):
        model = AutoModelForSeq2SeqLM.from_pretrained(f"google/flan-t5-{size}", cache_dir="..").to(device)
        print("Loading tokenizer!")
        tokenizer = AutoTokenizer.from_pretrained(f"google/flan-t5-{size}", cache_dir="..")
        print("Loaded tokenizer!")

        return StyleOracleTrainer(
            style_oracle_tokenizer=tokenizer,
            style_oracle=model,
            mean_pooling=mean_pooling,
        )

    def __init__(self, style_oracle_tokenizer, style_oracle: nn.Module, mean_pooling: bool = True):
        self.style_oracle_tokenizer = style_oracle_tokenizer
        self.style_oracle = style_oracle
        self.mean_pooling = mean_pooling

    @property
    def device(self):
        return self.style_oracle.device

    @staticmethod
    def update_metric(step, value: torch.Tensor, aggregate: nn.Parameter):
        # aggregate = (
        #       n / (n + 1) * aggregate
        #       + 1 / (n + 1) * value
        # )

        n = step  # rename for clarity

        aggregate *= n / (n + 1)
        aggregate += 1 / (n + 1) * value

    @staticmethod
    def to_sample(x__y_pair: tuple):
        try:
            x, y = x__y_pair
            assert x["id"] == y["id"]
            return {
                "input": x["input"],
                "label": y["output"],
                "profile": list(map(lambda d: d["text"], x["profile"])),
            }
        except Exception as e:
            return e

    @staticmethod
    def get_supervised_dataset(task_number: int,
                               split_name: Literal["training", "validation", "test"],
                               is_time_based: bool,
                               ) -> Iterator:
            x, y = [
                DataDownloader.DataEntry(
                    task_number=task_number,
                    split_name=split_name,
                    is_time_based=is_time_based,
                    is_output=is_output,
                ).load_json() for is_output in (False, True)
            ]

            dataset = map(StyleOracleTrainer.to_sample, zip(x, y["golds"]))
            dataset = filter(lambda sample: not isinstance(sample, Exception), dataset)

            return dataset

    # since I am training it on ZIH I allow myself to be computation-inefficient for ease of development
    @staticmethod
    def get_contrastive_index_pairs(dataset_length: int):
        positive_indices = list(range(dataset_length))
        negative_indices = list(range(dataset_length))

        random.shuffle(positive_indices)
        random.shuffle(negative_indices)

        # we randomize it in a way that pos != neg
        while len(positive_indices) > 0:
            if positive_indices[0] != negative_indices[0]:
                yield positive_indices.pop(0), negative_indices.pop(0)
            else:
                if len(positive_indices) == 1:
                    break  # we lose our last sample, because there is no way to cycle it away
                negative_indices.append(
                    negative_indices.pop(0))  # we cycle to guarantee an index non-equal to positive index

    @staticmethod
    def contrastive_sample_generator(self, dataset, batch_size: int = 64,

                                     # I believe in torch's ability to not retrace graph 5000 times like tf does
                                     padding: str = "longest",
                                     ):
        while True:
            batch_anchors = []
            batch_positive = []
            batch_negative = []
            for positive_sample_id, negative_sample_id in StyleOracleTrainer.get_contrastive_index_pairs(dataset_length=len(dataset)):
                assert positive_sample_id != negative_sample_id
                positive_sample = dataset[positive_sample_id]
                negative_sample = dataset[negative_sample_id]

                try:
                    positive_profile = copy.deepcopy(positive_sample["profile"])
                    negative = random.choice(negative_sample["profile"])
                except KeyError:
                    continue

                if len(positive_profile) <= 1:
                    continue  # we need at least 2

                anchor = random.choice(positive_profile)
                positive_profile.remove(anchor)
                positive = random.choice(positive_profile)

                batch_anchors.append(anchor)
                batch_positive.append(positive)
                batch_negative.append(negative)

                if len(batch_anchors) >= batch_size:
                    anchor_tokens = self.style_oracle_tokenizer(batch_anchors, padding=padding, return_tensors="pt").data
                    positive_tokens = self.style_oracle_tokenizer(batch_positive, padding=padding, return_tensors="pt").data
                    negative_tokens = self.style_oracle_tokenizer(batch_negative, padding=padding, return_tensors="pt").data

                    batch_anchors.clear()
                    batch_positive.clear()
                    batch_negative.clear()

                    yield {
                        "anchor": anchor_tokens,
                        "positive": positive_tokens,
                        "negative": negative_tokens,
                    }

    def predict_style_tensor(self, encoder_input: torch.Tensor, encoder_attention_mask: torch.Tensor) -> torch.Tensor:
        encoder_representations = self.style_oracle(
            input_ids=encoder_input,
            attention_mask=encoder_attention_mask,

            decoder_input_ids=torch.zeros_like(encoder_input[..., :1])
        ).encoder_last_hidden_state

        if self.mean_pooling:
            # mean pool over non-padding tokens
            # (we use mean pool because T5 has no cls token afaik)
            representation_weight_mask = encoder_attention_mask / encoder_attention_mask.sum(-1, keepdim=True)
            return (representation_weight_mask[..., None] * encoder_representations).sum(-2)
        else:
            # if not mean pooling, just return first token
            return encoder_representations[..., 0, :]

    @staticmethod
    def build_triplet_loss(positive_weight: float = 1, triplet_alpha: float = 0.9) -> Callable[[torch.Tensor, torch.Tensor, torch.Tensor], dict]:
        positive_weight = positive_weight / positive_weight + 1
        negative_weight = 1 / positive_weight + 1

        style_similarity = torch.nn.CosineSimilarity(dim=-1, eps=1e-08)

        def loss_fn(anchor_style: torch.Tensor, positive_style: torch.Tensor, negative_style: torch.Tensor):
            positive_similarity = style_similarity(anchor_style, positive_style)
            negative_similarity = style_similarity(anchor_style, negative_style)

            broadcast_triplet_alpha = torch.zeros_like(positive_similarity) + triplet_alpha  # broadcast

            # no nans expected
            loss = (
                negative_weight * negative_similarity
                -positive_weight * torch.minimum(positive_similarity, broadcast_triplet_alpha)
            )

            return {
                "loss": loss,
                "positive_similarity": positive_similarity,
                "negative_similarity": negative_similarity,
                "triplet_alpha": triplet_alpha,
                "positive_weight": positive_weight,
                "negative_weight": negative_weight,
            }

        return loss_fn

    def run_epoch(self,
                  dataset: Iterable,
                  batch_size: int,
                  loss_fn: Callable,
                  optimizer: Optional[torch.optim.Optimizer],
                  lr_scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
                  log_freq: int = 10,
                  cap_steps: Optional[int] = None
                  ):
        is_training = optimizer is not None

        if is_training:
            self.style_oracle.train()
        else:
            self.style_oracle.eval()

        def get_metrics_dict():
            prefix = "" if is_training else "val_"
            return {
                f"{prefix}total_steps": total_steps,
                f"{prefix}total_samples": total_steps * batch_size,
                f"{prefix}loss": loss_metric.item(),
                f"{prefix}positive_similarity": pos_sim_metric.item(),
                f"{prefix}negative_similarity": neg_sim_metric.item(),
            }

        with self.device:
            loss_metric = nn.Parameter(torch.zeros(()), requires_grad=False)
            pos_sim_metric = nn.Parameter(torch.zeros(()), requires_grad=False)
            neg_sim_metric = nn.Parameter(torch.zeros(()), requires_grad=False)
            steps_metric = nn.Parameter(torch.zeros(()), requires_grad=False)

            for total_steps, contrastive_sample in enumerate(
                    StyleOracleTrainer.contrastive_sample_generator(dataset=dataset, batch_size=batch_size), start=1
            ):
                with ([contextlib.nullcontext, torch.enable_grad][is_training])():
                    predicted_styles = {
                        sample_role: self.predict_style_tensor(
                            encoder_input=data["input_ids"].to(self.device),
                            encoder_attention_mask=data["attention_mask"].to(self.device),
                        )
                        for sample_role, data in contrastive_sample.items()
                    }

                    loss_info = loss_fn(
                        anchor_style=predicted_styles["anchor"],
                        positive_style=predicted_styles["positive"],
                        negative_style=predicted_styles["negative"],
                    ).mean()  # I expect no nans

                loss_info["loss"].backward()
                if is_training:
                    optimizer.step()
                if is_training and lr_scheduler is not None:
                    lr_scheduler.step()

                StyleOracleTrainer.update_metric(
                    step=steps_metric, value=loss_info["loss"].detach(), aggregate=loss_metric
                )
                StyleOracleTrainer.update_metric(
                    step=steps_metric, value=loss_info["positive_similarity"].detach().mean(), aggregate=pos_sim_metric
                )
                StyleOracleTrainer.update_metric(
                    step=steps_metric, value=loss_info["negative_similarity"].detach().mean(), aggregate=neg_sim_metric
                )

                steps_metric += 1

                if steps_metric % log_freq == 0:
                    print(json.dumps(get_metrics_dict(), indent=4))

                if cap_steps is not None:
                    if steps_metric <= steps_metric:
                        break

            return get_metrics_dict()

    def run(self,
            epochs: int,
            batch_size: int,
            optimizer_factory: Callable[[Iterator[nn.Parameter]], torch.optim.Optimizer],
            lr_scheduler_factory: Optional[Callable[[torch.optim.Optimizer], torch.optim.lr_scheduler.LRScheduler]] = None,
            loss_fn: Callable = None,
            log_freq: int = 10,
            run_name: str = None
            ):
        if loss_fn is None:
            loss_fn = StyleOracleTrainer.build_triplet_loss()

        if run_name is None:
            run_name = f"{datetime.now().replace(microsecond=0)}"

        optimizer = optimizer_factory(self.style_oracle.parameters())  # TODO: only pass encoder params
        if lr_scheduler_factory:
            lr_scheduler = lr_scheduler_factory(optimizer)
        else:
            lr_scheduler = None

        history = []

        # noinspection PyTypeChecker
        dataset_tng, dataset_val = (
            list(  # the datasets are small so its fine to load them all at once
                StyleOracleTrainer.get_supervised_dataset(
                    task_number=7,
                    split_name=split_name,
                    is_time_based=False,
                )
            ) for split_name in ("training", "validation")
        )

        for epoch in range(1, 1 + epochs):
            print(f"Training ({epoch=})")
            metrics_dict = {"epoch": epoch}
            metrics_dict |= self.run_epoch(
                dataset=dataset_tng,
                batch_size=batch_size,
                loss_fn=loss_fn,
                optimizer=optimizer,
                lr_scheduler=lr_scheduler,
                log_freq=log_freq,
                cap_steps=10437 // batch_size,
            )
            print(f"Validation ({epoch=})")
            metrics_dict |= self.run_epoch(
                dataset=dataset_tng,
                batch_size=batch_size,
                loss_fn=loss_fn,
                optimizer=None,  # validation
                lr_scheduler=None,
                log_freq=log_freq,
                cap_steps=1500 // batch_size,
            )

            print(json.dumps(metrics_dict, indent=4))

            # if wandb_api_key is not None:
            #     wandb.log(metrics_dict)

            history.append(metrics_dict)
            model_filepath = f"outputs/style_oracle/{run_name}_step{epoch}.pt"
            os.makedirs("/".join(model_filepath.split("/")[:-1]), exist_ok=True)
            torch.save(obj=self.style_oracle, f=model_filepath)
            with open(model_filepath + ".history", "w") as f:
                json.dump(history, f)
