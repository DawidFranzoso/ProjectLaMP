import contextlib
import copy
import itertools
from datetime import datetime
import json
import os
import random
from itertools import count
from typing import Literal, Iterable, Iterator, Optional, Callable, List
from types import MethodType
import numpy as np
import torch
from torch import nn
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, T5ForConditionalGeneration

from DataDownloader import DataDownloader


class StyleOracleTrainer:
    @staticmethod
    def from_pretrained_flan_t5(device: torch.device, size: str = "small", mean_pooling: bool = True):
        # my autistic way to ensure the weights are completely detached from each-other is using a factory twice :O)
        def load_model():
            print("Loading model!")
            return AutoModelForSeq2SeqLM.from_pretrained(f"google/flan-t5-{size}", cache_dir="..").to(device)

        print("Loading tokenizer!")
        tokenizer = AutoTokenizer.from_pretrained(f"google/flan-t5-{size}", cache_dir="..")

        style_oracle=load_model().encoder
        classifier_model=load_model()
        print("Loading complete")

        return StyleOracleTrainer(
            tokenizer=tokenizer,
            style_oracle=style_oracle,
            classifier_model=classifier_model,
            oracle_mean_pooling=mean_pooling,
        )

    def __init__(self, tokenizer,
                 style_oracle: nn.Module,
                 classifier_model: nn.Module,
                 oracle_mean_pooling: bool = True):
        self.tokenizer = tokenizer
        self.style_oracle = style_oracle  # encoder-only
        self.classifier_model = classifier_model  # encoder-decoder
        self.oracle_mean_pooling = oracle_mean_pooling

        assert not hasattr(self.style_oracle, "encoder")
        assert hasattr(self.classifier_model, "decoder")

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

    def contrastive_sample_generator(self, dataset: List, batch_size: int = 64,

                                     # I believe in torch's ability to not retrace graph 5000 times like tf does
                                     padding: str = "longest",
                                     ):
        while True:
            batch_anchors = []
            batch_positive = []
            batch_negative = []
            for positive_sample_id, negative_sample_id in StyleOracleTrainer.get_contrastive_index_pairs(
                    dataset_length=len(dataset)):
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
                    anchor_tokens = self.tokenizer(batch_anchors, padding=padding,
                                                   return_tensors="pt").data
                    positive_tokens = self.tokenizer(batch_positive, padding=padding,
                                                     return_tensors="pt").data
                    negative_tokens = self.tokenizer(batch_negative, padding=padding,
                                                     return_tensors="pt").data

                    batch_anchors.clear()
                    batch_positive.clear()
                    batch_negative.clear()

                    yield {
                        "anchor": anchor_tokens,
                        "positive": positive_tokens,
                        "negative": negative_tokens,
                    }

    def predict_style_tensor(self, encoder_input: torch.Tensor, encoder_attention_mask: torch.Tensor) -> torch.Tensor:
        assert encoder_input.shape == encoder_attention_mask.shape

        encoder_representations = self.style_oracle(
            input_ids=encoder_input.flatten(0, -2),
            attention_mask=encoder_attention_mask.flatten(0, -2)
        ).last_hidden_state

        encoder_representations = encoder_representations.view(*encoder_input.shape, encoder_representations.shape[-1])

        if self.oracle_mean_pooling:
            # mean pool over non-padding tokens
            # (we use mean pool because T5 has no cls token afaik)
            representation_weight_mask = (
                encoder_attention_mask
                / torch.maximum(torch.ones(()), encoder_attention_mask.sum(-1, keepdim=True))
            )
            return (representation_weight_mask[..., None] * encoder_representations).sum(-2)
        else:
            # if not mean pooling, just return first (any) token. Using the first one has the advantage that it
            # is always the same position. Since encoder is not causal,
            # the first token can contain as much as any other token
            return encoder_representations[..., 0, :]

    @staticmethod
    def build_triplet_loss(positive_weight: float = 1, triplet_alpha: float = 0.9) -> Callable[
        [torch.Tensor, torch.Tensor, torch.Tensor], dict]:
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
                    - positive_weight * torch.minimum(positive_similarity, broadcast_triplet_alpha)
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
                  dataset: List,
                  batch_size: int,
                  loss_fn: Callable,
                  triplet_mode: bool,
                  optimizer: Optional[torch.optim.Optimizer],
                  lr_scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
                  log_freq: int = 10,
                  cap_steps: Optional[int] = None
                  ):
        is_training = optimizer is not None

        if is_training:
            self.style_oracle.train()
            self.classifier_model.train()
        else:
            self.style_oracle.eval()
            self.classifier_model.eval()

        def get_metrics_dict():
            prefix = "" if is_training else "val_"
            ret = {
                f"{prefix}total_steps": total_steps,
                f"{prefix}total_samples": total_steps * batch_size,
                f"{prefix}loss": loss_metric.item(),
            }
            if triplet_mode:
                ret |= {
                    f"{prefix}positive_similarity": pos_sim_metric.item(),
                    f"{prefix}negative_similarity": neg_sim_metric.item(),
                }
            return ret

        with self.device:
            loss_metric = nn.Parameter(torch.zeros(()), requires_grad=False)
            if triplet_mode:
                pos_sim_metric = nn.Parameter(torch.zeros(()), requires_grad=False)
                neg_sim_metric = nn.Parameter(torch.zeros(()), requires_grad=False)
            else:
                # rouge = evaluate.load('rouge')
                pass

            steps_metric = nn.Parameter(torch.zeros(()), requires_grad=False)

            if triplet_mode:
                dataset = self.contrastive_sample_generator(dataset=dataset, batch_size=batch_size)
            else:
                def batched_dataset(unbatched_dataset):
                    keys = ("input", "label", "profile")
                    empty = {k: [] for k in keys}
                    ret = copy.copy(empty)
                    for s in unbatched_dataset:
                        for k in keys:
                            ret[k].append(s[k])

                        if len(ret["input"]) >= batch_size:
                            yield ret
                            ret = copy.copy(empty)

                dataset = batched_dataset(dataset)

            for total_steps, sample in enumerate(dataset, start=1):
                if not triplet_mode:
                    input_ = self.tokenizer(sample["input"], padding="longest", return_tensors="pt").data
                    label = self.tokenizer(label_raw := sample["label"], padding="longest", return_tensors="pt").data

                    # batching and padding shenanigans
                    profile_sizes = list(map(len, sample["profile"]))

                    flat_profiles = self.tokenizer(
                        list(itertools.chain.from_iterable(sample["profile"])),
                        padding="longest",
                        return_tensors="pt"
                    ).data

                    profiles = {
                        k: [
                            flat_profiles[k][start:end]
                            for start, end in zip([0, *np.cumsum(profile_sizes)[:-1]], [*np.cumsum(profile_sizes)])
                        ] for k in flat_profiles.keys()
                    }

                    max_profile_tokens = max(map(len, profiles["attention_mask"]))
                    max_profile_tokens = 6  # TODO: remove or make a variable (debug)

                    profiles = {
                        k: torch.stack([
                            torch.constant_pad_nd(
                                p[:max_profile_tokens],  # in case I decide to limit max profile tokens later

                                # I could make it cut from the left only, but idc enough to do it
                                # I prefer readability
                                pad=[0, 0, 0, max_profile_tokens - len(p)],
                                value=0
                            ) for p in profiles[k]
                        ], dim=0)
                        for k in profiles.keys()
                    }

                    sample = {"profile": profiles}

                with ([contextlib.nullcontext, torch.enable_grad][is_training])():
                    with ([torch.no_grad, contextlib.nullcontext][triplet_mode])():
                        predicted_styles = {
                            sample_role: self.predict_style_tensor(
                                encoder_input=data["input_ids"].to(self.device),
                                encoder_attention_mask=data["attention_mask"].to(self.device),
                            )
                            for sample_role, data in sample.items()
                        }

                    if triplet_mode:
                        loss_info = loss_fn(
                            anchor_style=predicted_styles["anchor"],
                            positive_style=predicted_styles["positive"],
                            negative_style=predicted_styles["negative"],
                        )  # I expect no nans
                    else:
                        encoder_outputs: torch.Tensor = self.classifier_model.encoder(
                            input_ids=input_["input_ids"],
                            attention_mask=input_["attention_mask"],
                        )["last_hidden_state"]

                        style_vectors = predicted_styles["profile"]

                        encoder_attention_mask = torch.concat(
                            tensors=[
                                torch.ones_like(encoder_outputs[..., 0], dtype=sample["profile"]["attention_mask"].dtype),
                                sample["profile"]["attention_mask"][..., 0]
                            ],
                            dim=-1,
                        )
                        encoder_outputs = torch.concat([encoder_outputs, style_vectors], dim=-2)

                        decoder_input = torch.constant_pad_nd(
                            label["input_ids"],
                            [1, 0],
                            self.classifier_model.generation_config.decoder_start_token_id
                        )[..., :-1]

                        outputs = self.classifier_model(
                            decoder_input_ids=decoder_input,
                            encoder_outputs=(encoder_outputs,),
                            attention_mask=encoder_attention_mask,
                            labels=label["input_ids"],
                        )

                        # TODO: implement rouge
                        """  
                        generation = self.classifier_model.generate(
                            decoder_input_ids=decoder_input,
                            encoder_outputs=encoder_outputs,
                            num_beams=4,
                            max_length=50,
                        )"""

                        loss_info = {
                            "loss": outputs.loss,
                            # "rouge": rouge.compute(
                            #     predictions=[...],
                            #     references=[label_raw],
                            #     use_aggregator=True,
                            # )
                        }

                loss_info["loss"].mean().backward()
                if is_training:
                    optimizer.step()
                if is_training and lr_scheduler is not None:
                    lr_scheduler.step()

                StyleOracleTrainer.update_metric(
                    step=steps_metric, value=loss_info["loss"].detach().mean(), aggregate=loss_metric
                )
                if triplet_mode:
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
                    if cap_steps <= steps_metric:
                        break

            return get_metrics_dict()

    def run_ce_epoch(self,
                     dataset: List,
                     batch_size: int,
                     loss_fn: Callable,
                     optimizer: Optional[torch.optim.Optimizer],
                     lr_scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
                     log_freq: int = 10,
                     ):
        ...

    def run(self,
            triplet_mode: bool,
            epochs: int,
            batch_size: int,
            optimizer_factory: Callable[[Iterator[nn.Parameter]], torch.optim.Optimizer],
            lr_scheduler_factory: Optional[
                Callable[[torch.optim.Optimizer], torch.optim.lr_scheduler.LRScheduler]] = None,
            loss_fn: Callable = None,
            log_freq: int = 10,
            run_name: str = None,
            ):
        if loss_fn is None:
            if triplet_mode:
                loss_fn = StyleOracleTrainer.build_triplet_loss()
            else:
                loss_fn = None  # nn.CrossEntropyLoss() deprecated

        if run_name is None:
            run_name = f"{datetime.now().replace(microsecond=0)}"

        optimizer = optimizer_factory([self.classifier_model, self.style_oracle][triplet_mode].parameters())
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
                triplet_mode=triplet_mode,
                optimizer=optimizer,
                lr_scheduler=lr_scheduler,
                log_freq=log_freq,
                cap_steps=(10437 // batch_size) if triplet_mode else None,
            )
            print(f"Validation ({epoch=})")
            metrics_dict |= self.run_epoch(
                dataset=dataset_val,
                batch_size=batch_size,
                loss_fn=loss_fn,
                triplet_mode=triplet_mode,
                optimizer=None,  # validation
                lr_scheduler=None,
                log_freq=log_freq,
                cap_steps=(1500 // batch_size) if triplet_mode else None,
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
