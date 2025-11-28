import copy
import json
import os
import random
from datetime import datetime
from inspect import Parameter
import torch.cuda

import torch.optim

import wandb
from dotenv import load_dotenv
from torch import nn

from DataDownloader import DataDownloader

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

load_dotenv()

# I was asked to do this
random.seed(42)
torch.manual_seed(42)

wandb_api_key = os.environ.get("WANDB_API_KEY", None)
device = torch.device(os.environ.get("DEVICE", "cuda"))
print(f"{torch.cuda.is_available() = }")
model_size = os.environ.get("MODEL_SIZE", "xxl")

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


DataDownloader.maybe_download_all(tasks=[7])

x, y = [
    DataDownloader.DataEntry(
        task_number=7,
        split_name="training",
        is_time_based=False,
        is_output=is_output,
    ).load_json() for is_output in (False, True)
]


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


dataset = map(to_sample, zip(x, y["golds"]))
dataset = filter(lambda sample: not isinstance(sample, Exception), dataset)

dataset = list(dataset)  # the task7 dataset is very small so it's ok


def load_model(size: str = "small"):
    model = AutoModelForSeq2SeqLM.from_pretrained(f"google/flan-t5-{size}", cache_dir="..").to(device)
    print("Loading tokenizer!")
    tokenizer = AutoTokenizer.from_pretrained(f"google/flan-t5-{size}", cache_dir="..")
    print("Loaded tokenizer!")

    return model, tokenizer


style_oracle, style_oracle_tokenizer = load_model(size=model_size)
# prediction_model, prediction_tokenizer = load_model(size=model_size)


# since I am training it on ZIH I allow myself to be computation-inefficient for ease of development
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


def contrastive_sample_generator(dataset, batch_size: int = 64,

                                 # I believe in torch's ability to not retrace graph 5000 times like tf does
                                 padding: str = "longest",
                                 ):
    while True:
        batch_anchors = []
        batch_positive = []
        batch_negative = []
        for positive_sample_id, negative_sample_id in get_contrastive_index_pairs(dataset_length=len(dataset)):
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
                anchor_tokens = style_oracle_tokenizer(batch_anchors, padding=padding, return_tensors="pt").data
                positive_tokens = style_oracle_tokenizer(batch_positive, padding=padding, return_tensors="pt").data
                negative_tokens = style_oracle_tokenizer(batch_negative, padding=padding, return_tensors="pt").data

                batch_anchors.clear()
                batch_positive.clear()
                batch_negative.clear()

                yield {
                    "anchor": anchor_tokens,
                    "positive": positive_tokens,
                    "negative": negative_tokens,
                }


def predict_style_tensor(encoder_input: torch.Tensor, encoder_attention_mask: torch.Tensor, mean_pooling: bool = True) -> torch.Tensor:
    encoder_representations = style_oracle(
        input_ids=encoder_input,
        attention_mask=encoder_attention_mask,

        decoder_input_ids=torch.zeros_like(encoder_input[..., :1])
    ).encoder_last_hidden_state

    if mean_pooling:
        # mean pool over non-padding tokens
        # (we use mean pool because T5 has no cls token afaik)
        representation_weight_mask = encoder_attention_mask / encoder_attention_mask.sum(-1, keepdim=True)
        return (representation_weight_mask[..., None] * encoder_representations).sum(-2)
    else:
        # if not mean pooling, just return first token
        return encoder_representations[..., 0, :]


def update_metric(value: torch.Tensor, aggregate: nn.Parameter):
    global steps_metric
    # aggregate = (
    #       n / (n + 1) * aggregate
    #       + 1 / (n + 1) * value
    # )

    n = steps_metric  # rename for clarity

    aggregate *= n / (n + 1)
    aggregate += 1 / (n + 1) * value


with device:
    style_oracle_optimizer = torch.optim.AdamW(
        params=style_oracle.parameters(),  # TODO: only pass encoder params
        lr=5e-5,  # from paper
        weight_decay=1e-4,
    )
    style_similarity = torch.nn.CosineSimilarity(dim=-1, eps=1e-08)

    style_oracle.train()
    log_freq = 5000

    loss_metric = nn.Parameter(torch.zeros(()), requires_grad=False)
    pos_sim_metric = nn.Parameter(torch.zeros(()), requires_grad=False)
    neg_sim_metric = nn.Parameter(torch.zeros(()), requires_grad=False)
    steps_metric = nn.Parameter(torch.zeros(()), requires_grad=False)

    for total_steps, contrastive_sample in enumerate(contrastive_sample_generator(dataset=dataset, batch_size=64), start=1):
        with torch.enable_grad():
            print(total_steps)
            predicted_styles = {
                sample_role: predict_style_tensor(
                    encoder_input=data["input_ids"].to(device),
                    encoder_attention_mask=data["attention_mask"].to(device),
                )
                for sample_role, data in contrastive_sample.items()
            }

            positive_similarity = style_similarity(predicted_styles["anchor"], predicted_styles["positive"])
            negative_similarity = style_similarity(predicted_styles["anchor"], predicted_styles["negative"])
            triplet_epsilon = 0.9
            triplet_epsilon = torch.zeros_like(positive_similarity) + triplet_epsilon  # broadcast
            loss = (
                negative_similarity
                - torch.minimum(positive_similarity, triplet_epsilon)
            ).mean()  # I expect no nans

        loss.backward()
        style_oracle_optimizer.step()

        update_metric(value=loss.detach(), aggregate=loss_metric)
        update_metric(value=positive_similarity.detach().mean(), aggregate=pos_sim_metric)
        update_metric(value=negative_similarity.detach().mean(), aggregate=neg_sim_metric)

        steps_metric += 1

        if steps_metric >= log_freq:
            metrics_dict = {
                "total_steps": total_steps,
                "total_samples": total_steps * (batch_size := contrastive_sample["anchor"]["input_ids"].shape[0]),
                "loss": loss_metric.item(),
                "positive_similarity": pos_sim_metric.item(),
                "negative_similarity": neg_sim_metric.item(),
            }
            print(json.dumps(metrics_dict, indent=4))
            if wandb_api_key is not None:
                wandb.log(metrics_dict)
            steps_metric *= 0  # resets metrics
            model_filepath = f"outputs/style_oracle/{run_name}_step{total_steps}.pt"
            os.makedirs("/".join(model_filepath.split("/")[:-1]), exist_ok=True)
            torch.save(obj=style_oracle, f=model_filepath)
