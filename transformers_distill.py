import os
import time
from typing import Any
from dataclasses import dataclass
from itertools import chain

import torch

torch.set_float32_matmul_precision("high")

from torch.distributed import (
    init_process_group,
    destroy_process_group,
    get_world_size,
)

rank = int(os.environ.get("RANK", -1))
device = torch.device("cuda", rank)
torch.cuda.set_device(device)
init_process_group(backend="nccl")
world_size = get_world_size()

from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    FPQuantConfig,
    PreTrainedTokenizerBase,
)
from trl.trainer.utils import pad

from tqdm import tqdm
from liger_kernel.transformers import LigerFusedLinearJSD

from nanochat.adamw import DistAdamW
from fp_quant import finalize_master_weights


MODEL = "meta-llama/Llama-3.1-8B-Instruct"
DATASET = "allenai/tulu-3-sft-mixture"
MICRO_BATCH_SIZE = 16
TOTAL_BATCH_SIZE = 128
SEQ_LEN = 1024
LR = 4e-6

if rank == 0:
    import wandb

    wandb_run = wandb.init(
        project="fpquant-distill",
        config={
            "model": MODEL,
            "dataset": DATASET,
            "micro_bs": MICRO_BATCH_SIZE,
            "total_bs": TOTAL_BATCH_SIZE,
            "lr": LR,
            "world_size": world_size,
        },
    )

assert TOTAL_BATCH_SIZE % (MICRO_BATCH_SIZE * world_size) == 0


@dataclass
class PaddingDataCollatorForChatML:
    """
    Data collator for ChatML format datasets.
    """

    tokenizer: PreTrainedTokenizerBase
    ignore_index: int = -100
    max_length: int = None
    prompt_key: str = "prompt"
    messages_key: str = "messages"
    pad_to_multiple_of: int = 128

    def __post_init__(self):
        if self.tokenizer.pad_token_id is None:
            raise ValueError(
                "The tokenizer does not have a pad token. Please set `pad_token_id` in the tokenizer."
            )
        if self.max_length is None:
            # set a sensible default
            self.max_length = min(self.tokenizer.model_max_length, 1024)

    def __call__(self, examples: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        input_ids = []
        attention_mask = []
        prompts_input_ids = []
        prompt_attention_mask = []
        labels = []

        for example in examples:
            formatted_prompt = example.get(self.prompt_key, None)
            if formatted_prompt is None:
                prompt = example[self.messages_key][:-1]
                formatted_prompt = self.tokenizer.apply_chat_template(
                    prompt, tokenize=False, add_generation_prompt=True
                )

            if "input_ids" not in example:
                message = example[self.messages_key]
                formatted_message = self.tokenizer.apply_chat_template(
                    message, tokenize=False, add_generation_prompt=False
                )
                tokenized_message = self.tokenizer(
                    formatted_message,
                    truncation=True,
                    max_length=self.max_length,
                    padding=False,
                    return_tensors=None,
                    add_special_tokens=False,
                )
                input_ids.append(tokenized_message["input_ids"])
                if "attention_mask" in example:
                    attention_mask.append(tokenized_message["attention_mask"])
                else:
                    attention_mask.append([1] * len(tokenized_message["input_ids"]))
            else:
                input_ids.append(example["input_ids"])
                if "attention_mask" in example:
                    attention_mask.append(example["attention_mask"])
                else:
                    attention_mask.append([1] * len(example["input_ids"]))

            tokenized_prompt = self.tokenizer(
                formatted_prompt,
                truncation=True,
                max_length=len(input_ids[-1]),
                padding=False,
                return_tensors=None,
                add_special_tokens=False,
            )

            prompts_input_ids.append(tokenized_prompt["input_ids"])
            prompt_attention_mask.append(tokenized_prompt["attention_mask"])

            # Create the labels that will have all but the completion tokens of the example["input_ids"] set to ignore_index
            label = [self.ignore_index] * len(input_ids[-1])
            completion_start_idx = len(tokenized_prompt["input_ids"])
            label[completion_start_idx:] = input_ids[-1][completion_start_idx:]
            labels.append(label)

        # convert to list of tensors and pad
        input_ids = [torch.tensor(ids, dtype=torch.long) for ids in input_ids]
        attention_mask = [
            torch.tensor(mask, dtype=torch.long) for mask in attention_mask
        ]
        labels = [torch.tensor(label, dtype=torch.long) for label in labels]
        input_ids = pad(
            input_ids,
            padding_side="left",
            padding_value=self.tokenizer.pad_token_id,
            pad_to_multiple_of=self.pad_to_multiple_of,
        )
        attention_mask = pad(
            attention_mask,
            padding_side="left",
            padding_value=0,
            pad_to_multiple_of=self.pad_to_multiple_of,
        )
        labels = pad(
            labels,
            padding_side="left",
            padding_value=self.ignore_index,
            pad_to_multiple_of=self.pad_to_multiple_of,
        )

        prompts_input_ids = [
            torch.tensor(ids, dtype=torch.long) for ids in prompts_input_ids
        ]
        prompt_attention_mask = [
            torch.tensor(mask, dtype=torch.long) for mask in prompt_attention_mask
        ]
        prompts_input_ids = pad(
            prompts_input_ids,
            padding_side="left",
            padding_value=self.tokenizer.pad_token_id,
            pad_to_multiple_of=self.pad_to_multiple_of,
        )
        prompt_attention_mask = pad(
            prompt_attention_mask,
            padding_side="left",
            padding_value=0,
            pad_to_multiple_of=self.pad_to_multiple_of,
        )

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "prompts": prompts_input_ids,
            "prompt_attention_mask": prompt_attention_mask,
        }


# Student with FP-Quant pseudoquantization (behavior parity)
student = AutoModelForCausalLM.from_pretrained(
    MODEL,
    dtype=torch.bfloat16,
    device_map=device,
    # Real quantization
    quantization_config=FPQuantConfig(
        store_master_weights=True,
        forward_dtype="mxfp4",
        forward_method="abs_max",
        backward_dtype="mxfp8",
    ),
    # Pseudoquantization
    # quantization_config=FPQuantConfig(
    #     store_master_weights=True,
    #     pseudoquantization=True,
    #     forward_dtype="mxfp4",
    #     forward_method="abs_max",
    # ),
)
student.config.use_cache = False
student.gradient_checkpointing_enable(
    gradient_checkpointing_kwargs={"use_reentrant": False}
)
student_head = student.lm_head.weight
student_model = student.model


# Teacher (frozen)
teacher = AutoModelForCausalLM.from_pretrained(
    MODEL,
    dtype=torch.bfloat16,
    device_map=device,
)
for p in teacher.parameters():
    p.requires_grad_(False)
teacher.eval()
# also keep cache off to avoid any surprises
teacher.config.use_cache = False
teacher_head = teacher.lm_head.weight
teacher_model = teacher.model


# Dataset
ds = load_dataset(DATASET, split="train").shuffle(seed=42)
ten_percent = ds.select(range(int(0.1 * len(ds))))
splits = ten_percent.train_test_split(test_size=0.01, seed=42)
train_dataset = splits["train"]
eval_dataset = splits["test"]


def has_assistant(ex):
    msgs = ex.get("messages") or []
    return any(m.get("role") == "assistant" and m.get("content") for m in msgs)


train_dataset = train_dataset.filter(has_assistant)
eval_dataset = eval_dataset.filter(has_assistant)


# Tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL, use_fast=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"
data_collator = PaddingDataCollatorForChatML(
    tokenizer=tokenizer, max_length=SEQ_LEN, pad_to_multiple_of=SEQ_LEN
)


# Opt and loss
optimizer = DistAdamW(
    chain(
        student.parameters(),
    ),
    lr=LR,
)
scheduler = torch.optim.lr_scheduler.SequentialLR(
    optimizer,
    schedulers=[
        torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=0.001,
            end_factor=1.0,
            total_iters=100,
        ),
        torch.optim.lr_scheduler.ConstantLR(
            optimizer,
            factor=1.0,
            total_iters=1000,
        ),
    ],
    milestones=[100],
)

loss_fn = LigerFusedLinearJSD(ignore_index=data_collator.ignore_index)
smooth_loss = 0.0


# Compile
teacher_model = torch.compile(teacher_model, dynamic=False)
student_model = torch.compile(student_model, dynamic=False)


# Train
for step, data in tqdm(
    enumerate(train_dataset.iter(batch_size=TOTAL_BATCH_SIZE)),
    total=len(train_dataset) // TOTAL_BATCH_SIZE,
):
    features = [dict(zip(data, t)) for t in zip(*data.values())]
    if len(features) < TOTAL_BATCH_SIZE:
        break  # skip last batch

    global_batch_loss = 0.0

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for batch_start in range(0, TOTAL_BATCH_SIZE, MICRO_BATCH_SIZE * world_size):
        local_batch_start = batch_start + rank * MICRO_BATCH_SIZE
        local_batch_end = local_batch_start + MICRO_BATCH_SIZE

        data = data_collator(features[local_batch_start:local_batch_end])

        with torch.no_grad():
            teacher_hidden = teacher_model(
                input_ids=data["input_ids"].to(teacher.device),
                use_cache=False,
            ).last_hidden_state

        student_hidden = student_model(
            input_ids=data["input_ids"].to(teacher.device),
            use_cache=False,
        ).last_hidden_state

        loss = loss_fn(
            student_hidden.flatten(end_dim=-2),
            student_head,
            teacher_hidden.flatten(end_dim=-2),
            teacher_head,
            data["labels"].flatten().to(student_hidden.device),
        ) / (TOTAL_BATCH_SIZE // (MICRO_BATCH_SIZE * world_size))

        global_batch_loss += loss.item()

        loss.backward()

    norm = torch.nn.utils.clip_grad_norm_(student_model.parameters(), 1.0).item()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    scheduler.step()
    t1 = time.perf_counter()
    dt = t1 - t0

    if rank == 0:
        smooth_loss = 0.9 * smooth_loss + 0.1 * global_batch_loss
        debiased_smooth_loss = smooth_loss / (1 - 0.9 ** (step + 1))
        wandb_run.log(
            {
                "step": step,
                "loss": global_batch_loss,
                "smooth_loss": debiased_smooth_loss,
                "grad_norm": norm,
                "tok/sec": TOTAL_BATCH_SIZE * SEQ_LEN / dt,
            }
        )


if rank == 0:
    finalize_master_weights(student)

    student.save_pretrained(f"/tmp/models/{MODEL.split('/')[-1]}-FPQuant-QAT-MXFP4")

destroy_process_group()
