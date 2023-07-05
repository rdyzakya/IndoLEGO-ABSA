# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.6
#   kernelspec:
#     display_name: base
#     language: python
#     name: python3
# ---

# %%
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

# %%
import random
import numpy as np
import torch

def set_seed(seed: int):
    """Set the seed for reproducibility across multiple libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(42)

# %%
n_gpu = torch.cuda.device_count()

# %% [markdown]
# # Data

# %%
import sys
sys.path.append("../../../src")
import data_utils
train_path = "../../data/absa/en/zhang/interim/interim_2/rest1516/train.txt"
val_path = "../../data/absa/en/zhang/interim/interim_2/rest1516/dev.txt"
test_path = "../../data/absa/en/zhang/interim/interim_2/rest1516/test.txt"

train = data_utils.read_data(train_path)
val = data_utils.read_data(val_path)
test = data_utils.read_data(test_path)

# %%
train_tasks = [
    {
        "paradigm" : "extraction",
        "se_order" : "a",
        "prompt" : "gas",
        "answer" : "gas"
    },
    {
        "paradigm" : "extraction",
        "se_order" : "o",
        "prompt" : "gas",
        "answer" : "gas"
    },
    {
        "paradigm" : "extraction",
        "se_order" : "c",
        "prompt" : "gas",
        "answer" : "gas"
    },

    {
        "paradigm" : "extraction",
        "se_order" : "oa",
        "prompt" : "gas",
        "answer" : "gas"
    },
    {
        "paradigm" : "extraction",
        "se_order" : "as",
        "prompt" : "gas",
        "answer" : "gas"
    },
    {
        "paradigm" : "extraction",
        "se_order" : "sc",
        "prompt" : "gas",
        "answer" : "gas"
    },

    {
        "paradigm" : "extraction",
        "se_order" : "oasc",
        "prompt" : "gas",
        "answer" : "gas"
    },
]

val_tasks = [
    {
        "paradigm" : "extraction",
        "se_order" : "oasc",
        "prompt" : "gas",
        "answer" : "gas"
    }
]

test_tasks = [
    {
        "paradigm" : "extraction",
        "se_order" : "oasc",
        "prompt" : "gas",
        "answer" : "gas"
    }
]

# %%
train_ds = data_utils.data_gen(data=train, nt_se_order="acso", tasks=train_tasks, n_fold=3, algo="random", shuffle=True)
val_ds = data_utils.data_gen(data=val, nt_se_order="acso", tasks=val_tasks, n_fold=1, algo="round_robin", shuffle=False)
test_ds = data_utils.data_gen(data=test, nt_se_order="acso", tasks=test_tasks, n_fold=1, algo="round_robin", shuffle=False)

# %%
train_ds[0]

# %%
from datasets import Dataset

train_ds = Dataset.from_list(train_ds)
val_ds = Dataset.from_list(val_ds)
test_ds = Dataset.from_list(test_ds)

train_ds.to_csv("train.csv")
val_ds.to_csv("val.csv")

# %%
train_ds

# %%
val_ds

# %%
test_ds

# %% [markdown]
# # Tokenize

# %%
from transformers import AutoTokenizer

encoding_args = {
    "max_length" : 256,
    "padding" : True,
    "truncation" : True,
    "return_tensors" : "pt"
}

encode_fn = lambda x: tokenizer(x["input"], text_target=x["output"], **encoding_args)

tokenizer = AutoTokenizer.from_pretrained("google/mt5-base")

# %%
train_tok = train_ds.map(encode_fn, batched=True, remove_columns=train_ds.column_names)
train_tok.set_format("torch")

val_tok = val_ds.map(encode_fn, batched=True, remove_columns=val_ds.column_names)
val_tok.set_format("torch")

test_tok = test_ds.map(encode_fn, batched=True, remove_columns=test_ds.column_names)
test_tok.set_format("torch")

# %%
from transformers import DataCollatorForSeq2Seq

data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer)

# %% [markdown]
# # Train

# %%
from transformers import Seq2SeqTrainingArguments

train_args = {
    "num_train_epochs": 10,
    "learning_rate": 3e-4,
    "save_total_limit": 2,
    "gradient_accumulation_steps": 1,
    "per_device_train_batch_size": 16//n_gpu,
    "per_device_eval_batch_size": 8,
    "save_strategy": "epoch",
    "evaluation_strategy": "epoch",
    "logging_strategy" : "epoch",
    "metric_for_best_model": "overall_f1_score",
    "load_best_model_at_end": True,
    "adam_epsilon": 1e-08,
    "output_dir": "./output",
    "logging_dir" : "./output/log",
    "include_inputs_for_metrics" : True
}

train_args = Seq2SeqTrainingArguments(**train_args)

# %%
import torch
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
print(device)

# %%
from transformers import Seq2SeqTrainer, AutoModelForSeq2SeqLM

set_seed(42)

model = AutoModelForSeq2SeqLM.from_pretrained("google/mt5-base")
model.to(device)


# %%
def preprocess_logits_for_metrics(logits, targets):
    pred_logits = logits[0] if isinstance(logits,tuple) else logits
    pred_ids = torch.argmax(pred_logits, dim=-1)
    return pred_ids, targets


# %%
from evaluation import compute_metrics

catch_answer_fn = getattr(data_utils.AnswerCatcher(),"gas")
decoding_args = {
    "skip_special_tokens" : False
}

trainer = Seq2SeqTrainer(
        model = model,
        args = train_args,
        tokenizer = tokenizer,
        data_collator = data_collator,
        train_dataset = train_tok,
        eval_dataset = val_tok,
        compute_metrics = lambda eval_preds: compute_metrics(catch_answer_fn, eval_preds, decoding_args, tokenizer, val_ds["se_order"]),
        preprocess_logits_for_metrics=preprocess_logits_for_metrics
    )

trainer.train()

# %%
