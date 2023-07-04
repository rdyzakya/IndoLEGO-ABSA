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
import data_utils
train_path = "../data/absa/id/william/train.txt"
val_path = "../data/absa/id/william/dev.txt"
test_path = "../data/absa/id/william/test.txt"

train = data_utils.read_data(train_path)
val = data_utils.read_data(val_path)
test = data_utils.read_data(test_path)

# %%
train_tasks = [
    {
        "paradigm" : "extraction",
        "se_order" : "oa",
        "method" : "lego_absa"
    },
    {
        "paradigm" : "extraction",
        "se_order" : "as",
        "method" : "lego_absa"
    },
    {
        "paradigm" : "imputation",
        "reduced_se_order" : "oa",
        "se_order" : "oas",
        "method" : "lego_absa"
    },
    {
        "paradigm" : "imputation",
        "reduced_se_order" : "as",
        "se_order" : "oas",
        "method" : "lego_absa"
    },
]

val_tasks = [
    {
        "paradigm" : "extraction",
        "se_order" : "oas",
        "method" : "lego_absa"
    }
]

test_tasks = [
    {
        "paradigm" : "extraction",
        "se_order" : "oas",
        "method" : "lego_absa"
    }
]

# %%
train_ds = data_utils.data_gen(data=train, nt_se_order="aos", tasks=train_tasks, n_fold=4, algo="random", shuffle=True)
val_ds = data_utils.data_gen(data=val, nt_se_order="aos", tasks=val_tasks, n_fold=1, algo="round_robin", shuffle=False)
test_ds = data_utils.data_gen(data=test, nt_se_order="aos", tasks=test_tasks, n_fold=1, algo="round_robin", shuffle=False)

# %%
for el in train_ds:
    if el["input"].startswith("pngen kembali lagi buat menginap"):
        print(el)

# %%
train_ds[0]

# %%
from datasets import Dataset

train_ds = Dataset.from_list(train_ds)
val_ds = Dataset.from_list(val_ds)
test_ds = Dataset.from_list(test_ds)

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
    "num_train_epochs": 20,
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

model = AutoModelForSeq2SeqLM.from_pretrained("google/mt5-base")
model.to(device)


# %%
def preprocess_logits_for_metrics(logits, targets):
    pred_logits = logits[0] if isinstance(logits,tuple) else logits
    pred_ids = torch.argmax(pred_logits, dim=-1)
    return pred_ids, targets


# %%
from evaluation import compute_metrics

catch_answer_fn = data_utils.AnswerCatcher().lego_absa
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