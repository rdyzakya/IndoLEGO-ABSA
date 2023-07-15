from argparse import ArgumentParser
import os
from datasets import Dataset
import utils
import preprocess
import postprocess
import pandas as pd
import json
import random
from copy import deepcopy
from transformers import (Trainer, TrainingArguments, 
                          Seq2SeqTrainer ,Seq2SeqTrainingArguments,
                          DataCollatorForLanguageModeling, DataCollatorForSeq2Seq,
                          AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM)

data_reader = preprocess.DataReader()
data_augmentator = preprocess.DataAugmentator()
cleaner = postprocess.Cleaner()
answer_catcher = postprocess.AnswerCatcher()

def init_args():
    parser = ArgumentParser()
    parser.add_argument("--seed", type=int, help="Training seed", default=42)
    parser.add_argument("--n_gpu", type=str, help="Gpu device(s) used", default='0')

    parser.add_argument("--td_config", type=str, 
                        help="Path to train data configuration json file.", 
                        default="../configs/td_config.json")
    parser.add_argument("--vd_config", type=str,
                        help="Path to validation data configuration json file.", 
                        required=False)
    parser.add_argument("--na_config", type=str,
                        help="Path to non absa data configuration json file.",
                        required=False)
    parser.add_argument("--train_args", type=str,
                        help="Path to train configuration json file.",
                        default="../configs/train_args.json")
    
    parser.add_argument("--max_len", type=int,
                        help="Maximum sequence length.",
                        default=128
                        )
    
    parser.add_argument("--model_name_or_path", type=str,
                        help="Model name or path.",
                        default="google/mt5-base")

    parser.add_argument("--prompt", type=str,
                        help="Prompt type [lego_absa, bartabsa, gas, prefix, one_token, no_prompt].",
                        default="lego_absa")
    
    parser.add_argument("--answer", type=str,
                        help="Answer type [lego_absa, bartabsa, gas].",
                        default="lego_absa")
    
    parser.add_argument("--remove", type=str, nargs='+',
                        help="Token/phrase/word needed to be remove, for example id_ID or en_XX in mbart output.",
                        default=[])
    
    parser.add_argument("--shuffle_train", action="store_true", help="Shuffle overall dataset")
    
    args = parser.parse_args()

    return args

def set_env(args):
    # Environment
    utils.set_seed(args.seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.n_gpu

def get_data(absa_config, non_absa_config, prompt, answer, shuffle):
    ds = []
    for config in absa_config:
        path = config.pop("path")
        data = data_reader.do(path)

        augmentation_args = deepcopy(config)
        for i in range(len(augmentation_args["tasks"])):
            augmentation_args["tasks"][i] = {
                "se_order" : augmentation_args["tasks"][i],
                "prompt" : prompt,
                "answer" : answer
            }
        augmentation_args.update({
            "data" : data
        })

        augmented_data = data_augmentator.do(**augmentation_args)
        ds.extend(augmented_data)
    
    if non_absa_config != None:
        for path in non_absa_config:
            df = pd.read_csv(path)
            df["se_order"] = "non_absa"
            data = df.to_dict(orient="records")
            ds.extend(data)
    
    if shuffle:
        random.shuffle(ds)

    ds = Dataset.from_list(ds)

    return ds

def main():
    args = init_args()

    # Set environment
    set_env(args)

    # Training data
    ## ABSA Dataset
    with open(args.td_config, 'r') as fp:
        td_config = json.load(fp)
    ## Non ABSA Dataset
    na_config = None
    if args.na_config != None:
        with open(args.na_config, 'r') as fp:
            na_config = json.load(fp)
    train = get_data(td_config, na_config, 
                     prompt=args.prompt, answer=args.answer, 
                     shuffle=args.shuffle_train)
    train.to_csv("train.csv")
    
    # Validation data
    do_eval = bool(args.vd_config)
    val = None
    if do_eval:
        with open(args.vd_config, 'r') as fp:
            vd_config = json.load(fp)
        val = get_data(vd_config, None, 
                       prompt=args.prompt, answer=args.answer, 
                       shuffle=False)
        val.to_csv("val.csv")

    # Prepare tokenizer and answer utilities
    with open(args.train_args, 'r') as fp:
        train_args = json.load(fp)
    output_dir = train_args["output_dir"]

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    encode_seq2seq = lambda x: tokenizer(x["input"], text_target=x["output"], 
                                    max_length=args.max_len, padding=True, truncation=True,
                                    return_tensors="pt")
    encode_clm = lambda x: tokenizer([el["input"] + ' ' + tokenizer.sep_token + ' ' + el["output"] + ' ' + tokenizer.eos_token
                                      for el in x], 
                                    max_length=args.max_len, padding=True, truncation=True,
                                    return_tensors="pt")
    
    catch_answer_fn = getattr(answer_catcher, args.answer)

    def catch_answer_seq2seq(out, se_order, text):
        out = cleaner.one(out, remove=[tokenizer.eos_token, tokenizer.pad_token] + args.remove)
        return catch_answer_fn(out, se_order, text)
    
    def catch_answer_clm(out, se_order, text):
        out = out.split(tokenizer.sep_token)[-1]
        out = cleaner.one(out, remove=[tokenizer.eos_token, tokenizer.pad_token] + args.remove)
        return catch_answer_fn(out, se_order, text)
    
    decoding_args = {
        "skip_special_tokens" : False
    }

    # Prepare model and training utilities
    try:
        model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name_or_path)
        train_tok = train.map(encode_seq2seq, batched=True, remove_columns=train.column_names)
        val_tok = None
        data_collator = DataCollatorForSeq2Seq(tokenizer)
        train_args = Seq2SeqTrainingArguments(**train_args)
        trainer_args = dict(model=model,
                            args=train_args,
                            tokenizer=tokenizer,
                            data_collator=data_collator,
                            train_dataset=train_tok,)
        if do_eval:
            val_tok = val.map(encode_seq2seq, batched=True, remove_columns=train.column_names)
            trainer_args.update(dict(eval_dataset=val_tok,
                                 compute_metrics=lambda x: utils.compute_metrics(catch_answer_seq2seq, x, decoding_args, tokenizer, val["se_order"]),
                                 preprocess_logits_for_metrics=utils.preprocess_logits_for_metrics))
        trainer = Seq2SeqTrainer(**trainer_args)
    except ValueError as ve:
        try:
            model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path)
            utils.add_token_clm(model, tokenizer)
            train_tok = train.map(encode_clm, batched=True, remove_columns=train.column_names)
            val_tok = None
            data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
            train_args = TrainingArguments(**train_args)
            trainer_args = dict(model=model,
                            args=train_args,
                            tokenizer=tokenizer,
                            data_collator=data_collator,
                            train_dataset=train_tok,)
            if do_eval:
                val_tok = val.map(encode_clm, batched=True, remove_columns=train.column_names)
                trainer_args.update(dict(eval_dataset=val_tok,
                                 compute_metrics=lambda x: utils.compute_metrics(catch_answer_clm, x, decoding_args, tokenizer, val["se_order"]),
                                 preprocess_logits_for_metrics=utils.preprocess_logits_for_metrics))
            trainer = Trainer(**trainer_args)
        except ValueError as ve:
            raise NotImplementedError("Only Seq2Seq and CausalLM Model")
        
    # Training
    trainer.train()

    # Save
    if trainer.is_world_process_zero():
        tokenizer.save_pretrained(save_directory=output_dir)
        model.save_pretrained(save_directory=output_dir)

if __name__ == "__main__":
    main()