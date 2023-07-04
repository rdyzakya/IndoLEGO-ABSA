import json
import subprocess
import sys
sys.path.append("../..")
import gpu
import os

task_path = "../combination_task.json"

config_dir = "../../../configs"

data_config_path = f"{config_dir}/data_config.json"
model_config_path = f"{config_dir}/model_config.json"
train_args_path = f"{config_dir}/train_args.json"
pattern_config_path = f"{config_dir}/pattern_config.json"
prompt_config_path = f"{config_dir}/prompt_config.json"
encoding_args_path = f"{config_dir}/encoding_args.json"
decoding_args_path = f"{config_dir}/decoding_args.json"

output_dir = "./output"

data_path = "../../../data/absa/en/zhang/interim/interim_2"

script_path = "../../../src/main.py"

model_name_or_path = "t5-base"

tasks = json.load(open(task_path,'r'))

dataset = ["rest15","rest16"]

n_gpu = 6

temp_data_config = {
    "train" : {
        "absa" : {
            "data_path" : "",
            "target_format" : "acso"
        },
        "non_absa" : [],
        "absa_builder_args" : {
            "tasks" : {
                "extraction" : ["aos","cs"],
                "imputation" : {}
            },
            "multiply" : True,
            "shuffle" : True,
            "random_state" : 0,
            "pattern_index" : None
        }
    },
    "val" : {
        "absa" : {
            "data_path" : "",
            "target_format" : "acso"
        },
        "non_absa" : [],
        "absa_builder_args" : {
            "tasks" : {
                "extraction" : ["aos","cs"],
                "imputation" : {}
            },
            "multiply" : True,
            "shuffle" : True,
            "random_state" : 0,
            "pattern_index" : {
                "acos" : 0,
                "aos" : 0,
                "acs" : 0,
                "ao" : 0,
                "as" : 0,
                "cs" : 0,
                "c" : 0,
                "a" : 0
            }
        }
    },
    "test" : {
        "absa" : {
            "data_path" : "",
            "target_format" : "acso"
        },
        "non_absa" : [],
        "absa_builder_args" : {
            "task_tree" : {"aos" : [], "cs" : []},
            "i_pattern" : 0
        }
    }
}

temp_model_config = {
    "model_name_or_path" : model_name_or_path,
    "model_args" : {},
    "tokenizer_args" : {}
}

temp_train_args = {
    "num_train_epochs" : 20,
    "learning_rate" : 3e-4,
    "save_total_limit" : 2,
    "gradient_accumulation_steps" : 1,
    "per_device_train_batch_size" : 16,
    "per_device_eval_batch_size" : 8,
    "save_strategy" : "epoch",
    "evaluation_strategy" : "epoch",
    "metric_for_best_model" : "overall_f1_score",
    "load_best_model_at_end" : True,
    "adam_epsilon" : 1e-8,
    "output_dir" : ""
}

def result_dict(path):
    if not os.path.exists(path):
        result = {}
    else:
        with open(summary_score_path,'r') as fp:
            result = json.load(fp)
    return result

summary_score_path = "summary_score.json"
log_history_path = "log_history.json"
stderr_path = "stderr.json"

summary_score = result_dict(summary_score_path)
log_history = result_dict(log_history_path)
stderr = result_dict(stderr_path)

if not os.path.exists("./preds"):
    os.makedirs("./preds")

print("Train...")
for extraction_task in tasks["extraction"]["combination"]:
    for ds in dataset:
        task_code = '-'.join(extraction_task)
        print(f"Train combination {task_code} with {ds} dataset...")
        # Change dataset
        data_config = temp_data_config.copy()
        data_config["train"]["absa"]["data_path"] = data_path + f"/{ds}/train.txt"
        data_config["val"]["absa"]["data_path"] = data_path + f"/{ds}/dev.txt"
        data_config["test"]["absa"]["data_path"] = data_path + f"/{ds}/test.txt"
        data_config["train"]["absa_builder_args"]["tasks"]["extraction"] = extraction_task
        with open(data_config_path,'w') as fp:
            json.dump(data_config,fp)
        # # Change task
        # model_config = temp_model_config.copy()
        # model_config["model_name_or_path"] = model_name_or_path
        # model_config["tokenizer_args"]["padding_side"] = "left" if model_type == "causal_lm" else "right"
        # with open(model_config_path,'w') as fp:
        #     json.dump(model_config,fp)
        # Change output directory
        train_args = temp_train_args.copy()
        train_args["output_dir"] = output_dir + f"/{task_code}/{ds}"
        with open(train_args_path,'w') as fp:
            json.dump(train_args,fp)
        
        gpu.waiting_gpu_loop(n_gpu,threshold=5000)
        
        process = subprocess.run([
            "python",
            script_path,
            "--do_train", "--do_eval", "--do_predict",
            "--train_seed", "0",
            "--n_gpu", str(n_gpu),
            "--model_config", model_config_path,
            "--data_config", data_config_path,
            "--pattern_config", pattern_config_path,
            "--prompt_config", prompt_config_path,
            "--train_args", train_args_path,
            "--encoding_args", encoding_args_path,
            "--decoding_args", decoding_args_path,
            "--patience", "3"
        ])

        if process.returncode == 0:
            # Success
            score_path = train_args["output_dir"] + "/absa_score.json"
            ckpt_folders = [fname for fname in os.listdir(train_args["output_dir"]) if fname.startswith("checkpoint")]
            latest_ckpt_dir = train_args["output_dir"] + '/' + max(ckpt_folders,key=lambda fname: int(fname.split('-')[1]))
            trainer_state_path = latest_ckpt_dir + "/trainer_state.json"

            with open(score_path,'r') as fp:
                score = json.load(fp)
            if task_code not in summary_score:
                summary_score[task_code] = {}
            summary_score[task_code][f"{ds}-aos"] = score["aos"]["f1_score"]
            summary_score[task_code][f"{ds}-cs"] = score["cs"]["f1_score"]
            with open(summary_score_path,'w') as fp:
                json.dump(summary_score,fp)
            
            with open(trainer_state_path,'r') as fp:
                trainer_state = json.load(fp)
            if task_code not in log_history:
                log_history[task_code] = {}
            log_history[task_code][ds] = trainer_state["log_history"]
            with open(log_history_path, 'w') as fp:
                json.dump(log_history,fp)
            
            # Copy the absa_preds and str preds
            subprocess.call([
                "cp", train_args["output_dir"] + "/absa_pred.csv", f"./preds/{task_code}_{ds}_pred.csv"
            ])

            subprocess.call([
                "cp", train_args["output_dir"] + "/absa_str_pred.csv", f"./preds/{task_code}_{ds}_str_pred.csv"
            ])
            
            # Delete model folder to save space
            subprocess.call([
                "rm", "-rf", train_args["output_dir"]
            ])
        else:
            # Error
            print(f"ERROR for {task_code} during training {ds}")
            # stderr[f"{model_name_or_path}-{ds}"] = process.stderr.read()
            # with open(stderr_path,'w') as fp:
            #     json.dump(stderr,fp)
print("DONE!")