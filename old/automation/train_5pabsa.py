import subprocess
import shutil
import os
from pynvml import nvmlInit,nvmlDeviceGetHandleByIndex,nvmlDeviceGetMemoryInfo


def gpu_utilization(index):
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(index)
    info = nvmlDeviceGetMemoryInfo(handle)
    return info.used//1024**2

def command(data_id,random_seed):
    main_cmd = "python /srv/nas_data1/text/randy/absa/facebook-absa/gabsa/main.py"
    args = f"""--do_train \
            --do_eval \
            --do_predict \
            --train_args /srv/nas_data1/text/randy/absa/facebook-absa/gabsa/train_args/train_args.json \
            --model_type t5 \
            --model_name_or_path Wikidepia/IndoT5-base \
            --max_len 256 \
            --task "aste" \
            --paradigm extraction \
            --prompt_path /srv/nas_data1/text/randy/absa/facebook-absa/gabsa/prompts/prompt.json \
            --prompt_option_path /srv/nas_data1/text/randy/absa/facebook-absa/gabsa/prompts/option_no_prompt.json \
            --pattern /srv/nas_data1/text/randy/absa/facebook-absa/gabsa/patterns/default.json \
            --data_dir /srv/nas_data1/text/randy/absa/facebook-absa/data/mono/{data_id} \
            --trains "train" \
            --devs "dev" \
            --tests "test" \
            --blank_frac 1.0 \
            --random_state {random_seed} \
            --output_dir /srv/nas_data1/text/randy/absa/models/facebook_research/{data_id}/non_mtl \
            --per_device_predict_batch_size 32"""
    # bashCommand = main_cmd.split() + args.split()
    # args = args.replace('"','""')
    args = args.split("--")
    bashCommand = []
    for c in main_cmd.split():
        bashCommand.append(c)
    for a in args:
        a = a.strip()
        if len(a) == 0:
            continue
        splitted_args = a.split()
        key = splitted_args[0]
        bashCommand.append("--" + key)

        if len(splitted_args) > 1:
            value = ' '.join(splitted_args[1:])
            value = value.replace('"','')
            bashCommand.append(value)
    # line = subprocess.list2cmdline(bashCommand)
    # bashCommand = line.split()
    return bashCommand

def main():
    os.environ["CUDA_VISIBLE_DEVICES"]="2"
    env = os.environ
    datasets = ["hotel_id_v2"]
    # datasets = ["16res"]
    random_seeds = [0,17,42,123,500]
    for ds in datasets:
        for seed in random_seeds:
            while gpu_utilization(2) > 1024:
                print(f"Waiting for GPU index 2 is free... Current utilization : {gpu_utilization(2)}",end="\r") # do nothing
            bashCommand = command(ds,seed)
            print(f"Training model for random seed {seed} and dataset {ds}")
            # process = subprocess.Popen(bashCommand, stdout=subprocess.PIPE, env=env)
            # output, error = process.communicate()
            error = subprocess.call(bashCommand, env=env)
            if error != 0:
                raise Exception(str(error))
            else:
                src_path = f"{bashCommand[-3]}/t5/t5_aste_S256_wid_base_blank=1.0"
                destination_path = f"/srv/nas_data1/text/randy/absa/facebook-absa/gabsa/score_5_runs_non_mtl/r{seed}/{ds}"
                prediction_metrics_src_path = f"{src_path}/prediction_metrics.json"
                prediction_metrics_destination_path = f"{destination_path}/prediction_metrics.json"
                prediction_csv_src_path = f"{src_path}/prediction.csv"
                prediction_csv_destination_path = f"{destination_path}/prediction.csv"

                if not os.path.exists(destination_path):
                    os.makedirs(destination_path)

                shutil.move(prediction_metrics_src_path,prediction_metrics_destination_path)
                shutil.move(prediction_csv_src_path,prediction_csv_destination_path)

                shutil.rmtree(src_path)

if __name__ == "__main__":
    main()