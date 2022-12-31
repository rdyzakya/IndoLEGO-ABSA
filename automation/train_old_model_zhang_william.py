import subprocess
import json
import os
from pynvml import nvmlInit,nvmlDeviceGetHandleByIndex,nvmlDeviceGetMemoryInfo

os.chdir("/srv/nas_data1/text/randy/absa/transformers")

def gpu_utilization(index):
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(index)
    info = nvmlDeviceGetMemoryInfo(handle)
    return info.used//1024**2

def command(type_):
    cmd = f"bash /srv/nas_data1/text/randy/absa/transformers/app/transformers/models/scripts/run_finetune.sh \
    -c /srv/nas_data1/text/randy/absa/transformers/app/transformers/models/config/aste/{type_}.json \
    -n {type_}_model"

    cmd = cmd.split()
    return cmd

def main():
    config_path = "/srv/nas_data1/text/randy/absa/transformers/app/transformers/models/config/aste"
    aspect_config_path = config_path + "/aspect.json"
    opinion_config_path = config_path + "/opinion.json"
    relation_config_path = config_path + "/relation.json"

    aspect_config_file = open(aspect_config_path,'r')
    opinion_config_file = open(opinion_config_path,'r')
    relation_config_file = open(relation_config_path,'r')
    
    aspect_config = json.load(aspect_config_file)
    opinion_config = json.load(opinion_config_file)
    relation_config = json.load(relation_config_file)

    aspect_config_file.close()
    opinion_config_file.close()
    relation_config_file.close()

    data_dir = "/srv/nas_data1/text/randy/absa/facebook-absa/data/mono"
    data_listdir = os.listdir(data_dir)
    data_listdir.remove("news")
    data_listdir.remove("socmed_20")

    models = [{
        "m" : "xlm-roberta",
        "pd" : "mul"
    },
    {
        "m" : "spanbert",
        "pd" : "sg"
    }]

    os.environ["CUDA_VISIBLE_DEVICES"]="2"
    env = os.environ

    for data in data_listdir:
        for m in models:
            # ubah model
            aspect_config.update(m)
            opinion_config.update(m)
            relation_config.update(m)

            # ubah data
            data_path = data_dir + "/" + data + "/old_format"
            aspect_data_path = data_path + "/aspect"
            opinion_data_path = data_path + "/opinion"
            relation_data_path = data_path + "/relation"
            aspect_config["data_dir"] = aspect_data_path
            opinion_config["data_dir"] = opinion_data_path
            relation_config["data_dir"] = relation_data_path

            # ubah output dir
            output_dir = "/srv/nas_data1/text/randy/absa/models/facebook_research/" + data + "/two_staged"
            aspect_output_dir = output_dir + "/aspect"
            opinion_output_dir = output_dir + "/opinion"
            relation_output_dir = output_dir + "/relation"
            aspect_config["output_dir"] = aspect_output_dir
            opinion_config["output_dir"] = opinion_output_dir
            relation_config["output_dir"] = relation_output_dir

            aspect_config_file = open(aspect_config_path,'w')
            opinion_config_file = open(opinion_config_path,'w')
            relation_config_file = open(relation_config_path,'w')

            json.dump(aspect_config,aspect_config_file)
            json.dump(opinion_config,opinion_config_file)
            json.dump(relation_config,relation_config_file)

            aspect_config_file.close()
            opinion_config_file.close()
            relation_config_file.close()

            while gpu_utilization(2) > 4096:
                print(f"Waiting for GPU index 2 is free... Current utilization : {gpu_utilization(2)}",end="\r") # do nothing

            for t in ["aspect","opinion","relation"]:
                error = subprocess.call(command(t),env=env)
                print(error)
                if error != 0:
                    raise Exception(str(error))

if __name__ == "__main__":
    main()