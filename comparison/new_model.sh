#!/usr/bin/env bash

base="/srv/nas_data1/text/randy/aste/models"

python /srv/nas_data1/text/randy/aste/facebook-aste/comparison/new_model.py --aste_model_type t5 \
            --aste_model_name_or_path ${base}/generative/t5/t5_aste_128_blank=1.0 \
            --data /srv/nas_data1/text/randy/aste/facebook-aste/data/interim/gaste_format/test_news.txt \
            --n_gpu 1 \
            --fixing raw \
            --batch_size 4 \
            --output_dir /srv/nas_data1/text/randy/aste/facebook-aste/comparison/results/new_model/test_news/v_0 