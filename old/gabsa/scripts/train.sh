#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=2 python /srv/nas_data1/text/randy/absa/facebook-absa/gabsa/main.py --do_train \
            --do_eval \
            --do_predict \
            --train_args /srv/nas_data1/text/randy/absa/facebook-absa/gabsa/train_args/train_args.json \
            --model_type t5 \
            --model_name_or_path t5-base \
            --max_len 256 \
            --task "aste ate ote uabsa aope" \
            --paradigm extraction \
            --prompt_path /srv/nas_data1/text/randy/absa/facebook-absa/gabsa/prompts/prompt_en.json \
            --prompt_option_path /srv/nas_data1/text/randy/absa/facebook-absa/gabsa/prompts/option.json \
            --pattern /srv/nas_data1/text/randy/absa/facebook-absa/gabsa/patterns/default.json \
            --data_dir /srv/nas_data1/text/randy/absa/facebook-absa/data/mono/aste_data_v2_16res \
            --trains "train_triplets" \
            --devs "dev_triplets" \
            --tests "test_triplets" \
            --blank_frac 1.0 \
            --random_state 17 \
            --output_dir /srv/nas_data1/text/randy/absa/models/facebook_research/16res/pabsa \
            --per_device_predict_batch_size 32