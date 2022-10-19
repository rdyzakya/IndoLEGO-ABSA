#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=0 python main.py --do_train \
            --do_eval \
            --do_predict \
            --train_args /srv/nas_data1/text/randy/absa/facebook-absa/gabsa/train_args/train_args.json \
            --model_type t5 \
            --model_name_or_path t5-base \
            --max_len 256 \
            --task "aste aope ate ote uabsa" \
            --paradigm extraction \
            --prompt_option 0 \
            --prompt_path /srv/nas_data1/text/randy/absa/facebook-absa/gabsa/prompts/prompt_en.json \
            --prompt_option_path /srv/nas_data1/text/randy/absa/facebook-absa/gabsa/prompts/option.json \
            --pattern /srv/nas_data1/text/randy/absa/facebook-absa/gabsa/patterns/default.json \
            --data_dir /srv/nas_data1/text/randy/absa/facebook-absa/data/interim/gaste_format/aste_data_v2_14lap \
            --trains "train_triplets" \
            --devs "dev_triplets" \
            --tests "test_triplets" \
            --blank_frac 1.0 \
            --random_state 42 \
            --output_dir /srv/nas_data1/text/randy/absa/models/generative \
            --per_device_predict_batch_size 16
            # --trains "train_socmed_twenty_percent train_news_annotator train_news_student" \