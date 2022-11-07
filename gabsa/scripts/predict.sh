#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=2 python /srv/nas_data1/text/randy/absa/facebook-absa/gabsa/main.py --do_predict \
            --train_args /srv/nas_data1/text/randy/absa/facebook-absa/gabsa/train_args/train_args.json \
            --model_type t5 \
            --model_name_or_path /srv/nas_data1/text/randy/absa/models/semeval/14lap/t5/t5_pabsa_S256_en_small_blank=1.0 \
            --max_len 256 \
            --task "aope" \
            --paradigm extraction \
            --prompt_path /srv/nas_data1/text/randy/absa/facebook-absa/gabsa/prompts/prompt_en.json \
            --prompt_option_path /srv/nas_data1/text/randy/absa/facebook-absa/gabsa/prompts/option.json \
            --pattern /srv/nas_data1/text/randy/absa/facebook-absa/gabsa/patterns/default.json \
            --data_dir /srv/nas_data1/text/randy/absa/facebook-absa/data/mono/aste_data_v2_14lap \
            --tests "test_triplets" \
            --blank_frac 1.0 \
            --random_state 42 \
            --output_dir /srv/nas_data1/text/randy/absa/models/semeval/14lap/t5/t5_pabsa_S256_en_small_blank=1.0/aope \
            --per_device_predict_batch_size 32