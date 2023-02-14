#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=2 python /srv/nas_data1/text/randy/absa/facebook-absa/gabsa/main.py --do_predict \
            --train_args /srv/nas_data1/text/randy/absa/facebook-absa/gabsa/train_args/train_args.json \
            --model_type t5 \
            --model_name_or_path /srv/nas_data1/text/randy/absa/models/facebook_research/id/generative/fix/t5/t5_pabsa_S256_wid_small_blank=1.0 \
            --max_len 256 \
            --task "aste ate ote uabsa aope" \
            --paradigm extraction \
            --prompt_path /srv/nas_data1/text/randy/absa/facebook-absa/gabsa/prompts/prompt.json \
            --prompt_option_path /srv/nas_data1/text/randy/absa/facebook-absa/gabsa/prompts/option.json \
            --pattern /srv/nas_data1/text/randy/absa/facebook-absa/gabsa/patterns/default.json \
            --data_dir /srv/nas_data1/text/randy/absa/facebook-absa/data/combination/prosa/gaste_format \
            --tests "test_news test_socmed" \
            --blank_frac 1.0 \
            --random_state 500 \
            --output_dir /srv/nas_data1/text/randy/absa/models/facebook_research/id/generative/fix/t5/t5_pabsa_S256_wid_small_blank=1.0 \
            --per_device_predict_batch_size 32