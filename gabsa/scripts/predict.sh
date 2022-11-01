#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=2 python /srv/nas_data1/text/randy/absa/facebook-absa/gabsa/main.py --do_predict \
            --train_args /srv/nas_data1/text/randy/absa/facebook-absa/gabsa/train_args/train_args.json \
            --model_type t5 \
            --model_name_or_path /srv/nas_data1/text/randy/absa/models/facebook_research/generative/t5/t5_pabsa_S256_wid_base_blank=1.0 \
            --max_len 256 \
            --task "aste" \
            --paradigm extraction \
            --prompt_option 0 \
            --prompt_path /srv/nas_data1/text/randy/absa/facebook-absa/gabsa/prompts/prompt.json \
            --prompt_option_path /srv/nas_data1/text/randy/absa/facebook-absa/gabsa/prompts/option.json \
            --pattern /srv/nas_data1/text/randy/absa/facebook-absa/gabsa/patterns/default.json \
            --data_dir /srv/nas_data1/text/randy/absa/facebook-absa/data/combination/prosa/gaste_format \
            --trains "train_news_annotator" \
            --devs "test_news" \
            --tests "test_news" \
            --blank_frac 1.0 \
            --random_state 42 \
            --output_dir /srv/nas_data1/text/randy/absa/models/facebook_research/generative/t5/t5_pabsa_S256_wid_base_blank=1.0/aste \
            --per_device_predict_batch_size 64
            # --trains "train_socmed_twenty_percent train_news_annotator train_news_student" \