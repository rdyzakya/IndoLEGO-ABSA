#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=2 python /srv/nas_data1/text/randy/absa/facebook-absa/gabsa/main.py --do_train \
            --do_eval \
            --do_predict \
            --train_args /srv/nas_data1/text/randy/absa/facebook-absa/gabsa/train_args/train_args.json \
            --model_type byt5 \
            --model_name_or_path google/byt5-small \
            --max_len 1024 \
            --task "aste aope uabsa ate ote" \
            --paradigm extraction \
            --prompt_path /srv/nas_data1/text/randy/absa/facebook-absa/gabsa/prompts/prompt.json \
            --prompt_option_path /srv/nas_data1/text/randy/absa/facebook-absa/gabsa/prompts/option.json \
            --pattern /srv/nas_data1/text/randy/absa/facebook-absa/gabsa/patterns/default.json \
            --data_dir /srv/nas_data1/text/randy/absa/facebook-absa/data/combination/prosa/gaste_format \
            --trains "train_news_annotator train_news_student train_socmed_HaloBCA train_socmed_Telmark train_socmed_V3_sentiment" \
            --devs "test_news" \
            --tests "test_news" \
            --blank_frac 1.0 \
            --random_state 42 \
            --output_dir /srv/nas_data1/text/randy/absa/models/facebook_research/generative \
            --per_device_predict_batch_size 4
            # --trains "train_news_annotator train_news_student train_socmed_HaloBCA train_socmed_Telmark train_socmed_V3_sentiment" \
            # --task "aste aope ate ote uabsa" \