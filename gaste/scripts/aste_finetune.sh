#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES="1" python main.py --do_train \
            --do_eval \
            --do_predict \
            --n_gpu 1 \
            --output_dir /srv/nas_data1/text/randy/aste/models/generative \
            --data_dir /srv/nas_data1/text/randy/aste/facebook-aste/data/interim/gaste_format \
            --aste_train_args /srv/nas_data1/text/randy/aste/facebook-aste/gaste/train_args/aste_train_args.json \
            --train "train_socmed_twenty_percent train_news_annotator train_news_student" \
            --dev "test_news" \
            --test "test_news" \
            --aste_model_type t5 \
            --aste_model_name_or_path Wikidepia/IndoT5-base \
            --blank_frac 1.0 \
            --random_state 42 \
            --max_len 256 \
            --aste_model  \
            --prompt_option 0 \
            --quote \
            --quote_with_space
