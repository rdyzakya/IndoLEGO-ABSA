#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES="1" python main.py --do_train \
            --do_eval \
            --do_predict \
            --n_gpu 1 \
            --output_dir /srv/nas_data1/text/randy/aste/models/triplet_detection \
            --data_dir /srv/nas_data1/text/randy/aste/facebook-aste/data/interim/gaste_format \
            --triplet_detection_train_args /srv/nas_data1/text/randy/aste/facebook-aste/gaste/train_args/triplet_detection_train_args.json \
            --train "train_news_annotator train_news_student" \
            --dev "test_news" \
            --test "test_news" \
            --triplet_detection_model_type bert \
            --triplet_detection_model_name_or_path indolem/indobert-base-uncased \
            --blank_frac 1.0 \
            --random_state 42 \
            --max_len 256 \
            --triplet_detection_model
