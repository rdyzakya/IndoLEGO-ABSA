#!/usr/bin/env bash

base="/srv/nas_data1/text/randy/aste/models/two_staged"

python /srv/nas_data1/text/randy/aste/facebook-aste/comparison/old_model.py --concept ${base}/concept/spanbert/spanbert_lm_sg_L12_S256 \
            --sm ${base}/sentiment_marker/spanbert/spanbert_lm_sg_L12_S256 \
            --relation ${base}/relation/bert/bert_tf_mix_L12_S256 \
            --data /srv/nas_data1/text/randy/aste/facebook-aste/data/interim/gaste_format/test_socmed.txt \
            --n_gpu 1 \
            --batch_size 4 \
            --output_dir /srv/nas_data1/text/randy/aste/facebook-aste/comparison/results/old_model/test_socmed/v_2