#!/usr/bin/env bash

base="/srv/nas_data1/text/randy/absa/models/facebook_research/two_staged"

python /srv/nas_data1/text/randy/absa/facebook-absa/comparison/old_model.py --concept ${base}/aspect/spanbert/spanbert_lm_sg_L12_S256 \
            --sm ${base}/opinion-sentiment/spanbert/spanbert_lm_sg_L12_S256 \
            --relation ${base}/relation/bert/bert_tf_mix_L12_S256 \
            --data /srv/nas_data1/text/randy/absa/facebook-absa/data/combination/prosa/gaste_format/test_news.txt \
            --n_gpu 2 \
            --batch_size 16 \
            --output_dir /srv/nas_data1/text/randy/absa/facebook-absa/comparison/results/old_model/test_news