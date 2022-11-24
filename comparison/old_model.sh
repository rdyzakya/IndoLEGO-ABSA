#!/usr/bin/env bash

base="/srv/nas_data1/text/randy/absa/models/wartek"

python /srv/nas_data1/text/randy/absa/facebook-absa/comparison/old_model.py --concept ${base}/concept/xlm-roberta/xlm-roberta_lm_mul_L12_S256 \
            --sm ${base}/sentiment_marker/xlm-roberta/xlm-roberta_lm_mul_L12_S256 \
            --relation ${base}/relation/bert/bert_tf_mix_L12_S256 \
            --data /srv/nas_data1/text/randy/absa/data/combination/wartek/gaste_format/test_wartek.txt \
            --n_gpu 2 \
            --batch_size 16 \
            --output_dir /srv/nas_data1/text/randy/absa/error_analysis/wartek/result_test

# python /srv/nas_data1/text/randy/absa/facebook-absa/comparison/old_model.py --concept ${base}/aspect/spanbert/spanbert_lm_sg_L12_S256 \
#             --sm ${base}/opinion_sentiment/spanbert/spanbert_lm_sg_L12_S256 \
#             --relation ${base}/relation/bert/bert_lm_id_L12_S256 \
#             --data /srv/nas_data1/text/randy/absa/facebook-absa/data/combination/prosa/gaste_format/test_socmed.txt \
#             --n_gpu 2 \
#             --batch_size 16 \
#             --output_dir /srv/nas_data1/text/randy/absa/facebook-absa/comparison/results/old_model/test_socmed/v_3