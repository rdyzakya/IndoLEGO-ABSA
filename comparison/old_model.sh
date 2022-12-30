#!/usr/bin/env bash

base="/srv/nas_data1/text/randy/absa/models/facebook_research/hotel_id_v2/two_staged"

python /srv/nas_data1/text/randy/absa/facebook-absa/comparison/old_model.py --concept ${base}/aspect/spanbert/spanbert_lm_sg_L24_S256 \
            --sm ${base}/opinion/spanbert/spanbert_lm_sg_L24_S256 \
            --relation ${base}/relation/spanbert/spanbert_lm_sg_L24_S256 \
            --data /srv/nas_data1/text/randy/absa/facebook-absa/data/mono/hotel_id_v2/test.txt \
            --n_gpu 2 \
            --batch_size 16 \
            --output_dir /srv/nas_data1/text/randy/absa/facebook-absa/comparison/results/old_model/hotel_id_v2

# python /srv/nas_data1/text/randy/absa/facebook-absa/comparison/old_model.py --concept ${base}/aspect/spanbert/spanbert_lm_sg_L12_S256 \
#             --sm ${base}/opinion_sentiment/spanbert/spanbert_lm_sg_L12_S256 \
#             --relation ${base}/relation/bert/bert_lm_id_L12_S256 \
#             --data /srv/nas_data1/text/randy/absa/facebook-absa/data/combination/prosa/gaste_format/test_socmed.txt \
#             --n_gpu 2 \
#             --batch_size 16 \
#             --output_dir /srv/nas_data1/text/randy/absa/facebook-absa/comparison/results/old_model/test_socmed/v_3