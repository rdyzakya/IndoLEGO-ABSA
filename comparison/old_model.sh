#!/usr/bin/env bash

base="/srv/nas_data1/text/randy/absa/models/astra"

python /srv/nas_data1/text/randy/absa/facebook-absa/comparison/old_model.py --concept ${base}/concept/xlm-roberta/xlm-roberta_lm_mul_L12_S256 \
            --sm ${base}/sentiment_marker/xlm-roberta/xlm-roberta_lm_mul_L12_S256 \
            --relation ${base}/relation/bert/bert_tf_mix_L12_S256 \
            --data /srv/nas_data1/text/randy/absa/data/astra/interim/gaste_format/test_astra_blank=1.0.txt \
            --n_gpu 2 \
            --batch_size 16 \
            --output_dir /srv/nas_data1/text/randy/absa