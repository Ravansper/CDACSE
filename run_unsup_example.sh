#!/bin/bash

# In this example, we show how to train SimCSE on unsupervised Wikipedia data.
# If you want to train it with multiple GPU cards, see "run_sup_example.sh"
# about how to use PyTorch's distributed data parallel.
#princeton-nlp/unsup-simcse-bert-base-uncased
#princeton-nlp/unsup-simcse-bert-large-uncased
#princeton-nlp/unsup-simcse-roberta-base
#princeton-nlp/unsup-simcse-roberta-large
#princeton-nlp/sup-simcse-bert-base-uncased
#princeton-nlp/sup-simcse-bert-large-uncased
#princeton-nlp/sup-simcse-roberta-base
#princeton-nlp/sup-simcse-roberta-large

python train.py \
    --model_name_or_path princeton-nlp/unsup-simcse-bert-base-uncased \
    --train_file data/train_stage1.txt \
    --output_dir /result/unsup-simcse-bert-base-uncased/ \
    --num_train_epochs 1 \
    --per_device_train_batch_size 31 \
    --learning_rate 2e-6 \
    --max_seq_length 30 \
    --evaluation_strategy steps \
    --metric_for_best_model stsb_spearman \
    --load_best_model_at_end \
    --eval_steps 5 \
    --pooler_type cls \
    --mlp_only_train \
    --overwrite_output_dir \
    --temp 0.01 \
    --do_train \
    --do_eval \
    --fp16 \
    "$@"
