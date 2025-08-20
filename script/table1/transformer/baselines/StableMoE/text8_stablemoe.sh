#!/bin/bash
echo 'Run training...'
python -u train.py \
    --cuda \
    --data ./data/text8 \
    --dataset text8 \
    --n_layer 4 \
    --d_model 256 \
    --n_head 8 \
    --d_head 64 \
    --d_inner 512 \
    --dropout 0.1 \
    --dropatt 0.0 \
    --optim adam \
    --lr 0.00035 \
    --warmup_step 0 \
    --max_step 100000 \
    --tgt_len 512 \
    --eval_tgt_len 128 \
    --batch_size 44 \
    --multi_gpu \
    --attn_type 2 \
    --mem_len 0 \
    --moe --moe-num-expert 16 --moe-top-k 2 \
    --gate_name CustomNaiveGate_Balance_StableMoE \
    --load_balance 0.01 \
    --work_dir Trans-BL-StableMoE-Text8
