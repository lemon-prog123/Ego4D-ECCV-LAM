#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

nohup python -u run_copy.py  --head_query  --batch_size 32   --lr 1e-4 --val_stride 13  --o --filter --train_stride 3 --model GazeDETR --num_workers 3 --train_path output_gazequeryvaltrain --exp_path output_gazequeryval >gazequeryval.out &