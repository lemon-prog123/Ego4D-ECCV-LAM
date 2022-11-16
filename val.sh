#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

nohup python -u run.py --batch_size 128 --val_stride 13 --head_query  --checkpoint [your checkpoint] --model GazeDETR  --o --num_workers 3 --val --exp_path output_gazequeryval >gazequeryval.out &