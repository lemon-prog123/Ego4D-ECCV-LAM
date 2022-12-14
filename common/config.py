import argparse

argparser = argparse.ArgumentParser(description='Ego4d Social Benchmark')

argparser.add_argument('--source_path', type=str, default='data/video_imgs', help='Video image directory')
argparser.add_argument('--json_path', type=str, default='data/json_original', help='Face tracklets directory')
argparser.add_argument('--test_path', type=str, default='data/videos_challenge', help='Test set')
argparser.add_argument('--gt_path', type=str, default='data/result_LAM', help='Groundtruth directory')
argparser.add_argument('--train_file', type=str, default='data/split/train.list', help='Train list')
argparser.add_argument('--val_file', type=str, default='data/split/val.list', help='Validation list')
argparser.add_argument('--train_stride', type=int, default=3, help='Train subsampling rate')
argparser.add_argument('--val_stride', type=int, default=13, help='Validation subsampling rate')
argparser.add_argument('--test_stride', type=int, default=1, help='Test subsampling rate')
argparser.add_argument('--epochs', type=int, default=20, help='Maximum epoch')
argparser.add_argument('--batch_size', type=int, default=64, help='Batch size')
argparser.add_argument('--num_workers', type=int, default=2, help='Num workers')
argparser.add_argument('--lr', type=float, default=5e-4, help='Learning rate')
argparser.add_argument('--u_a', type=float, default=2.0, help='u_a')
argparser.add_argument('--l_a', type=float, default=0.0, help='l_a')
argparser.add_argument('--alpha', type=float, default=0.25, help='Alpha for focal loss')
argparser.add_argument('--lr_drop_ratio', type=float, default=0.2, help='Learning rate decay')
argparser.add_argument('--weights', type=list, default=[0.0730, 0.9269], help='Class weight')
argparser.add_argument('--eval', action='store_true', help='Running type')
argparser.add_argument('--val', action='store_true', help='Running type')
argparser.add_argument('--dist', action='store_true', help='Launch distributed training')
argparser.add_argument('--model', type=str, default='BaselineLSTM', help='Model architecture')
argparser.add_argument('--transformer', action='store_true',help='Transformer architecture')
argparser.add_argument('--cat', action='store_true',help='Transformer cat')
argparser.add_argument('--mag', action='store_true',help='MagLoss')
argparser.add_argument('--rank', type=int, default=0, help='Rank id')
argparser.add_argument('--thread', type=int, default=0, help='Thread id')
argparser.add_argument('--start_rank', type=int, default=0, help='Start rank')
argparser.add_argument('--device_id', type=int, default=0, help='Device id')
argparser.add_argument('--world_size', type=int, help='Distributed world size')
argparser.add_argument('--init_method', type=str, help='Distributed init method')
argparser.add_argument('--backend', type=str, default='nccl', help='Distributed backend')
argparser.add_argument('--exp_path', type=str, default='output', help='Path to results')
argparser.add_argument('--train_path', type=str, default='output_train', help='Path to train_results')
argparser.add_argument('--checkpoint', type=str, help='Checkpoint to load')
argparser.add_argument('--filter',action='store_true',help='Using filter')
argparser.add_argument('--focal',action='store_true',help='Focal loss')
argparser.add_argument('--gaze',action='store_true',help='Gaze Es')
argparser.add_argument('--o',action='store_true',help='o Gaze360')
argparser.add_argument('--warmup',action='store_true',help='Warmup')
argparser.add_argument('--correction',action='store_true',help='Correct')
argparser.add_argument('--weight_decay',action='store_true',help='Weight_decay')
argparser.add_argument('--mean',action='store_true',help='Mean_decay')
argparser.add_argument('--temporal',action='store_true',help='Time')
argparser.add_argument('--Times',action='store_true',help='Times')
argparser.add_argument('--flip',action='store_true',help='Flip')
argparser.add_argument('--headpose',action='store_true',help='head')
argparser.add_argument('--head_query',action='store_true',help='head')
argparser.add_argument('--towards',action='store_true',help='towards')
argparser.add_argument('--pure',action='store_true',help='Pure Gaze')
argparser.add_argument('--cls',action='store_true',help='Use CLS Token')
argparser.add_argument('--var',action='store_true',help='Use var')
argparser.add_argument('--split',action='store_true',help='Use split')
argparser.add_argument('--con',action='store_true',help='contrast')
argparser.add_argument('--bn',action='store_true',help='contrast')
argparser.add_argument('--GRU',action='store_true',help='USE GRU INSTEAD OF LSTM')
argparser.add_argument('--blur',action='store_true',help='Blur')
argparser.add_argument('--mask',action='store_true',help='Mask')
argparser.add_argument('--twins',action='store_true',help='Twins')
argparser.add_argument('--early',action='store_true',help='early fusion')
argparser.add_argument('--random',action='store_true',help='random')
argparser.add_argument('--max-grad-norm', type=float, default=0.0, help='if the l2 norm is large than this hyper-parameter, then we clip the gradient  (default: 0.0, no gradient clip)')
argparser.add_argument('--weight-decay', type=float, default=0.02,  help='weight decay, similar one used in AdamW (default: 0.02)')
argparser.add_argument('--opt-eps', default=None, type=float, metavar='EPSILON', help='optimizer epsilon to avoid the bad case where second-order moment is zero (default: None, use opt default 1e-8 in adan)')
argparser.add_argument('--opt-betas', default=None, type=float, nargs='+', metavar='BETA', help='optimizer betas in Adan (default: None, use opt default [0.98, 0.92, 0.99] in Adan)')
argparser.add_argument('--no-prox', action='store_true', default=False, help='whether perform weight decay like AdamW (default=False)')
argparser.add_argument('--num_encoder_layers', type=int, default=6, help='Encoder Layers')
argparser.add_argument('--num_decoder_layers', type=int, default=6, help='Decoder Layers')
argparser.add_argument('--dim_feedforward', type=int, default=2048, help='dim_feedforward')