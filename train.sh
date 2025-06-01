# hourly 12 steps
python FutureTST_train_eval_hourly_skip_batch.py --time_series_size 72 --d_model 128 --embedding_dropout 0.0 --mlp_dropout 0.1 --mlp_size 128 --num_heads 16 --num_transformer_layers 8 --patch_size 32 --pred_size 12 --stride_len 16

# hourly 24 steps
python FutureTST_train_eval_hourly_skip_batch.py --time_series_size 144 --d_model 128 --embedding_dropout 0.0 --mlp_dropout 0.1 --mlp_size 128 --num_heads 16 --num_transformer_layers 2 --patch_size 32 --pred_size 24 --stride_len 16

# hourly 48 steps
python FutureTST_train_eval_hourly_skip_batch.py --time_series_size 144 --d_model 128 --embedding_dropout 0.1 --mlp_dropout 0.1 --mlp_size 256 --num_heads 8 --num_transformer_layers 2 --patch_size 32 --pred_size 48 --stride_len 16

# 6-hourly 28 steps
python FutureTST_train_eval_hourly_skip_batch.py --time_series_size 216 --d_model 512 --embedding_dropout 0.0 --mlp_dropout 0.0 --mlp_size 64 --num_heads 16 --num_transformer_layers 8 --patch_size 32 --pred_size 28 --stride_len 16 --lr 0.00001 --sixhourly
