#!/bin/zsh

python practice_training.py \
 --model_name_or_path "microsoft/phi-2" \
 --dataset_name "databricks/databricks-dolly-15k" \
 --output_dir "../data/fine_tuned_phi2" \
 --logging_dir "../data/fine_tuned_phi2/logs" \
 --num_train_epochs 10 \
 --per_device_train_batch_size 64 \
 --per_device_eval_batch_size 64 \
 --learning_rate 1e-3 \
 --do_train True \
 --do_eval True \
 --evaluation_strategy "epoch" \
 --save_strategy "epoch" \
 --logging_strategy "epoch" \
 --load_best_model_at_end True
