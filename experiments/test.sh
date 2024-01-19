
# deepspeed --num_gpus=4 --master_port $MASTER_PORT run_mlora.py \
#     --do_predict \
#     --test_file $your_data_path/test.json \
#     --cache_dir $your_data_path \
#     --overwrite_cache \
#     --prompt_column input \
#     --response_column target \
#     --model_name_or_path $model_name_or_path \
#     --peft_path $your_checkpopint_path/checkpoint-${MAX_STEPS} \
#     --output_dir results/pred/moelora \
#     --overwrite_output_dir \
#     --max_source_length $MAX_SOURCE_LENGTH \
#     --max_target_length 196 \
#     --per_device_eval_batch_size 8 \
#     --predict_with_generate \
#     --lora_name moelora \
#     --expert_num 8
