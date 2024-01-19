
lora_rank=48
lora_trainable="q_proj,k_proj,v_proj,o_proj"
modules_to_save="null"
lora_dropout=0.1
LR=2e-4
MAX_STEPS=2400
SAVE_STEPS=100
MASTER_PORT=$(shuf -n 1 -i 10000-65535)
model_name_or_path="/opt/data/private/wcy/models/Llama-2-7b-hf"   
your_data_path="/opt/data/private/moe-lora/datasets"  
your_checkpopint_path="/opt/data/private/moe-lora/MOELoRA-peft/saved/ultrachat"  
mkdir -p $your_checkpopint_path
MAX_SOURCE_LENGTH=1024
data_type="ultrachat"
peft_path=""  

# # Training Command
deepspeed  --include="localhost:4,5,6,7" --master_port $MASTER_PORT run_mlora.py \
    --deepspeed src/ds.json \
    --do_train \
    --train_file /opt/data/private/moe-lora/data/corpus/mpnet/ultra_chat.json \
    --data_type $data_type \
    --cache_dir $your_data_path \
    --prompt_column input \
    --response_column target \
    --overwrite_cache \
    --model_name_or_path $model_name_or_path \
    --output_dir $your_checkpopint_path \
    --overwrite_output_dir \
    --max_source_length $MAX_SOURCE_LENGTH \
    --max_target_length 196 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --max_steps ${MAX_STEPS} \
    --logging_steps 10 \
    --save_steps ${SAVE_STEPS} \
    --learning_rate $LR \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.03 \
    --weight_decay 0 \
    --lora_rank ${lora_rank} \
    --trainable ${lora_trainable} \
    --modules_to_save ${modules_to_save} \
    --lora_dropout ${lora_dropout} \
    --preprocessing_num_workers 16 \
    --bf16 \
    --lora_name moelora \
    --expert_num 1 | tee ${your_checkpopint_path}/res.log

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



