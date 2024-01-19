# the same as eval_ppl_icl.sh
# for running two codes at the same time 

export CUDA_VISIBLE_DEVICES=$1

model_name=moelora6-ultrachat
input_dir=/opt/data/private/moe-lora/data/mmlu/raw_data/mpnet/ultrachat_clustered
output_path=/opt/data/private/moe-lora/data/mmlu/code_test/$model_name

mkdir -p $output_path
mkdir -p /opt/data/private/moe-lora/data/mmlu/result/$model_name

# 如果不需要adapter直接去掉--lora_weights就行
python /opt/data/private/moe-lora/MOELoRA-peft/eval/mmlu_zeroshot.py \
    --base_model /opt/data/private/wcy/models/Llama-2-7b-hf \
    --lora_weights /opt/data/private/moe-lora/MOELoRA-peft/saved/moelora6-ultrachat/checkpoint-800 \
    --data_type ultrachat \
    --moe_inference true \
    --input_dir $input_dir \
    --output_dir $output_path \
    --rangel $2 \
    --ranger $3


