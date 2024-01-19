# the same as eval_ppl_icl.sh
# for running two codes at the same time 

export CUDA_VISIBLE_DEVICES=$1

model_name=llama2
input_dir=/opt/data/private/moe-lora/data/mmlu/raw_data/mmlu
output_path=/opt/data/private/moe-lora/data/mmlu/code_test/$model_name

mkdir -p $output_path

# 如果不需要adapter直接去掉--lora_weights就行
python /opt/data/private/moe-lora/MOELoRA-peft/eval/mmlu_zeroshot_llama.py \
    --base_model /opt/data/private/wcy/models/Llama-2-7b-hf \
    --input_dir $input_dir \
    --output_dir $output_path \
    --rangel $2 \
    --ranger $3
