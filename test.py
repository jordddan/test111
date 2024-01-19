# import json
# import logging
# import os
# import sys
# import numpy as np
# import transformers
# from datasets import load_dataset
# from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
# from rouge_chinese import Rouge
# from transformers import (
#     AutoConfig,
#     AutoModel,
#     AutoTokenizer,
#     DataCollatorForSeq2Seq,
#     set_seed,
#     AutoModelForCausalLM,
#     LlamaForCausalLM,
#     LlamaTokenizer,
# )

# sys.path.append("./")



# tokenizer = AutoTokenizer.from_pretrained(
#     "/opt/data/private/wcy/models/Llama-2-7b-hf",
#     trust_remote_code=True,
# )
# # ans = ["A","B","C","D"]
# # for item in ans:
# #     print(tokenizer(item))

# config = AutoConfig.from_pretrained("/opt/data/private/moe-lora/test_model")
# model = AutoModel.from_config(config=config).half().cuda()
# import pdb
# pdb.set_trace()


import os

for root, dirs, files in os.walk("/opt/data/private/moe-lora/data/mmlu/code_test/moelora6/"):
    for file in sorted(files)[0:57]:
        path = os.path.join(root, file)
        new_name = path.replace("_embedding_clustered","")
        os.rename(path,new_name)