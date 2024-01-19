import os
import sys
sys.path.append("/opt/data/private/moe-lora/MOELoRA-peft")
import fire
import torch
from transformers import AutoConfig, AutoTokenizer

from transformers import LlamaForCausalLM, AutoModelForCausalLM
from transformers import GenerationConfig, LlamaTokenizer
from tqdm import tqdm
import pandas as pd 
import json
from src.MLoRA.peft import PeftModel, TaskType, get_peft_model
from src.MLoRA.peft import MMOELoraConfigS

from transformers import AutoConfig
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

try:
    if torch.backends.mps.is_available():
        device = "mps"
except:  # noqa: E722
    pass

ANSWER_IDS = [319,350,315,360]
ANSWER_MAP = {1:"A", 2:"B", 3:"C", 4:"D"}
def main(
    load_8bit: bool = False,
    base_model: str = "",
    lora_weights: str = None,
    rangel:int = 0,
    ranger:int = 56,
    share_gradio: bool = True,
    input_dir:str = "",
    output_dir:str = "",
    debug:bool = False
):
    # os.environ['CUDA_VISIBLE_DEVICES'] = str(device)
    assert (
        base_model
    ), "Please specify a --base_model, e.g. --base_model='decapoda-research/llama-7b-hf'"
    
    tokenizer = AutoTokenizer.from_pretrained(
    "/opt/data/private/wcy/models/Llama-2-7b-hf",
    trust_remote_code=True,
)
    # import pdb
    # pdb.set_trace()
    if debug:
        config = AutoConfig.from_pretrained("/opt/data/private/moe-lora/test_model")
        model = LlamaForCausalLM(config).cuda()
        
    else:
        if device == "cuda":

            model = AutoModelForCausalLM.from_pretrained(
                base_model,
                load_in_8bit=load_8bit,
                torch_dtype=torch.float16,
                device_map = "auto",
                trust_remote_code=True,
            )


    # unwind broken decapoda-research config
    model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
    model.config.bos_token_id = 1
    model.config.eos_token_id = 2

    if not load_8bit:
        model.half()  # seems to fix bugs for some users.

    model.eval()
    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)
    @torch.inference_mode()
    def evaluate(
        instruction,
    ):
        answers, ans_index = generate_prompt(instruction)

        input_ids = tokenizer.encode(answers)
        # input_ids.append(tokenizer.eos_token_id)
        attention_mask = torch.ones(len(input_ids),dtype=torch.long).to(device).unsqueeze(dim=0)
        input_ids = torch.LongTensor(input_ids).to(device).unsqueeze(dim=0)

        # inputs.pop("token_type_ids")
        # import pdb
        # pdb.set_trace()
        with torch.no_grad():
            logits = model(input_ids,attention_mask).logits
            ans_logits = logits[0][-1].index_select(-1, torch.LongTensor(ANSWER_IDS).to(device))
    
            index = torch.argmax(ans_logits).item()

        pred = ANSWER_MAP[index+1]
        # print(tokenizer.decode(input_ids[0]))
        # import pdb
        # pdb.set_trace()
        return pred, ans_index

    def construct_result_file(input_path:str,output_path:str):
        res = []

        with open(input_path,'r') as f:
            data = json.load(f)
        with tqdm(total=len(data)) as pbar:
            for line in tqdm(data):
                result, ans_index = evaluate(line)
                line["result"] = result
                line["ans"] = ans_index
                res.append(line)

        with open(output_path,"w") as f:
            json.dump(res,f,indent=2)

    file = input_dir
    for root, dirs, files in os.walk(file):
        for file in sorted(files)[rangel:ranger]:
            path = os.path.join(root, file)
            name = file[:-5] + ".json"
            output_path = os.path.join(output_dir,name)
            print(path)
            print(output_path)
            construct_result_file(path,output_path)

def generate_prompt(data):

    sample = data["data"]
   
    res = (
        f"The following are multiple choice questions (with answers).\n\n"
        f"{sample['question']}\n"
        f"A. {sample['res1']}\n"
        f"B. {sample['res2']}\n" 
        f"C. {sample['res3']}\n"
        f"D. {sample['res4']}\n"
        f"Answer:"
    )
    answer = sample['ans']
    return res, answer


if __name__ == "__main__":
    fire.Fire(main)
