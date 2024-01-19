# -*- encoding: utf-8 -*-
# here put the import lib
import json

IGNORE_INDEX = -100

from src.data_processor.prompter import Prompter
import torch

class UltrachatTrain(object):

    def __init__(self, data_args, model_args, prompt_column, 
                 response_column, history_column, prefix, tokenizer, task=False) -> None:
        
        self.data_args = data_args
        self.model_args = model_args
        self.prompt_column = prompt_column
        self.response_column = response_column
        self.history_column = history_column
        self.prefix = prefix
        self.tokenizer = tokenizer
        self.task = task
        self.max_seq_length = data_args.max_source_length
        self.prompter = Prompter()
        self.start_token = "\n"
        self.end_token = self.tokenizer.eos_token
        
    def __call__(self, examples):

        tokenized_full_prompt = self.generate_and_tokenize_prompt(examples)

        return tokenized_full_prompt

    def tokenize(self, prompt, add_eos_token=True):
        # there's probably a way to do this with the tokenizer settings
        # but again, gotta move fast
        result = self.tokenizer(
            prompt,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors=None,
        )
        import pdb
        pdb.set_trace()
        if (
            result["input_ids"][-1] != self.tokenizer.eos_token_id
            and len(result["input_ids"]) < self.max_length
            and add_eos_token
        ):
            result["input_ids"].append(self.tokenizer.eos_token_id)
            result["attention_mask"].append(1)

        result["labels"] = result["input_ids"].copy()

        return result

    def generate_and_tokenize_prompt(self,data_point):
        
        end_token = self.end_token
        labels = []
        tokenized_ids = []
        for i, c in enumerate(data_point["content"]):
            role = c["role"]
            content = c["content"]
            if role == "assistant":
                # model
                c_input = self.start_token + "Assitant" + ": "
                tokenized = self.tokenizer(c_input, add_special_tokens=False)
                tokenized_ids += tokenized["input_ids"]
                labels += [IGNORE_INDEX] * len(tokenized["input_ids"])

                c_generate = content + end_token
                tokenized = self.tokenizer(c_generate, add_special_tokens=False)
                tokenized_ids += tokenized["input_ids"]
                labels += tokenized["input_ids"]

            else:
                # user
                if i == 0:
                    # no start token
                    c_new = self.tokenizer.bos_token + "User" + ": " + content + end_token
                else:
                    c_new = self.start_token + "User" + ": " + content + end_token
                tokenized = self.tokenizer(c_new, add_special_tokens=False)
                tokenized_ids += tokenized["input_ids"]
                labels += [IGNORE_INDEX] * len(tokenized["input_ids"])

        assert len(tokenized_ids) == len(labels)
        
        res = {"input_ids": torch.LongTensor(tokenized_ids), "labels": torch.LongTensor(labels)}
        
        res = self.truncate(res)

        if self.model_args.expert_num > 1:
            res["expert_weight"] = torch.Tensor(data_point["cos_similarity"])
        
        return res
        
        
    def truncate(self, tokenized_example):
        old_len = len(tokenized_example["input_ids"])
        if old_len > self.max_seq_length:
            for k in tokenized_example:
                tokenized_example[k] = tokenized_example[k][:-(old_len - self.max_seq_length)]

        return tokenized_example
