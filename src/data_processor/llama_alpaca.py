# -*- encoding: utf-8 -*-
# here put the import lib
import json

import torch
from src.data_processor.prompter import Prompter

class AlpacaTrain(object):

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
        self.max_length = data_args.max_source_length
        self.prompter = Prompter()
        self.expert_type = model_args.expert_type
    def __call__(self, examples):

        tokenized_full_prompt = self.generate_and_tokenize_prompt(examples)
        input_ids = tokenized_full_prompt["input_ids"]
        labels = input_ids.copy()


        res = {"input_ids": torch.LongTensor(input_ids), "labels": torch.LongTensor(labels)}

        if self.model_args.expert_num > 1:  
            expert_weight = torch.Tensor(examples["cos_similarity"])
            if "top" in self.expert_type:
                topk = int(self.expert_type[-1])
                indices = torch.topk(expert_weight, k=topk).indices
                expert_weight = torch.zeros(expert_weight.shape)
                expert_weight[indices] = 1
            res["expert_weight"] = expert_weight 
        return res

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
        full_prompt = self.prompter.generate_prompt(
            data_point["instruction"],
            data_point["input"],
            data_point["output"],
        )
        tokenized_full_prompt = self.tokenize(full_prompt)
        user_prompt = self.prompter.generate_prompt(
            data_point["instruction"], data_point["input"]
        )
        tokenized_user_prompt = self.tokenize(
            user_prompt, add_eos_token=False
        )
        user_prompt_len = len(tokenized_user_prompt["input_ids"])

        # if add_eos_token:
        #     user_prompt_len -= 1

        tokenized_full_prompt["labels"] = [
            -100
        ] * user_prompt_len + tokenized_full_prompt["labels"][
            user_prompt_len:
        ]  # could be sped up, probably

        return tokenized_full_prompt




class AlpacaEval(object):

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
        self.max_length = data_args.max_source_length
        self.prompter = Prompter()

    def __call__(self, examples):
    
        # max_target_length = self.data_args.max_target_length
        # inputs, targets = [], []

        # if self.task:
        #     task_id = []
        #     task_dict = json.load(open("datasets/pre_data/task_dataset.json", "r"))
        #     task_dict = task_dict["str2id"]

        # for i in range(len(examples[self.prompt_column])):
        #     if self.examples[self.prompt_column][i]:
        #         query = examples[self.prompt_column][i]
        #         history = examples[self.history_column][i] if self.history_column is not None else None
        #         prompt = self.tokenizer.build_prompt(query, history)
        #         inputs.append(prompt)
        #         targets.append(examples[self.response_column][i])
            
        #     if self.task:
        #         task_id.append(task_dict[examples['task_dataset'][i]])

        # inputs = [self.prefix + inp for inp in inputs]
        # model_inputs = self.tokenizer(inputs,
        #                             max_length=self.data_args.max_source_length,
        #                             truncation=True,
        #                             padding=True)
        # labels = self.tokenizer(text_target=targets, max_length=max_target_length, truncation=True)

        # if self.data_args.ignore_pad_token_for_loss:
        #     labels["input_ids"] = [
        #         [(l if l != self.tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
        #     ]
        # model_inputs["labels"] = labels["input_ids"]

        # if self.task:
        #     model_inputs["task_id"] = task_id
        
        tokenized_full_prompt = self.generate_and_tokenize_prompt(examples)
        # import pdb
        # pdb.set_trace()
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
        full_prompt = self.prompter.generate_prompt(
            data_point["instruction"],
            data_point["input"],
            data_point["output"],
        )
        tokenized_full_prompt = self.tokenize(full_prompt)
        user_prompt = self.prompter.generate_prompt(
            data_point["instruction"], data_point["input"]
        )
        tokenized_user_prompt = self.tokenize(
            user_prompt, add_eos_token=False
        )
        user_prompt_len = len(tokenized_user_prompt["input_ids"])

        # if add_eos_token:
        #     user_prompt_len -= 1

        tokenized_full_prompt["labels"] = [
            -100
        ] * user_prompt_len + tokenized_full_prompt["labels"][
            user_prompt_len:
        ]  # could be sped up, probably
        return tokenized_full_prompt
