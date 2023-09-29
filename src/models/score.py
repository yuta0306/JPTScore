from typing import Literal
import re

import torch
import torch.nn as nn

from src.data.dataset import TEMPLATE, QUESTIONS, QUESTIONS_DSLC, TEMPLATE_WITH_TOPICS, TEMPLATE_WITH_SITUATION, TEMPLATE_RESPONSE


class JPTScore(object):
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.loss_fct = nn.CrossEntropyLoss(reduction="sum")
        self.model.eval()
        
    def score_dimension(
        self,
        *,
        context: str,
        dimension: str,
        response: str | None = None,
        mask_loss: bool = True,
    ) -> int:
        level = "turn" if response is not None else "dialogue"
        question = QUESTIONS[level][dimension]
        if response is not None:
            context += " " + response
        prompt = TEMPLATE.format_map({"question": question, "history": context})
        prompt = self.tokenizer(prompt, return_tensors="pt")
        prompt["labels"] = prompt.input_ids.clone()
        if mask_loss:
            mask = self.tokenizer(prompt.removesuffix("はい"))
            prompt["labels"][:len(mask)] = -100
        prompt = {k: v.to(self.model.device) for k, v in prompt.items()}
        with torch.no_grad():
            outputs = self.model(**prompt)
        logit = outputs.logits[0]
        label = prompt["labels"][0]
        return -self.loss_fct(logit, label).item() / len(label[label != -100])
    
    def _replace(self, string: str, regex: dict[str, str]) -> str:
        for k, v in regex.items():
            string = re.sub(k, v, string)
        return string
        
    def score_batch(
        self,
        *,
        context: str,
        response: str | None = None,
        mask_loss: bool = True,
        max_tokens: int = 600,
        regex: dict[str, str] = {"": ""},
        template_response: bool = False,
    ) -> dict[str, int]:
        level = "turn" if response is not None else "dialogue"
        T = TEMPLATE
        mapping = {"response": response}
        def update(org, dic):
            org_cpy = org.copy()
            org_cpy.update(dic)
            return org_cpy
        
        if template_response:
            T = TEMPLATE_RESPONSE
            regex = {
                "AIによる発話": "与えられた会話に続く，次のAIによる応答"
            }
            mapping.update()
        if response is not None and not template_response:
            context += " " + response
        
        prompts = [
            T.format_map(update(mapping, {
            "question": self._replace(question, regex),
            "history": context,
        }))
            for question in QUESTIONS[level].values()
        ]
        masks = prompts.copy()
        prompts = self.tokenizer(
            prompts,
            padding="longest",
            return_tensors="pt",
        )
        prompts["labels"] = prompts.input_ids.clone()
        if mask_loss:
            for labels, mask in zip(prompts["labels"], masks):
                mask = self.tokenizer(mask.removesuffix("はい")).input_ids
                labels[:len(mask)] = -100
        prompts = {k: v.to(self.model.device) for k, v in prompts.items()}
        prompts["labels"][prompts["labels"] == self.tokenizer.pad_token_id] = -100
        if "token_type_ids" in list(prompts.keys()):
            prompts.pop("token_type_ids")
        
        with torch.no_grad():
            if prompts["input_ids"].size(1) < max_tokens:
                logits = self.model(**prompts).logits
            else:  # prevent out of memory
                logits = torch.stack([
                    self.model(
                        input_ids=ids.unsqueeze(0),
                        attention_mask=mask.unsqueeze(0),
                    ).logits[0]
                    for ids, mask in zip(prompts["input_ids"], prompts["attention_mask"])
                ])
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = prompts["labels"][..., 1:].contiguous()
        ret =  {
            dimension: -self.loss_fct(logit, label).item() / len(label[label != -100])
            for dimension, logit, label in zip(
                QUESTIONS[level].keys(),
                shift_logits,
                shift_labels,
            )
        }
        
        return ret
    
    def score_batch_dslc(
        self,
        *,
        context: str,
        topics: list[str],
        max_tokens: int = 1024,
        mask_loss: bool = True,
    ) -> dict[str, int]:
        task = "open"
        prompts = [
            TEMPLATE.format_map({"question": QUESTIONS_DSLC[task]["自然性"], "history": context}),
            TEMPLATE_WITH_TOPICS.format_map({"question": QUESTIONS_DSLC[task]["話題追随"], "history": context, "topics": "，".join(topics)}),
            TEMPLATE_WITH_TOPICS.format_map({"question": QUESTIONS_DSLC[task]["話題提供"], "history": context, "topics": "，".join(topics)}),
        ]
        masks = prompts.copy()
        prompts = self.tokenizer(
            prompts,
            padding="longest",
            return_tensors="pt",
        )
        prompts["labels"] = prompts.input_ids.clone()
        if mask_loss:
            for labels, mask in zip(prompts["labels"], masks):
                mask = self.tokenizer(mask.removesuffix("はい")).input_ids
                labels[:len(mask)] = -100
        prompts = {k: v.to(self.model.device) for k, v in prompts.items()}
        prompts["labels"][prompts["labels"] == self.tokenizer.pad_token_id] = -100
        if "token_type_ids" in list(prompts.keys()):
            prompts.pop("token_type_ids")
        
        with torch.no_grad():
            if prompts["input_ids"].size(1) < max_tokens:
                logits = self.model(**prompts).logits
            else:  # prevent out of memory
                logits = torch.stack([
                    self.model(
                        input_ids=ids.unsqueeze(0),
                        attention_mask=mask.unsqueeze(0),
                    ).logits[0]
                    for ids, mask in zip(prompts["input_ids"], prompts["attention_mask"])
                ])
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = prompts["labels"][..., 1:].contiguous()
        ret =  {
            dimension: -self.loss_fct(logit, label).item() / len(label[label != -100])
            for dimension, logit, label in zip(
                QUESTIONS_DSLC[task].keys(),
                shift_logits,
                shift_labels,
            )
        }
        
        return ret