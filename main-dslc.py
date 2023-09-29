import argparse
import json
from pathlib import Path

from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaTokenizer
from tqdm import tqdm
from peft import PeftModel

from src.data.loader import load_dslc
from src.utils.utils import list2dic
from src.models.score import JPTScore


def load_dataset(path: str) -> Dataset:
    data = load_dslc(path, task="open")
    def turns2str(turns: list[dict]) -> str:
        string = ""
        mapping = {"S": "システム", "U": "ユーザ"}
        for turn in turns:
            string += f"{mapping[turn['speaker']]}：{turn['utterance']} "
        return string
    data = [
        {
            "context": turns2str(dialog["turns"]),
            "scores": dialog["scores"],
            "topics": dialog["topics"]
        }
        for dialog in data
    ]
    return Dataset.from_dict(list2dic(data))


def load_model(path: str, load_in_8bit: bool):
    if path in ("stabilityai/japanese-stablelm-base-alpha-7b"):
        tokenizer = LlamaTokenizer.from_pretrained("novelai/nerdstash-tokenizer-v1", additional_special_tokens=['▁▁'], trust_remote_code=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(path)
    if load_in_8bit:
        model = AutoModelForCausalLM.from_pretrained(path, device_map="auto", load_in_8bit=True, trust_remote_code=True)
    else:
        model = AutoModelForCausalLM.from_pretrained(path, device_map="auto", torch_dtype="auto", trust_remote_code=True)
    return tokenizer, model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=False, type=str, default="data/livecompetition3/")
    parser.add_argument("--model_path", required=True, type=str)
    parser.add_argument("--name", required=True, type=str)
    parser.add_argument("--output_dir", required=False, type=str, default="results/dslc/")
    parser.add_argument("--lora_path", required=False, type=str, default=None)
    parser.add_argument("--load_in_8bit", action="store_true")
    # parser.add_argument("--no_mask", action="store_false")
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    dataset = load_dataset(args.input)
    tokenizer, model = load_model(args.model_path, args.load_in_8bit)
    if args.lora_path is not None:
        model = PeftModel.from_pretrained(model, args.lora_path, torch_dtype="auto")
    scorer = JPTScore(model, tokenizer)
    
    # Turn level evaluation
    results_turn = []
    for i, data in enumerate(tqdm(dataset)):
        scores = scorer.score_batch_dslc(
            context=data["context"],
            topics=data["topics"],
        )
        data.update({"preds": scores})
        results_turn.append(data)
        if i < 1:
            print(data)
    with open(output_dir / f"{args.name}.json", "w", encoding="utf-8") as f:
        json.dump(results_turn, f, indent=2, ensure_ascii=False)
