import argparse
import json
from pathlib import Path

from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaTokenizer
from tqdm import tqdm
from peft import PeftModel

from src.data.loader import load_data
from src.utils.utils import list2dic
from src.models.score import JPTScore


def load_datasets(path: str) -> tuple[Dataset, Dataset]:
    data_turn, data_dialog = load_data(path)
    data_turn = list2dic(data_turn)
    data_dialog = list2dic(data_dialog)
    
    dataset_turn = Dataset.from_dict(data_turn)
    dataset_dialog = Dataset.from_dict(data_dialog)
    return dataset_turn, dataset_dialog


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
    parser.add_argument("--input", required=False, type=str, default="data/fed_data_ja.json")
    parser.add_argument("--model_path", required=True, type=str)
    parser.add_argument("--output_dir", required=False, type=str, default="results/")
    parser.add_argument("--lora_path", required=False, type=str, default=None)
    parser.add_argument("--response", action="store_true")
    parser.add_argument("--load_in_8bit", action="store_true")
    # parser.add_argument("--no_mask", action="store_false")
    parser.add_argument("--turn_only", action="store_true")
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    dataset_turn, dataset_dialog = load_datasets(args.input)
    tokenizer, model = load_model(args.model_path, args.load_in_8bit)
    if args.lora_path is not None:
        model = PeftModel.from_pretrained(model, args.lora_path, torch_dtype="auto")
    scorer = JPTScore(model, tokenizer)
    
    # Turn level evaluation
    results_turn = []
    for i, data in enumerate(tqdm(dataset_turn)):
        scores = scorer.score_batch(
            context=data["context"],
            response=data["response"],
            regex={"AI": "システム"},
            template_response=args.response,
        )
        data.update({"preds": scores})
        results_turn.append(data)
        if i < 1:
            print(data)
    with open(output_dir / "fed_turn.json", "w", encoding="utf-8") as f:
        json.dump(results_turn, f, indent=2, ensure_ascii=False)
    
    # Dialogue level evaluation
    if not args.turn_only:
        results_dialog = []
        for i, data in enumerate(tqdm(dataset_dialog)):
            scores = scorer.score_batch(
                context=data["context"],
            )
            data.update({"preds": scores})
            results_dialog.append(data)
            if i < 1:
                print(data)
        with open(output_dir / "fed_dialog.json", "w", encoding="utf-8") as f:
            json.dump(results_dialog, f, indent=2, ensure_ascii=False)
