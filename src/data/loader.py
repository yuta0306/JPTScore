import json
from typing import Literal
from pathlib import Path

import pandas as pd


def load_data(path: str) -> list:
    data = json.load(open(path, "r"))
    turn_level = [item for item in data if "response" in tuple(item.keys())]
    dialog_level = [item for item in data if "response" not in tuple(item.keys())]
    
    for item in turn_level:
        for k, dimension in item["annotations"].items():
            item["annotations"][k] = convert_nan(dimension)
    for item in dialog_level:
        for k, dimension in item["annotations"].items():
            item["annotations"][k] = convert_nan(dimension)
    return turn_level, dialog_level


def convert_nan(dimension: list[int, str]) -> list[int, None]:
    return [score if isinstance(score, int) else None for score in dimension]


def load_dslc(path: str, task: Literal["open", "situation"]):
    path = Path(path) / task
    dataset = []
    scores = None
    if task == "open":
        header = ["name", "自然性", "話題追随", "話題提供", "topic1", "topic2"]
        metrics = ["自然性", "話題追随", "話題提供"]
    else:
        header = []
        metrics = []
    for file in path.glob("**/*_score.csv"):
        if scores is None:
            scores = pd.read_csv(file, names=header)
        else:
            scores = pd.concat([scores, pd.read_csv(file, names=header)], axis=0)
    scores = scores.reset_index(drop=True)
            
    for top in path.glob("*"):
        if top.is_file():
            continue
        for file in top.glob("*.log.json"):
            data = json.load(open(file, "r"))
            score = scores[scores["name"] == data["dialogue-id"]][metrics].to_dict()
            score = {k: s for k, v in score.items() for s in v.values()}
            data.update({"scores": score})
            if task == "open":
                topics = scores[scores["name"] == data["dialogue-id"]][["topic1", "topic2"]].to_dict()
                data.update({"topics": [t for topic in topics.values() for t in topic.values()]})
            dataset.append(data)
    return dataset


if __name__ == "__main__":
    data_turn, data_dialog = load_data("/home/jupyter/JPTScore/data/fed_data.json")
    print(len(data_turn), len(data_dialog))
    print(data_turn[0])
    print(data_dialog[0])