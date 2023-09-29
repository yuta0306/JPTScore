import argparse
import json
from pathlib import Path

import pandas as pd


def load_results(
    top: str,
    metric: str = "spearman",
    turn_only: bool = False,
):
    top = Path(top)
    res_turns = []
    res_dialogs = []
    for directory in top.glob("*"):
        name = directory.name
        try:
            turns = json.load(open(directory / "fed_turn.json", "r"))
            if not turn_only:
                dialog = json.load(open(directory / "fed_dialog.json", "r"))
        except:
            continue
        turns = {k: v[metric]["coefficient"] for k, v in turns.items()}
        turns["model"] = name
        res_turns.append(turns)
        if not turn_only:
            dialog = {k: v[metric]["coefficient"] for k, v in dialog.items()}
            dialog["model"] = name
            res_dialogs.append(dialog)
    if not turn_only:
        return pd.DataFrame(data=res_turns).set_index("model"), pd.DataFrame(data=res_dialogs).set_index("model")
    return pd.DataFrame(data=res_turns).set_index("model"), None
    

def load_dslc(top: str, metric: str = "spearman"):
    top = Path(top)
    ret = []
    for filename in top.glob("*.json"):
        name = filename.name
        try:
            res = json.load(open(filename, "r"))
        except:
            continue
        res = {k: v[metric]["coefficient"] for k, v in res.items()}
        res["model"] = name
        ret.append(res)
    return pd.DataFrame(data=ret).set_index("model")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dslc", action="store_true")
    parser.add_argument("--output_dir", required=False, type=str, default="outputs/")
    parser.add_argument("--turn_only", action="store_true")
    args = parser.parse_args()
    
    if args.dslc:
        results = load_dslc(args.output_dir)
        results = results.loc[results.mean(axis=1).sort_values(ascending=False).keys(), :]
        print(results.to_markdown(floatfmt=".3f"))
    else:
        turns, dialogs = load_results(args.output_dir, turn_only=args.turn_only)
        print(turns)
        turns = turns.loc[turns.mean(axis=1).sort_values(ascending=False).keys(), :]
        print(turns.to_markdown(floatfmt=".3f"))
        if not args.turn_only:
            dialogs = dialogs.loc[dialogs.mean(axis=1).sort_values(ascending=False).keys(), :]
            print(dialogs.to_markdown(floatfmt=".3f"))
