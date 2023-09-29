import argparse
import json
from pathlib import Path

import pandas as pd
import numpy as np

from scipy.stats import pearsonr, spearmanr, kendalltau


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=str)
    parser.add_argument("--output", required=True, type=str)
    parser.add_argument("--dslc", action="store_true")
    args = parser.parse_args()

    assert args.output.endswith(".json") and args.input.endswith(".json")
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    
    results = json.load(open(args.input, "r"))

    if args.dslc:
        df_gold = pd.DataFrame(data=[res["scores"] for res in results])
    else:
        df_gold = pd.DataFrame(data=[
            {
                k: np.nanmean([np.nan if score is None else score for score in v])
                for k, v in res["annotations"].items()
            }
            for res in results
        ])
    df = pd.DataFrame(data=[res["preds"] for res in results])
    
    dimensions = df.columns
    corrs = {}
    for dim in dimensions:
        pred = df[dim]
        gold = df_gold[dim]
        res = {}
        gold = np.nan_to_num(gold, nan=np.nanmean(gold))
        corr, p = pearsonr(gold, pred)
        res["pearson"] = {
            "coefficient": corr,
            "p-value": p,
        }
        corr, p = spearmanr(gold, pred)
        res["spearman"] = {
            "coefficient": corr,
            "p-value": p,
        }
        corr, p = kendalltau(gold, pred)
        res["kendalltau"] = {
            "coefficient": corr,
            "p-value": p,
        }
        corrs[dim] = res
    
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(corrs, f, indent=2, ensure_ascii=False)
