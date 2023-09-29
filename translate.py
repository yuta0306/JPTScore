import argparse
import json
from pathlib import Path

from tqdm import tqdm
from google.cloud import translate_v2 as translate

from src.data.loader import load_data


def get_translation(client, src: dict):
    ret = {}
    for k, v in src.items():
        if k in ("context", "response"):
            result = client.translate(
                v,
                source_language="en",
                target_language="ja"
            )
            ret[k] = result["translatedText"]
        else:
            ret[k] = v
    return ret


parser = argparse.ArgumentParser()
parser.add_argument("--input", required=True, type=str)
parser.add_argument("--output_dir", required=False, type=str, default="data/")
args = parser.parse_args()

assert args.input.endswith(".json")

translate_client = translate.Client()
data_turn, data_dialog = load_data(args.input)

# Turn level
data_turn_ja = [get_translation(translate_client, item) for item in tqdm(data_turn, leave=False)]

# Dialogue level
data_dialog_ja = [get_translation(translate_client, item) for item in tqdm(data_dialog, leave=False)]

output_dir = Path(args.output_dir)

with open(output_dir / "fed_data_ja.json", "w") as f:
    json.dump(data_turn_ja + data_dialog_ja, f)