import json
import re
import argparse

def extract_option(pred):
    for pattern in [
        r"<answer>(.*?)</answer>",
        r"<answer>(.*?)<answer>",
        r"^([A-Z])[.,:]",
        r"Answer:\s*([A-Z])\s*",
    ]:
        match = re.search(pattern, pred, re.DOTALL)
        if match is not None:
            pred = match.group(1)
    return pred.replace("<", "").replace(">", "").strip()

def convert_to_kto_format(input_json_path, output_json_path):
    with open(input_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    kto_data = []
    for item in data:
        kto_tag = extract_option(item["predict"]) == extract_option(item["output"])
        new_item = {
            "conversations": [
                {"from": "human", "value": item["instruction"]},
                {"from": "gpt", "value": item["predict"]}
            ],
            "kto_tag": kto_tag
        }
        kto_data.append(new_item)

    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(kto_data, f, indent=4, ensure_ascii=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert data to kto format")
    parser.add_argument("--input_path", type=str, required=True, help="Path to input JSON file")
    parser.add_argument("--output_path", type=str, required=True, help="Path to output JSON file")
    args = parser.parse_args()
    convert_to_kto_format(args.input_path, args.output_path)
