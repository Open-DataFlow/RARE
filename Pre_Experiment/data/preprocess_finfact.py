import json
import argparse
OPTION="""\nA. true\nB. false\nC. NEI"""
def convert_json(json_path, output_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    new_data = []
    for item in data:
        new_item = {
            "id": item["url"],
            "x": item["claim"]+OPTION ,
            "R_x": item["documents"],
            "r": item["predict"],
            "output": item["output"],
        }
        new_data.append(new_item)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(new_data, f, indent=4, ensure_ascii=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert finfact JSON format')
    parser.add_argument('--input', type=str, required=True, help='Path to input JSON file')
    parser.add_argument('--output', type=str, required=True, help='Path to output JSON file')
    
    args = parser.parse_args()
    convert_json(args.input, args.output)