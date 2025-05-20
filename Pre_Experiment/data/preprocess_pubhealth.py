import json
import argparse

OPTION="""\nA. true - The statement is entirely accurate and supported by solid evidence.\nB. false - The statement is completely untrue and contradicted by strong evidence.\nC. mixture - The statement is partially true but contains some inaccuracies or misleading elements.\nD. unproven - There is insufficient evidence to confirm or refute the statement."""
def convert_json(json_path, output_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    new_data = []
    for item in data:
        new_item = {
            "id": item["id"],
            "x":item["text_1"]+OPTION,
            "R_x": item["text_2"],
            "r": item["predict"],
            "output": item["output"],
        }
        new_data.append(new_item)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(new_data, f, indent=4, ensure_ascii=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert pubhealth JSON format')
    parser.add_argument('--input', type=str, required=True, help='Path to input JSON file')
    parser.add_argument('--output', type=str, required=True, help='Path to output JSON file')
    
    args = parser.parse_args()
    convert_json(args.input, args.output)