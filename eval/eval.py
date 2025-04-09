import json
import re
import argparse


def extract_option(pred):
    # 1. get A/B/C/D
    for pattern in [
        r"<answer>(.*?)</answer>",
        r"<answer>(.*?)<answer>",
        r"^([A-Z])[.,:]",
        r"Answer:\s*([A-Z])\s*",
    ]:
        match = re.search(pattern, pred, re.DOTALL)
        if match is not None:
            pred = match.group(1)

    # 2. remove <>
    pred = pred.replace("<", "").replace(">", "")
    pred = pred.strip()

    return pred


def calculate_accuracy(data, prediction_key):
    correct = 0
    total = len(data)

    valid_options = ["A", "B", "C", "D", "E"]
    valid_extractions = 0

    for item in data:

        output = extract_option(item["output"])

        predict = extract_option(item[prediction_key])

        if predict in valid_options:
            valid_extractions += 1

        if output == predict:
            correct += 1
        # else:
        #     print(f"Incorrect: {predict} vs {output}")

    accuracy = correct / total if total > 0 else 0

    return accuracy, correct, total, valid_extractions


def main():
    parser = argparse.ArgumentParser(description="Calculate accuracy")
    parser.add_argument("--file", type=str, required=True, help="path")
    parser.add_argument("--prediction_key", type=str, help="name of key")

    args = parser.parse_args()

    with open(args.file, "r") as f:
        data = json.load(f)


    accuracy, correct, total, valid_extractions = calculate_accuracy(data, args.prediction_key)

    print(f"accuracy: {accuracy * 100:.2f}%, {correct}, {total}")
    print(f"Valid extraction rate: {valid_extractions}/{total}")


if __name__ == "__main__":
    main()
