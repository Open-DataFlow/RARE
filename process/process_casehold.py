import json
import os
from datasets import load_dataset

template = """
You are a professional legal expert specializing in case law analysis, skilled in analyzing determine which of the following holding # Statements correctly. Please first think step-by-step using the # Retrieved Documents and then identify the correct holding # Statement by using your own knowledge. Your responses will be used for research purposes only, so please have a definite answer.
You should respond in the format:
<think>
...
</think>
<answer>A/B/C/D/E</answer> (only one option can be chosen)

# Retrieved Documents
{citing_prompt}

# Statements
A. {holding_0}
B. {holding_1}
C. {holding_2}
D. {holding_3}
E. {holding_4}"""


def process_casehold_dataset():
    data_dir = os.path.join(".", "data")

    label_to_output = {
        "0": "<answer>A</answer>",
        "1": "<answer>B</answer>",
        "2": "<answer>C</answer>",
        "3": "<answer>D</answer>",
        "4": "<answer>E</answer>",
    }

    dataset = load_dataset("casehold/casehold", "all", trust_remote_code=True)

    for split_name in ["train", "test"]:
        print(f"processing {split_name} split...")
        split_data = dataset[split_name]

        processed_data = []

        count = 0

        for item in split_data:
            processed_item = {}

            for key in item:
                processed_item[key] = item[key]

            label = str(item.get("label"))
            if label in label_to_output:
                processed_item["output"] = label_to_output[label]
                count += 1
            else:
                print(f"warning, unknown label: {label}")

            citing_prompt = item.get("citing_prompt", "")
            holding_0 = item.get("holding_0", "")
            holding_1 = item.get("holding_1", "")
            holding_2 = item.get("holding_2", "")
            holding_3 = item.get("holding_3", "")
            holding_4 = item.get("holding_4", "")

            processed_item["instruction"] = template.format(
                citing_prompt=citing_prompt,
                holding_0=holding_0,
                holding_1=holding_1,
                holding_2=holding_2,
                holding_3=holding_3,
                holding_4=holding_4,
            )

            processed_data.append(processed_item)

        output_file = os.path.join(data_dir, f"{split_name}_casehold.json")
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(processed_data, f, ensure_ascii=False, indent=2)

        print(f"Have processed {count}, save to: {output_file}")


if __name__ == "__main__":
    process_casehold_dataset()
