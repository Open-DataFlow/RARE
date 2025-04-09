from datasets import load_dataset
import json
import os
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

template = """You are a professional medical expert in fact-checking, skilled in analyzing the accuracy of # Statement. Please first think step-by-step using the # Retrieved Documents and then check # Statement by using your own knowledge. Your responses will be used for research purposes only, so please have a definite answer.
You should respond in the format:
<think>
...
</think>
<answer>A/B/C/D</answer> (only one option can be chosen)

# Retrieved Documents
{text_2}

# Statement
{text_1}
A. true - The statement is entirely accurate and supported by solid evidence.
B. false - The statement is completely untrue and contradicted by strong evidence.
C. mixture - The statement is partially true but contains some inaccuracies or misleading elements.
D. unproven - There is insufficient evidence to confirm or refute the statement."""


def download_and_process_pubhealth():
    logging.info("Starting download of bigbio/pubhealth dataset...")

    data_dir = os.path.join(".", "data")
    os.makedirs(data_dir, exist_ok=True)

    dataset = load_dataset("bigbio/pubhealth", "pubhealth_bigbio_pairs", trust_remote_code=True)
    logging.info(f"Dataset loaded with splits: {dataset.keys()}")

    label_to_output = {
        "true": "<answer>A</answer>",
        "false": "<answer>B</answer>",
        "mixture": "<answer>C</answer>",
        "unproven": "<answer>D</answer>",
    }

    train_data = []

    for split in ["train", "validation"]:
        if split in dataset:
            split_data = dataset[split]
            logging.info(f"Processing {split} split with {len(split_data)} items")

            for item in split_data:
                processed_item = process_item(item, label_to_output, template)
                if processed_item:
                    train_data.append(processed_item)

    train_output_file = os.path.join(data_dir, "train_pubhealth.json")
    save_json(train_data, train_output_file)
    logging.info(f"Saved train data to {train_output_file} ({len(train_data)} items)")

    test_data = []
    if "test" in dataset:
        test_split = dataset["test"]
        logging.info(f"Processing test split with {len(test_split)} items")

        for item in test_split:
            processed_item = process_item(item, label_to_output, template)
            if processed_item:
                test_data.append(processed_item)

    test_output_file = os.path.join(data_dir, "test_pubhealth.json")
    save_json(test_data, test_output_file)
    logging.info(f"Saved test data to {test_output_file} ({len(test_data)} items)")

    logging.info("Processing complete!")


def process_item(item, label_to_output, instruction_template):

    label = item.get("label")
    if label not in label_to_output:
        logging.warning(f"Warning: Unknown label encountered: {label}")
        return None

    text_1 = item.get("text_1", "")
    text_2 = item.get("text_2", "")

    processed_item = {
        "output": label_to_output[label],
        "documents": text_2,
        "question": text_1,
        "instruction": instruction_template.format(text_1=text_1, text_2=text_2),
        "id": item.get("id", ""),
        "name": "pubhealth",
    }

    return processed_item


def save_json(data, output_file):
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    download_and_process_pubhealth()
