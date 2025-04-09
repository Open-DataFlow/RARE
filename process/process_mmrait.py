import json
import os
import shutil

input_file1 = "process/fact_verify/train_data.jsonl"
output_file1 = "data/train_mmrait.json"

input_file2 = "process/fact_verify/val_data.jsonl"
output_file2 = "data/test_mmrait.json"

source_train_img_dir = "process/fact_verify/train_document_images"
target_train_img_dir = "data/train_document_images"

source_val_img_dir = "process/fact_verify/val_document_images"
target_val_img_dir = "data/val_document_images"

instruction_template = """You are a professional medical expert in fact-checking, skilled in analyzing the accuracy of # Statement. Please first think step-by-step using the # Retrieved Documents and # Image related and then check # Statement by using your own knowledge. Your responses will be used for research purposes only, so please have a definite answer.

You should respond in the format:
<think>
...
</think>
<answer>A/B/C</answer> (only one option can be chosen)

# Retrieved Documents
{documents}

# Image
<image>

# Statement
{claim}
A. Support
B. Refute
C. Insufficient"""

def process_line(line, is_train=True):
    data = json.loads(line)

    if data["category"] == "Support":
        data["output"] = "<answer>A</answer>"
    elif data["category"] == "Refute":
        data["output"] = "<answer>B</answer>"
    elif data["category"] == "Insufficient":
        data["output"] = "<answer>C</answer>"

    data["instruction"] = instruction_template.format(
        documents=data["document"],
        claim=data["claim"]
    )

    data["messages"] = [
        {
            "role": "user",
            "content": data["instruction"]
        }
    ]

    old_img_path = data["document_image"]
    img_filename = os.path.basename(old_img_path)

    if is_train:
        new_img_path = f"data/train_document_images/{img_filename}"
    else:
        new_img_path = f"data/val_document_images/{img_filename}"

    data["images"] = [new_img_path]

    return data

def copy_image_directory(source_dir, target_dir):
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
        print(f"Create a directory: {target_dir}")

    if os.path.exists(source_dir):
        for filename in os.listdir(source_dir):
            source_file = os.path.join(source_dir, filename)
            target_file = os.path.join(target_dir, filename)

            if os.path.isfile(source_file):
                shutil.copy2(source_file, target_file)

        print(f"Copied image from {source_dir} to {target_dir}")
    else:
        print(f"Warning: Source directory does not exist {source_dir}")

def main():
    if not os.path.exists("data"):
        os.makedirs("data")
        print("Create a data directory")

    copy_image_directory(source_train_img_dir, target_train_img_dir)
    copy_image_directory(source_val_img_dir, target_val_img_dir)

    train_processed_data = []
    with open(input_file1, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                processed_item = process_line(line, is_train=True)
                train_processed_data.append(processed_item)

    with open(output_file1, 'w', encoding='utf-8') as f:
        json.dump(train_processed_data, f, ensure_ascii=False, indent=2)

    print(f"Processed {len(train_processed_data)} training data and saved to {output_file1}")

    val_processed_data = []
    with open(input_file2, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                processed_item = process_line(line, is_train=False)
                val_processed_data.append(processed_item)

    with open(output_file2, 'w', encoding='utf-8') as f:
        json.dump(val_processed_data, f, ensure_ascii=False, indent=2)

    print(f"Processed {len(val_processed_data)} test data and saved to {output_file2}")

if __name__ == "__main__":
    main()
