import json
import random
import os
import math
from datasets import load_dataset

output_train_path = "data/train_finfact.json"
output_test_path = "data/test_finfact.json"

# template
template = """You are a professional financial expert in fact-checking, skilled in analyzing the accuracy of # Statement. Please first think step-by-step using the # Retrieved Documents and then check # Statement by using your own knowledge. Your responses will be used for research purposes only, so please have a definite answer.

You should respond in the format:
<think>
...
</think>
<answer>A/B/C</answer> (only one option can be chosen)

# Retrieved Documents
{documents}

# Statement
{claim}\nA. true\nB. false\nC. NEI"""

ds = load_dataset("amanrangapur/Fin-Fact")

data = []
for item in ds["train"]:
    data_item = {}
    for key in item:
        data_item[key] = item[key]


for item in data:
    if item["label"] == "true":
        item["output"] = "<answer>A</answer>"
    elif item["label"] == "false":
        item["output"] = "<answer>B</answer>"
    elif item["label"] == "NEI":
        item["output"] = "<answer>C</answer>"
    
    documents_text = ""
    if "evidence" in item and isinstance(item["evidence"], list):
        for evidence in item["evidence"]:
                
            if evidence["sentence"] is None or (isinstance(evidence["sentence"], float) and math.isnan(evidence["sentence"])):
                continue
                
            if isinstance(evidence["sentence"], str):
                documents_text += evidence["sentence"] + "\n"
            else:
                try:
                    documents_text += str(evidence["sentence"]) + "\n"
                except:
                    pass
    
    item["documents"] = documents_text.strip()
    
    item["instruction"] = template.format(
        documents=item["documents"],
        claim=item["claim"]
    )

# Shuffle the data randomly
random.seed(42)
random.shuffle(data)

# Split into training set (80%) and test set (20%)
split_idx = int(len(data) * 0.8)
train_data = data[:split_idx]
test_data = data[split_idx:]

# Save as JSON files
with open(output_train_path, "w", encoding="utf-8") as f:
    json.dump(train_data, f, ensure_ascii=False, indent=2)

with open(output_test_path, "w", encoding="utf-8") as f:
    json.dump(test_data, f, ensure_ascii=False, indent=2)

print(f"Processing complete!")
print(f"Training set size: {len(train_data)} items, saved to {output_train_path}")
print(f"Test set size: {len(test_data)} items, saved to {output_test_path}")