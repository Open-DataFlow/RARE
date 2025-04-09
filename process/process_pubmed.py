import argparse
import json
import os
from os.path import join as pjoin
import re
from typing import List, Dict, Tuple
import logging
import random

import torch
import numpy as np
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm
from qdrant_client import QdrantClient
from qdrant_client.http import models

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

template = """You are a professional medical expert to answer the # Question. Please first think step-by-step using the # Retrieved Documents and then answer the question. Your responses will be used for research purposes only, so please have a definite answer.

The format should be like:
<think>
...
</think>
<answer>A/B/C/D</answer> (only one option can be chosen)

# Retrieved Documents
{documents}

# Question
{question}"""


class DocumentProcessor:
    def __init__(self, model_name: str = "BAAI/llm-embedder"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info(f"Using device: {self.device}")

        logging.info(f"Loading model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

        self.qdrant = QdrantClient(":memory:")

    def extract_document_blocks(self, text: str) -> List[Dict[str, str]]:
        blocks = []
        lines = []
        current_source = None

        for line in text.split("\n"):
            if line.startswith("## source:"):
                if current_source and lines:
                    blocks.append(
                        {"source": current_source, "content": "\n".join(lines)}
                    )
                    lines = []
                current_source = line
                lines = [line]
            elif current_source is not None:
                lines.append(line)

        if current_source and lines:
            blocks.append({"source": current_source, "content": "\n".join(lines)})

        return blocks

    def get_embedding(self, text: str) -> np.ndarray:
        with torch.no_grad():
            inputs = self.tokenizer(
                text, return_tensors="pt", max_length=512, truncation=True, padding=True
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            outputs = self.model(**inputs)
            embeddings = outputs.last_hidden_state.mean(dim=1)
            return embeddings.cpu().numpy()

    def rank_blocks(self, blocks: List[Dict[str, str]], query: str) -> List[str]:
        if not blocks:
            return []

        block_contents = [block["content"] for block in blocks]
        doc_embeddings = np.vstack(
            [self.get_embedding(content) for content in block_contents]
        )
        query_embedding = self.get_embedding(query)

        vector_size = doc_embeddings.shape[1]

        collection_name = "temp_collection"
        self.qdrant.recreate_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(
                size=vector_size, distance=models.Distance.COSINE
            ),
        )

        self.qdrant.upload_points(
            collection_name=collection_name,
            points=[
                models.PointStruct(
                    id=idx, vector=embedding.tolist(), payload={"content": content}
                )
                for idx, (embedding, content) in enumerate(
                    zip(doc_embeddings, block_contents)
                )
            ],
        )

        k = min(len(blocks), 3)

        results = self.qdrant.search(
            collection_name=collection_name,
            query_vector=query_embedding[0].tolist(),
            limit=k,
        )
        relevant_blocks = [hit.payload["content"] for hit in results]

        self.qdrant.delete_collection(collection_name)

        return relevant_blocks

    def process_example(self, content: str, question: str) -> Dict:
        blocks = self.extract_document_blocks(content)
        if not blocks:
            documents_str = ""
        else:
            relevant_blocks = self.rank_blocks(blocks, question)
            documents_str = "\n".join(relevant_blocks)

        instruction = template.format(documents=documents_str, question=question)

        return {
            "output": "",
            "documents": documents_str,
            "question": question,
            "instruction": instruction,
        }


def load_dataset(dataset_name: str) -> List[Dict]:
    plan_name = f"system=planner_addret,dataset={dataset_name},debug=False"
    output_all_path = pjoin("process", "rare_share", plan_name, "output_all.json")
    logging.info(f"Loading data from {output_all_path}")

    try:
        with open(output_all_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logging.error(f"Error loading dataset {dataset_name}: {str(e)}")
        return []


def process_dataset(processor, dataset_items, dataset_type, target_names):
    filtered_data = {name: [] for name in target_names}

    logging.info(f"Processing {len(dataset_items)} items for {dataset_type} dataset")

    for item in tqdm(dataset_items):
        try:
            if "pred" not in item or "doc_path" not in item["pred"]:
                logging.warning(
                    f"Missing pred or doc_path in item: {item.get('id', 'NO_ID')}"
                )
                continue

            item_name = item.get("name", "")
            if item_name not in target_names:
                continue

            # 使用更新后的doc_path
            doc_path = item["pred"]["doc_path"]

            doc_path = doc_path.replace("\\", "/")

            # 确保文件路径存在
            if not os.path.exists(doc_path):
                # 尝试从工作目录解析相对路径
                if doc_path.startswith("process/"):
                    # 移除前缀"process/"以使其相对于当前目录
                    alternative_path = doc_path[8:]
                    if os.path.exists(alternative_path):
                        doc_path = alternative_path
                    else:
                        logging.warning(f"Document file not found: {doc_path}")
                        continue
            
            try:
                with open(doc_path, "r", encoding="utf-8") as f:
                    content = f.read()
            except FileNotFoundError:
                logging.warning(f"Document file not found: {doc_path}")
                continue
            except Exception as e:
                logging.warning(f"Error reading document {doc_path}: {str(e)}")
                continue

            question = item.get("question", "")
            processed_item = processor.process_example(content, question)
            processed_item["output"] = f"<answer>{item.get('gold', '')}</answer>"
            processed_item["name"] = item_name
            processed_item["id"] = item.get("id", "")

            filtered_data[item_name].append(processed_item)

        except Exception as e:
            logging.error(f"Error processing item {item.get('id', 'NO_ID')}: {str(e)}")
            continue

    return filtered_data


def save_results(filtered_data, dataset_type, target_names):
    data_dir = os.path.join(".", "data")
    os.makedirs(data_dir, exist_ok=True)

    for name in target_names:
        items = filtered_data[name]
        if items:
            output_filename = os.path.join(data_dir, f"{dataset_type}_pubmed.json")
            logging.info(f"Saving {len(items)} {name} items to {output_filename}")

            with open(output_filename, "w", encoding="utf-8") as f:
                json.dump(items, f, ensure_ascii=False, indent=2)
        else:
            logging.warning(
                f"No {name} items found, skipping file creation for {dataset_type}_pubmed.json"
            )

    total_processed = sum(len(items) for items in filtered_data.values())
    logging.info(
        f"Successfully processed and saved {total_processed} examples for {dataset_type} dataset"
    )


def main():
    processor = DocumentProcessor()
    target_names = ["pubmedqa"]

    train_datasets = ["all_train", "all_dev"]
    train_items = []
    for dataset in train_datasets:
        train_items.extend(load_dataset(dataset))

    train_filtered_data = process_dataset(processor, train_items, "train", target_names)
    save_results(train_filtered_data, "train", target_names)

    test_items = load_dataset("all_test")
    test_filtered_data = process_dataset(processor, test_items, "test", target_names)
    save_results(test_filtered_data, "test", target_names)

    logging.info("All processing completed successfully!")


if __name__ == "__main__":
    main()