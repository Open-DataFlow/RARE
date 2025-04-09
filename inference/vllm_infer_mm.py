import json
import gc
import os
import argparse
import torch
from vllm import LLM, SamplingParams
from vllm.distributed.parallel_state import (
    destroy_distributed_environment,
    destroy_model_parallel,
)

from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info


# def clean_up():
#     """only for npu"""
#     destroy_model_parallel()
#     destroy_distributed_environment()
#     gc.collect()
#     torch.npu.empty_cache()


def format_prompt(instruction, template):
    """Format instruction based on model template"""
    if template.lower() == "qwen":
        return f"<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n"
    elif template.lower() == "llama":
        return f"<|start_header_id|>user<|end_header_id|>\n{instruction}\n<|start_header_id|>assistant<|end_header_id|>\n\n"
    elif template.lower() == "mistral":
        return f"[INST] {instruction}[/INST] "
    else:
        # Generic fallback
        return f"USER: {instruction}\nASSISTANT: "


def get_stop_tokens(template):
    """Get stop tokens based on template"""
    if template.lower() == "qwen":
        return ["<|im_end|>"]
    elif template.lower() == "llama":
        return ["<|end_header_id|>"]
    elif template.lower() == "mistral":
        return ["[INST]"]
    else:
        return ["USER:"]


def load_dataset(dataset_path):
    """Load dataset from JSON or JSONL file"""
    data = []
    file_extension = os.path.splitext(dataset_path)[1].lower()

    try:
        if file_extension == ".json":
            with open(dataset_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                # Handle both list and dict formats
                if isinstance(data, dict):
                    data = [data]
        elif file_extension == ".jsonl":
            with open(dataset_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        data.append(json.loads(line))
        else:
            # Try both formats if extension doesn't match
            try:
                with open(dataset_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    # Handle both list and dict formats
                    if isinstance(data, dict):
                        data = [data]
            except json.JSONDecodeError:
                with open(dataset_path, "r", encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            data.append(json.loads(line))

        return data
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return []


def prepare_multimodal_inputs(item):
    """Prepare multimodal inputs from dataset item without modifying original format"""
    # Create a deep copy to avoid modifying the original data
    messages = []
    for msg in item.get("messages", []):
        # Create a copy of each message
        messages.append(dict(msg))

    images = item.get("images", [])

    if not messages and "instruction" in item:
        messages = [{"role": "user", "content": item["instruction"]}]

    if images and messages:
        temp_messages = []
        for msg in messages:
            msg_copy = dict(msg)

            # Only modify user messages that contain text content
            if msg_copy["role"] == "user" and isinstance(msg_copy["content"], str):
                new_content = []

                # Add images
                for image_path in images:
                    new_content.append(
                        {
                            "type": "image",
                            "image": image_path,
                            "min_pixels": 224 * 224,
                            "max_pixels": 1280 * 28 * 28,
                        }
                    )

                # Add the original text
                new_content.append({"type": "text", "text": msg_copy["content"]})

                # Update temporary message
                msg_copy["content"] = new_content

            temp_messages.append(msg_copy)

        return temp_messages

    return messages


def vllm_infer(
    model_name_or_path: str,
    dataset_path: str,
    template: str = "qwen",
    temperature: float = 0.95,
    top_p: float = 0.7,
    top_k: int = 50,
    max_new_tokens: int = 8192,
    repetition_penalty: float = 1.0,
    tensor_parallel_size: int = 4,
    max_model_len: int = 10240,
    prediction_key: str = "predict",
):
    """
    With multimodal support always enabled: python vllm_infer_mm.py --model_name_or_path Qwen2.5-VL-32B-Instruct --dataset_path data/test_mmrait.json
    """
    print(f"Loading model: {model_name_or_path}")
    print(f"Using template: {template}")
    print(f"Multimodal mode: Enabled")

    # Load dataset
    dataset = load_dataset(dataset_path)
    if not dataset:
        print(f"Failed to load dataset from {dataset_path} or dataset is empty.")
        return

    print(f"Loaded {len(dataset)} examples from dataset.")

    # Load the processor for multimodal models
    processor = AutoProcessor.from_pretrained(model_name_or_path)
    print(f"Loaded AutoProcessor for multimodal model")

    prompts = []
    original_data = []

    for item in dataset:
        # Handle multimodal input
        messages = prepare_multimodal_inputs(item)

        if not messages:
            print(f"Warning: No valid messages found in item: {item}")
            continue

        # Create a temporary copy for processing
        temp_messages = messages.copy()

        # Process the prompt using the model's processor
        prompt = processor.apply_chat_template(
            temp_messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        # Process vision information
        image_inputs, video_inputs, video_kwargs = process_vision_info(
            temp_messages, return_video_kwargs=True
        )

        # Prepare multimodal data
        mm_data = {}
        if image_inputs:
            mm_data["image"] = image_inputs

        # Prepare LLM input
        llm_input = {
            "prompt": prompt,
            "multi_modal_data": mm_data,
        }

        prompts.append(llm_input)
        original_data.append(item)

    if not prompts:
        print("No valid prompts found in dataset. Exiting.")
        return

    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        max_tokens=max_new_tokens,
        repetition_penalty=repetition_penalty,
    )

    print(f"Initializing LLM with tensor_parallel_size={tensor_parallel_size}")

    llm_kwargs = {
        "model": model_name_or_path,
        "tensor_parallel_size": tensor_parallel_size,
        "distributed_executor_backend": "mp",
        "max_model_len": max_model_len,
        "trust_remote_code": True,
        "limit_mm_per_prompt": {"image": 10},
    }

    llm = LLM(**llm_kwargs)

    print("Starting batch generation...")
    outputs = llm.generate(prompts, sampling_params)

    print(f"Using prediction key: {prediction_key}")

    for i, output in enumerate(outputs):
        generated_text = output.outputs[0].text.strip()

        # Add assistant response as a new message if messages array exists
        if "messages" in original_data[i]:
            original_data[i]["messages"].append(
                {"role": "assistant", "content": generated_text}
            )

        # Also store in the prediction key for compatibility
        original_data[i][prediction_key] = generated_text

    file_extension = os.path.splitext(dataset_path)[1].lower()

    # Create backup of original file
    backup_path = dataset_path + ".bak"
    if not os.path.exists(backup_path):
        try:
            with open(dataset_path, "rb") as src, open(backup_path, "wb") as dst:
                dst.write(src.read())
            print(f"Created backup of original dataset at {backup_path}")
        except Exception as e:
            print(f"Warning: Failed to create backup: {e}")

    try:
        if file_extension == ".json":
            with open(dataset_path, "w", encoding="utf-8") as f:
                json.dump(original_data, f, ensure_ascii=False, indent=2)
        else:  # Use JSONL format by default
            with open(dataset_path, "w", encoding="utf-8") as f:
                for item in original_data:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")

        print("*" * 70)
        print(
            f"{len(original_data)} records updated with predictions using key '{prediction_key}'"
        )
        print(f"Updated dataset saved to {dataset_path}")
        print("*" * 70)
    except Exception as e:
        print(f"Error saving results: {e}")
        print("Please check the backup file if needed.")

    del llm
    # clean_up()


def parse_args():
    parser = argparse.ArgumentParser(
        description="vLLM Inference for NPU with Multimodal Support"
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        required=True,
        help="Path to dataset file (JSON or JSONL)",
    )
    parser.add_argument(
        "--template",
        type=str,
        default="qwen",
        choices=["qwen", "llama", "mistral"],
        help="Prompt template to use",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.6, help="Sampling temperature"
    )
    parser.add_argument(
        "--top_p", type=float, default=0.7, help="Top-p sampling parameter"
    )
    parser.add_argument(
        "--top_k", type=int, default=50, help="Top-k sampling parameter"
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=8192,
        help="Maximum number of tokens to generate",
    )
    parser.add_argument(
        "--repetition_penalty",
        type=float,
        default=1.0,
        help="Repetition penalty parameter",
    )
    parser.add_argument(
        "--tensor_parallel_size",
        type=int,
        default=4,
        help="Tensor parallel size for distributed inference",
    )
    parser.add_argument(
        "--max_model_len", type=int, default=10240, help="Maximum model sequence length"
    )
    parser.add_argument(
        "--prediction_key",
        type=str,
        default="predict",
        help="Key to use when storing model predictions in the dataset",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    vllm_infer(
        model_name_or_path=args.model_name_or_path,
        dataset_path=args.dataset_path,
        template=args.template,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_new_tokens=args.max_new_tokens,
        repetition_penalty=args.repetition_penalty,
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_len,
        prediction_key=args.prediction_key,
    )
