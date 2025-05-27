import json
import gc
import os
import re
import argparse
import torch
from vllm import LLM, SamplingParams
from vllm.distributed.parallel_state import (
    destroy_distributed_environment,
    destroy_model_parallel,
)


def clean_up():
    """Clean up resources - support both NPU and GPU"""
    destroy_model_parallel()
    destroy_distributed_environment()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif hasattr(torch, 'npu') and torch.npu.is_available():
        torch.npu.empty_cache()


def format_prompt(instruction, template):
    """Format instruction based on model template"""
    if template.lower() == "qwen":
        return f"<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n"
    elif template.lower() == "llama":
        return f"<|start_header_id|>user<|end_header_id|>\n{instruction}\n<|start_header_id|>assistant<|end_header_id|>\n\n"
    elif template.lower() == "mistral":
        return f"[INST] {instruction}[/INST] "
    elif template.lower() == "deepseek":
        return f"<｜User｜>{instruction}<｜Assistant｜>"
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
    elif template.lower() == "deepseek":
        return ["<｜User｜>"]
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
            except json.JSONDecodeError:
                with open(dataset_path, "r", encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            data.append(json.loads(line))

        return data
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return []


def extract_option(pred):
    """Extract answer option from model output"""
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

    # 3. Only keep first character if it's a valid option
    if pred and pred[0] in "ABCDE":
        return pred[0]
    return pred


def check_answer(prediction, correct_answer):
    """Check if the prediction matches the correct answer"""
    extracted_pred = extract_option(prediction)
    extracted_answer = extract_option(correct_answer)
    return extracted_pred == extracted_answer


def resampling_inference(
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
    prediction_key: str = "qwq_sft",
    max_attempts: int = 8,
    output_path: str = None,
):
    """
    Perform resampling inference: retry incorrect answers up to max_attempts times
    """
    print(f"Loading model: {model_name_or_path}")
    print(f"Using template: {template}")

    # Load dataset
    dataset = load_dataset(dataset_path)
    if not dataset:
        print(f"Failed to load dataset from {dataset_path} or dataset is empty.")
        return

    print(f"Loaded {len(dataset)} examples from dataset.")

    # Initialize LLM
    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        max_tokens=max_new_tokens,
        repetition_penalty=repetition_penalty,
        stop=get_stop_tokens(template),
    )

    print(f"Initializing LLM with tensor_parallel_size={tensor_parallel_size}")
    llm = LLM(
        model=model_name_or_path,
        tensor_parallel_size=tensor_parallel_size,
        distributed_executor_backend="mp",
        max_model_len=max_model_len,
        trust_remote_code=True,
    )

    # Initialize tracking variables
    items_to_resample = list(range(len(dataset)))
    attempt_counts = [0] * len(dataset)
    correct_items = set()
    
    # Store original results
    original_results = [None] * len(dataset)
    
    # Track metrics
    total_correct = 0
    initial_correct = 0

    # Resampling loop - continue until all items correct or max attempts reached
    while items_to_resample and max(attempt_counts) < max_attempts:
        current_batch_indices = items_to_resample.copy()
        items_to_resample = []
        
        # Prepare prompts for current batch
        prompts = []
        for idx in current_batch_indices:
            item = dataset[idx]
            instruction = item.get(
                "instruction", item.get("input", item.get("prompt", item.get("query", "")))
            )
            if not instruction:
                print(f"Warning: Couldn't find instruction in item: {item}")
                continue
            
            formatted_prompt = format_prompt(instruction, template)
            prompts.append(formatted_prompt)
            attempt_counts[idx] += 1
        
        if not prompts:
            break
            
        print(f"Starting batch generation for {len(prompts)} items...")
        outputs = llm.generate(prompts, sampling_params)
        
        # Process outputs and determine which items need resampling
        for batch_idx, (idx, output) in enumerate(zip(current_batch_indices, outputs)):
            item = dataset[idx]
            generated_text = output.outputs[0].text.strip()
            
            # Save first attempt result
            if attempt_counts[idx] == 1:
                original_results[idx] = generated_text
                item[f"{prediction_key}_original"] = generated_text
            
            # Check if answer is correct
            is_correct = False
            if "output" in item:
                is_correct = check_answer(generated_text, item["output"])
                
                # Track first attempt accuracy
                if attempt_counts[idx] == 1 and is_correct:
                    initial_correct += 1
            
            # Update item with latest prediction
            item[prediction_key] = generated_text
            
            if is_correct:
                correct_items.add(idx)
                total_correct += 1
                print(f"Item {idx} correct on attempt {attempt_counts[idx]}")
            else:
                # If not correct and under max attempts, add to resample list
                if attempt_counts[idx] < max_attempts:
                    items_to_resample.append(idx)
                    
                    # For items reaching max attempts without success, restore original result
                    if attempt_counts[idx] == max_attempts - 1:
                        print(f"Item {idx} failed after {max_attempts} attempts, reverting to original result")
    
    # For items that reached max attempts without success, restore original result
    for idx in range(len(dataset)):
        if idx not in correct_items and attempt_counts[idx] >= max_attempts:
            dataset[idx][prediction_key] = original_results[idx]
    
    # Calculate accuracy
    final_correct = sum(1 for idx in range(len(dataset)) 
                       if "output" in dataset[idx] and 
                       check_answer(dataset[idx][prediction_key], dataset[idx]["output"]))
    
    print("\n" + "="*50)
    print(f"Resampling results:")
    print(f"Total examples: {len(dataset)}")
    print(f"Initial accuracy: {initial_correct/len(dataset)*100:.2f}% ({initial_correct}/{len(dataset)})")
    print(f"Final accuracy: {final_correct/len(dataset)*100:.2f}% ({final_correct}/{len(dataset)})")
    print(f"Improvement: {(final_correct-initial_correct)/len(dataset)*100:.2f}%")
    
    # Attempt distribution
    attempts_hist = {}
    for count in attempt_counts:
        attempts_hist[count] = attempts_hist.get(count, 0) + 1
    
    print("\nAttempt distribution:")
    for attempt, count in sorted(attempts_hist.items()):
        print(f"  {attempt} attempt(s): {count} example(s)")
    print("="*50)
    
    # Determine output path
    if output_path is None:
        filename, ext = os.path.splitext(dataset_path)
        output_path = f"{filename}_resampled{ext}"
    
    # Save results
    try:
        file_extension = os.path.splitext(output_path)[1].lower()
        if file_extension == ".json":
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(dataset, f, ensure_ascii=False, indent=2)
        else:  # Use JSONL format by default
            with open(output_path, "w", encoding="utf-8") as f:
                for item in dataset:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")

        print(f"Results saved to {output_path}")
    except Exception as e:
        print(f"Error saving results: {e}")
    
    if output_path:
        base_path, ext = os.path.splitext(output_path)
        true_output_path = f"{base_path}_true{ext}"
    else:
        base_path, ext = os.path.splitext(dataset_path)
        true_output_path = f"{base_path}_resampled_true{ext}"

    # 筛选正确答案
    correct_data = []
    for item in dataset:
        if "output" in item and check_answer(item[prediction_key], item["output"]):
            correct_data.append(item)

    # 保存正确答案数据集
    try:
        if os.path.splitext(true_output_path)[1].lower() == ".json":
            with open(true_output_path, "w", encoding="utf-8") as f:
                json.dump(correct_data, f, ensure_ascii=False, indent=2)
        else:
            with open(true_output_path, "w", encoding="utf-8") as f:
                for item in correct_data:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"Correct results saved to {true_output_path}")
        print(f"Correct samples count: {len(correct_data)}/{len(dataset)} "
              f"({len(correct_data)/len(dataset)*100:.2f}%)")
    except Exception as e:
        print(f"Error saving correct results: {e}")

    # Clean up resources
    del llm
    clean_up()


def parse_args():
    parser = argparse.ArgumentParser(description="vLLM Resampling Inference")
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
        choices=["qwen", "llama", "mistral", "deepseek"],
        help="Prompt template to use",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.95, help="Sampling temperature"
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
        default=10240,
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
        default=8,
        help="Tensor parallel size for distributed inference",
    )
    parser.add_argument(
        "--max_model_len", 
        type=int, 
        default=20480, 
        help="Maximum model sequence length"
    )
    parser.add_argument(
        "--prediction_key",
        type=str,
        default="qwq_sft",
        help="Key to use when storing model predictions in the dataset",
    )
    parser.add_argument(
        "--max_attempts",
        type=int,
        default=8,
        help="Maximum number of resampling attempts for each incorrect answer",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Path to save the resampled dataset (defaults to dataset_path with '_resampled' suffix)",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    resampling_inference(
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
        max_attempts=args.max_attempts,
        output_path=args.output_path,
    )