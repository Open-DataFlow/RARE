import os
from dataclasses import dataclass, field, asdict
from typing import Optional
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
from datasets import Dataset
import transformers
import trl
import json


@dataclass
class TrainingConfig:
    model_name_or_path: str = field(default="meta-llama/Llama-3.1-8B-Instruct")
    block_size: int = field(default=32768)
    # wandb_project: Optional[str] = field(default="RARE")
    # wandb_entity: Optional[str] = field(default="1111")
    train_file_path: Optional[str] = field(
        default="data/train_pubmed.json"
    )
    dagger: bool = field(default=False)

    # def __post_init__(self):
    #     os.environ["WANDB_PROJECT"] = self.wandb_project
    #     os.environ["WANDB_ENTITY"] = self.wandb_entity


def train():
    # parsing input
    parser = transformers.HfArgumentParser((TrainingConfig, trl.SFTConfig))
    config, args = parser.parse_args_into_dataclasses()
    log_config = {**asdict(config), **asdict(args)}
    logging.info(f"Training config: {log_config}")

    # loading model
    kwargs = {}
    if "70B" in config.model_name_or_path:
        kwargs = {
            "device_map": "auto",
            "torch_dtype": "auto",
            "attn_implementation": "flash_attention_2",
            "use_cache": False,
        }
        model = transformers.AutoModelForCausalLM.from_pretrained(
            config.model_name_or_path, **kwargs
        )
    else:
        model = transformers.AutoModelForCausalLM.from_pretrained(config.model_name_or_path)

    # Load and process your custom dataset
    with open(config.train_file_path, "r") as f:
        data = json.load(f)

    # Prepare the dataset in the format expected by the trainer
    # Combine instruction and output into a single text field
    processed_data = []

    for item in data:
        # Check if we need to handle specific formats or templates
        if "Qwen" in config.model_name_or_path:
            # Format for Qwen models
            text = f"<|im_start|>user\n{item['instruction']}<|im_end|>\n<|im_start|>assistant\n{item['predict']}<|im_end|>"
        elif "Llama" in config.model_name_or_path:
            # Format for Llama models
            text = f"<|start_header_id|>user<|end_header_id|>\n{item['instruction']}\n<|start_header_id|>assistant<|end_header_id|>\n\n{item['predict']}"
        elif "Mistral" in config.model_name_or_path:
            text = f"[INST] {item['instruction']}[/INST] {item['predict']}"
        else:
            # Generic format for other models
            text = f"USER: {item['instruction']}\nASSISTANT: {item['predict']}"
        processed_data.append({"text": text})

    # Create a Hugging Face dataset from the processed data
    dataset = Dataset.from_list(processed_data)

    train_dataset = dataset

    # If you want a  validation set, uncomment these lines:
    # train_size = int(0.95 * len(dataset))
    # train_dataset = dataset.select(range(train_size))
    # eval_dataset = dataset.select(range(train_size, len(dataset)))

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        config.model_name_or_path, use_fast=True
    )
    if "Llama" in config.model_name_or_path:
        instruction_template = "<|start_header_id|>user<|end_header_id|>"
        response_template = "<|start_header_id|>assistant<|end_header_id|>\n\n"
        # Use a token that is never used
        tokenizer.pad_token = "<|reserved_special_token_5|>"
    elif "Qwen" in config.model_name_or_path:
        instruction_template = "<|im_start|>user"
        response_template = "<|im_start|>assistant\n"
        # Use a token that is never used
        tokenizer.pad_token = "<|fim_pad|>"
    elif "Mistral" in config.model_name_or_path:
        instruction_template = tokenizer.encode("[INST]", add_special_tokens=False)
        response_template_tokens = tokenizer.encode("[/INST]", add_special_tokens=False)
        tokenizer.pad_token = "[control_766]"

    collator = None
    if "Mistral" in config.model_name_or_path:
        response_template_tokens = tokenizer.encode(
            "\n[/INST]", add_special_tokens=False
        )[2:]
        collator = trl.DataCollatorForCompletionOnlyLM(
            response_template=response_template_tokens, tokenizer=tokenizer, mlm=False
        )
    else:
        collator = trl.DataCollatorForCompletionOnlyLM(
            instruction_template=instruction_template,
            response_template=response_template,
            tokenizer=tokenizer,
            mlm=False,
        )

    args.dataset_text_field = "text"
    args.max_seq_length = config.block_size
    trainer = trl.SFTTrainer(
        model,
        train_dataset=train_dataset,
        # Use the whole dataset for training, no separate eval dataset
        eval_dataset=None,
        args=args,
        data_collator=collator,
    )

    trainer.train()
    trainer.save_model(output_dir=args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    trainer.accelerator.wait_for_everyone()


if __name__ == "__main__":
    train()
