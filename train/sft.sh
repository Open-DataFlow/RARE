# Reference Running: bash train/sft.sh
# {'train_runtime': 5268.8407, 'train_samples_per_second': 0.949, 'train_steps_per_second': 0.119, 'train_loss': 0.1172730620391667, 'epoch': 5.0}
# export HCCL_CONNECT_TIMEOUT=2000 # for NPU
uid="$(date +%Y%m%d_%H%M%S)"
base_model="meta-llama/Llama-3.1-8B-Instruct"
lr=1e-5
min_lr=0
epochs=5
weight_decay=1e-4 # -> the same training pipe as slurm_training
micro_batch_size=1 # -> batch_size will be 16 if 16 gpus
gradient_accumulation_steps=8 # requires more GPU memory
max_steps=-1
gpu_count=8
push_to_hub=false

torchrun --nproc-per-node ${gpu_count} --master_port 12345 \
    train/sft.py \
    --block_size=32768 \
    --train_file_path="data/train_pubmed" \
    --per_device_train_batch_size=${micro_batch_size} \
    --per_device_eval_batch_size=${micro_batch_size} \
    --gradient_accumulation_steps=${gradient_accumulation_steps} \
    --num_train_epochs=${epochs} \
    --model_name_or_path=${base_model} \
    --warmup_ratio=0.05 \
    --fsdp="full_shard auto_wrap" \
    --fsdp_config="train/fsdp_config_llama.json" \
    --bf16=True \
    --eval_strategy="no" \
    --logging_steps=1 \
    --save_strategy="no" \
    --lr_scheduler_type="cosine" \
    --learning_rate=${lr} \
    --weight_decay=${weight_decay} \
    --adam_beta1=0.9 \
    --adam_beta2=0.95 \
    --output_dir="saves/pubmed-llama" \
    --push_to_hub=${push_to_hub} \
    --save_only_model=True \
    --gradient_checkpointing=True \
    --report_to="none"  # remove this parameter to use wandb
    # --accelerator_config='{"gradient_accumulation_kwargs": {"sync_each_batch": true}}'
