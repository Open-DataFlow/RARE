# Model name or path specifying the pretrained model to use
model_name_or_path="meta-llama/Llama-3.1-8B-Instruct"

# Variable to select which model type to use, making it easy to switch models
temp=llama3
# Other example models you can switch to:
# temp=qwen
# temp=mistral

# Dataset name indicating which dataset is being used, for more details you can check RL_KTO/data/dataset_info.json
dataset=covert_kto

# Output directory for saving results
OUTPUT_DIR=saves/RL_KTO/${temp}_${dataset}

llamafactory-cli train \
    --stage kto \
    --do_train True \
    --model_name_or_path ${model_path} \
    --preprocessing_num_workers 16 \
    --finetuning_type full \
    --template ${temp} \
    --flash_attn auto\
    --dataset_dir RL_KTO/data \
    --dataset ${dataset} \
    --cutoff_len ${len} \
    --learning_rate 5e-06 \
    --num_train_epochs 3 \
    --max_samples 100000 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --lr_scheduler_type cosine \
    --max_grad_norm 1.0 \
    --logging_steps 1 \
    --save_steps 100 \
    --warmup_steps 0 \
    --packing False \
    --report_to none \
    --output_dir ${OUTPUT_DIR} \
    --bf16 True \
    --plot_loss True \
    --trust_remote_code True \
    --ddp_timeout 180000000 \
    --include_num_input_tokens_seen True \
    --optim adamw_torch \
    --pref_beta 0.1 \
    --pref_ftx 0 \
    --pref_loss sigmoid \
    --deepspeed RL_KTO/ds_z3_offload_config.json \
    --gradient_checkpointing True 