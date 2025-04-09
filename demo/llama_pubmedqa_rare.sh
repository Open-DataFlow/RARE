# preprocess data
# download
huggingface-cli download --repo-type dataset --resume-download yuhkalhic/rare_share --local-dir process/rare_share
unzip process/rare_share/system=planner_addret,dataset=all_dev,debug=False.zip -d process/rare_share/system=planner_addret,dataset=all_dev,debug=False
unzip process/rare_share/system=planner_addret,dataset=all_train,debug=False.zip -d process/rare_share/system=planner_addret,dataset=all_train,debug=False
unzip process/rare_share/system=planner_addret,dataset=all_test,debug=False.zip -d process/rare_share/system=planner_addret,dataset=all_test,debug=False
python process/process_pubmed.py

modelscope download --model Qwen/QwQ-32B --local_dir saves/QwQ-32B

# distill
python inference/vllm_infer_text.py \
    --model_name_or_path saves/QwQ-32B \
    --dataset_path data/train_pubmed.json \
    --template qwen

# train
# To achieve the best effect, please use 8 or more A100 as much as possible.
torchrun --nproc-per-node 8 --master_port 12345 \
    train/sft.py \
    --block_size=32768 \
    --per_device_train_batch_size=1 \
    --per_device_eval_batch_size=1 \
    --gradient_accumulation_steps=8 \
    --num_train_epochs=5 \
    --train_file_path="data/train_pubmed.json" \
    --model_name_or_path="meta-llama/Llama-3.1-8B-Instruct" \
    --warmup_ratio=0.05 \
    --fsdp="full_shard auto_wrap" \
    --fsdp_config="train/fsdp_config_llama.json" \
    --bf16=True \
    --eval_strategy="no" \
    --logging_steps=1 \
    --save_strategy="no" \
    --lr_scheduler_type="cosine" \
    --learning_rate=1e-5 \
    --weight_decay=1e-4 \
    --adam_beta1=0.9 \
    --adam_beta2=0.95 \
    --output_dir="saves/pubmed-llama" \
    --push_to_hub=false \
    --save_only_model=True \
    --gradient_checkpointing=True \
    --report_to="none" 

# inference
python inference/vllm_infer_text.py  \
    --model_name_or_path saves/pubmed-llama \
    --dataset_path data/test_pubmed.json \
    --template llama \
    --prediction_key llm_predict_rare_llama \
    --tensor_parallel_size 8

# eval
python eval/eval.py \
    --file data/test_pubmed.json \
    --prediction_key llm_predict_rare_llama