# preprocess data
# download
huggingface-cli download whalezzz/M2RAG --repo-type dataset --local-dir process --include "fact_verify/*"
python process/process_mmrait.py

modelscope download \
    --model Qwen/Qwen2.5-VL-32B-Instruct \
    --local_dir saves/Qwen2.5-VL-32B-Instruct

# distill
python inference/vllm_infer_mm.py \
    --model_name_or_path saves/Qwen2.5-VL-32B-Instruct \
    --dataset_path data/train_mmrait.json

# select only true
python process/select_true.py data/train_mmrait.json --mm

# train
accelerate launch \
    --config_file train/accelerate_config_mm.yaml train/train.py train/training_args_mm.yaml

# Transfer checkpoint
python saves/mmrait-qwenvl/zero_to_fp32.py saves/mmrait-qwenvl/ --safe_serialization

# inference
python inference/vllm_infer_mm.py \
    --model_name_or_path saves/mmrait-qwenvl \
    --dataset_path data/test_mmrait.json \
    --prediction_key llm_predict_rare_qwen2vl \
    --tensor_parallel_size 4

# eval
python eval/eval.py \
    --file data/test_mmarit.json \
    --prediction_key llm_predict_rare_qwen2vl