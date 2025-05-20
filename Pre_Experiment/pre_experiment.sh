# Run pre-experiment with parameters:
# --retrieval_ratio: Controls how much retrieval content to use (ratio/4 of document):
#   0 = None
#   1 = 1/4
#   2 = 1/2
#   3 = 3/4
#   4 = Full content

#For pubhealth
python Pre_Experiment/pre_experiment.py \
    --dataset_path Pre_Experiment/data/pre_pubhealth.json\
    --dataset_name pubhealth \
    --retrieval_ratio 1 \

#For casehold
python Pre_Experiment/pre_experiment.py \
    --dataset_path Pre_Experiment/data/pre_casehold.json\
    --dataset_name casehold \
    --retrieval_ratio 2 \

#For finfact
python Pre_Experiment/pre_experiment.py \
    --dataset_path Pre_Experiment/data/pre_finfact.json\
    --dataset_name finfact \
    --retrieval_ratio 3 \