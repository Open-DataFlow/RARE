import argparse
import json
import numpy as np
from collections import Counter
import spacy
from vllm import LLM, SamplingParams

# Constants
SPECIAL_TOKENS = {
    "retrieval_start": "[RETRIEVAL]",
    "entity_start": "[ENTITIES]",
    "context_sep": "[SEP]",
    "reasoning": "[REASONING]"
}

def parse_args():
    parser = argparse.ArgumentParser(description='Run RARE experiment with configurable parameters')
    parser.add_argument('--model_path_or_name', type=str, default='Pre_Experiment/model/custom_model', help='Path to custom model')
    parser.add_argument('--tokenizer_path', type=str, default='Pre_Experiment/model/custom_tokenizer', help='Path to custom tokenizer')
    parser.add_argument('--dataset_path', type=str, required=True, help='Path to dataset JSON file')
    parser.add_argument('--extractor_path', type=str, default='Pre_Experiment/model/en_core_web_sm', help='Path to spaCy extractor model')
    parser.add_argument('--dataset_name', type=str, required=True, help='Name of dataset for output file')
    parser.add_argument('--retrieval_ratio', type=int, choices=[0,1,2,3,4], required=True, help='Ratio of retrieval content to use (0-4)')
    
    return parser.parse_args()

class VLLMLossCalculator:
    def __init__(self, model_path, tokenizer_path):
        self.model = LLM(
            model=model_path,
            tokenizer=tokenizer_path,
            tensor_parallel_size=8,
            gpu_memory_utilization=0.7,
            trust_remote_code=True,
            max_model_len=8000,
            enable_chunked_prefill=True,
            max_num_seqs=1,
            enforce_eager=True
        )
        self.tokenizer = self.model.get_tokenizer()
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.sampling_params = SamplingParams(
            temperature=0.6,
            top_p=1.0,
            max_tokens=1,
            prompt_logprobs=1,
        )

    def compute_losses(self, samples, extractor, ratio):
        prompts = [self.build_augmented_input(s, ratio) for s in samples]
        return self._get_logprobs(prompts, [s['r'] for s in samples], [s['R_x'] for s in samples], extractor)

    def build_augmented_input(self, sample, ratio):
        if ratio == 0:
            return sample['x']
        n = len(sample['R_x'])
        return f"{sample['x']}\n{sample['R_x'][:n * ratio // 4]}"

    def _get_logprobs(self, prompts, target_texts, R_xs, extractor):
        full_texts = [p + SPECIAL_TOKENS["reasoning"] + t for p, t in zip(prompts, target_texts)]
        outputs = self.model.generate(full_texts, self.sampling_params, use_tqdm=True)
        
        knowledge_losses, reasoning_losses = [], []
        for i, output in enumerate(outputs):
            k = self.extract_entities(R_xs[i], extractor)
            token_list = list(set(self.tokenizer.convert_tokens_to_ids(
                self.tokenizer.tokenize(word)) for word in k)
            token_list = [x for sublist in token_list for x in sublist]
            
            knowledge, reasoning = [], []
            start_extract = False
            for entry in list(output.prompt_logprobs):
                if isinstance(entry, dict):
                    first_token_id = next(iter(entry))
                    if first_token_id == 128259:
                        start_extract = True
                    if start_extract:
                        (knowledge if first_token_id in token_list else reasoning).append(
                            entry[first_token_id].logprob)
            
            knowledge_losses.append(-np.mean(knowledge) if knowledge else 0.0)
            reasoning_losses.append(-np.mean(reasoning) if reasoning else 0.0)
            
        return knowledge_losses, reasoning_losses

    def extract_entities(self, text, nlp):
        doc = nlp(text)
        ents = [ent.text for ent in doc.ents]
        return [ent for ent, _ in Counter(ents).most_common(100)] + [" "+ent for ent in ents]

def run_experiment(args):
    nlp = spacy.load(args.extractor_path)
    calculator = VLLMLossCalculator(args.model_path, args.tokenizer_path)
    
    with open(args.dataset_path) as f:
        test_samples = json.load(f)
    
    knowledge_losses, reasoning_losses = calculator.compute_losses(
        test_samples, nlp, args.retrieval_ratio)
    
    output_data = {
        "samples": [{
            "sample_id": i+1,
            "L_A": k_loss,
            "L_B": r_loss,
            "delta_L": r_loss-k_loss
        } for i, (k_loss, r_loss) in enumerate(zip(knowledge_losses, reasoning_losses))],
        "average_LA": round(np.mean(knowledge_losses), 4),
        "average_LB": round(np.mean(reasoning_losses), 4),
        "average_LB/LA": round(np.mean(reasoning_losses)/np.mean(knowledge_losses), 4)
    }
    
    output_path = f"Pre_Experiment/result/pre_experiment_{args.data_name}_{args.retrieval_ratio}_4.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=4)
    
    print(f"Results saved to {output_path}")

if __name__ == "__main__":
    args = parse_args()
    run_experiment(args)