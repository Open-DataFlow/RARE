from transformers import AutoTokenizer, AutoModelForCausalLM
import argparse

def customize_model(model_name, output_dir):
    # Load original model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    special_tokens = ["[RETRIEVAL]", "[ENTITIES]", "[SEP]", "[REASONING]"]
    
    tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
    model.resize_token_embeddings(len(tokenizer))

    # Save customized version
    tokenizer.save_pretrained(f"{output_dir}/custom_tokenizer")
    model.save_pretrained(f"{output_dir}/custom_model")

    print(f"Customized model and tokenizer saved to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Customize LLM with special tokens')
    parser.add_argument('--model_name_or_path', type=str, required=True, 
                        help='Path to original model')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save customized model')
    
    args = parser.parse_args()
    customize_model(args.model_name, args.output_dir, args.special_tokens)