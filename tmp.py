from transformers import AutoModelForCausalLM

if __name__ == '__main__':
    config_name = 'meta-llama/Llama-3.2-3B'
    # model_name = 'meta-llama/Llama-2-7b-hf'
    # tokenizer = AutoTokenizer.from_pretrained(config_name, trust_remote_code=True)

    model = AutoModelForCausalLM.from_pretrained(config_name)
    layer = model.model.layers[20]

    print(layer.state_dict().keys())
